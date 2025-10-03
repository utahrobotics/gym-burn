use std::{
    marker::PhantomData,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

use burn::{config::Config, data::dataset::Dataset};
use crossbeam::queue::SegQueue;
use dashmap::DashMap;
use parking_lot::Mutex;
use rusqlite::{Connection, params};
use serde::de::DeserializeOwned;
use serde_json::Number;

#[derive(Debug, Config)]
pub struct SqliteDatasetConfig {
    pub db_file: PathBuf,
    pub get_sql: String,
    pub len_sql: String,
    #[config(default = 32)]
    pub cache_len: usize,
}

pub trait ItemCache<I>: Send + Sync {
    fn noop(&self) -> bool;
    fn has(&self, index: usize) -> bool;
    fn get(&self, index: usize) -> Option<I>;
    fn set(&self, index: usize, item: I) -> I;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NoCache;

impl<I> ItemCache<I> for NoCache {
    fn noop(&self) -> bool {
        true
    }

    fn get(&self, _: usize) -> Option<I> {
        None
    }

    fn set(&self, _: usize, item: I) -> I {
        item
    }

    fn has(&self, _: usize) -> bool {
        false
    }
}

#[derive(Debug, Default)]
pub struct AlwaysCache<I> {
    map: DashMap<usize, I, rustc_hash::FxBuildHasher>,
}

impl<I: Send + Sync + Clone> ItemCache<I> for AlwaysCache<I> {
    fn noop(&self) -> bool {
        false
    }

    fn get(&self, index: usize) -> Option<I> {
        self.map.get(&index).map(|x| x.clone())
    }

    fn set(&self, index: usize, item: I) -> I {
        self.map.insert(index, item.clone());
        item
    }

    fn has(&self, index: usize) -> bool {
        self.map.contains_key(&index)
    }
}

#[derive(Debug)]
pub struct LruCache<I> {
    lru: Option<Mutex<lru::LruCache<usize, I, rustc_hash::FxBuildHasher>>>,
}

impl<I> LruCache<I> {
    pub fn new(max_len: usize) -> Self {
        if let Some(max_len) = NonZeroUsize::new(max_len) {
            Self {
                lru: Some(Mutex::new(lru::LruCache::with_hasher(
                    max_len,
                    Default::default(),
                ))),
            }
        } else {
            Self { lru: None }
        }
    }
}

impl<I: Send + Sync + Clone> ItemCache<I> for LruCache<I> {
    fn noop(&self) -> bool {
        self.lru.is_none()
    }

    fn get(&self, index: usize) -> Option<I> {
        self.lru.as_ref()?.lock().get(&index).map(|x| x.clone())
    }

    fn set(&self, index: usize, item: I) -> I {
        if self.lru.is_none() {
            return item;
        }
        self.lru.as_ref().unwrap().lock().push(index, item.clone());
        item
    }

    fn has(&self, index: usize) -> bool {
        if self.lru.is_none() {
            return false;
        }
        self.lru.as_ref().unwrap().lock().contains(&index)
    }
}

pub struct SqliteDataset<I, C = LruCache<I>> {
    conn_queue: SegQueue<Connection>,
    db_file: PathBuf,
    get_sql: String,
    cache: C,
    len_sql: String,
    cache_hits: AtomicUsize,
    reads: AtomicUsize,
    _phantom: PhantomData<fn() -> I>,
}

impl<I> SqliteDataset<I> {
    pub fn new(
        db_file: impl AsRef<Path>,
        get_sql: String,
        len_sql: String,
        cache_len: usize,
    ) -> rusqlite::Result<Self> {
        let db_file = db_file.as_ref().to_path_buf();
        let conn_queue = SegQueue::new();

        let conn = Connection::open(&db_file)?;
        conn.prepare_cached(&get_sql)?;
        conn.prepare_cached(&len_sql)?;

        Ok(Self {
            conn_queue,
            db_file,
            cache: LruCache::new(cache_len),
            get_sql,
            cache_hits: AtomicUsize::default(),
            reads: AtomicUsize::default(),
            len_sql,
            _phantom: PhantomData,
        })
    }
}

impl<I, C> SqliteDataset<I, C> {
    pub fn get_cache_hits(&self) -> usize {
        self.cache_hits.load(Ordering::Relaxed)
    }

    pub fn get_reads(&self) -> usize {
        self.reads.load(Ordering::Relaxed)
    }

    pub fn reset_cache_hits(&self) {
        self.cache_hits.store(0, Ordering::Relaxed);
    }

    pub fn get_cache(&mut self) -> &mut C {
        &mut self.cache
    }

    pub fn set_cache<C2>(self, cache: C2) -> SqliteDataset<I, C2> {
        SqliteDataset {
            conn_queue: self.conn_queue,
            db_file: self.db_file,
            get_sql: self.get_sql,
            cache,
            reads: AtomicUsize::default(),
            len_sql: self.len_sql,
            cache_hits: AtomicUsize::default(),
            _phantom: PhantomData,
        }
    }

    fn get_conn(&self) -> Connection {
        self.conn_queue.pop().unwrap_or_else(|| {
            Connection::open(&self.db_file).expect("Sqlite database should still be accessible")
        })
    }
}

impl<I> TryFrom<SqliteDatasetConfig> for SqliteDataset<I> {
    type Error = rusqlite::Error;

    fn try_from(value: SqliteDatasetConfig) -> Result<Self, Self::Error> {
        Self::new(value.db_file, value.get_sql, value.len_sql, value.cache_len)
    }
}

impl<I: DeserializeOwned, C: ItemCache<I>> Dataset<I> for SqliteDataset<I, C> {
    fn get(&self, index: usize) -> Option<I> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        if let Some(item) = self.cache.get(index) {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
            return Some(item);
        }
        let conn = self.get_conn();
        let mut selected_item = None;
        {
            let mut stmt = conn.prepare_cached(&self.get_sql).unwrap();
            let mut rows = stmt.query(params![index]).unwrap();

            while let Some(row) = rows.next().unwrap() {
                let retrieved_index: usize = row.get("row_id").unwrap();

                let add_to_cache = !self.cache.noop() && !self.cache.has(retrieved_index);
                if retrieved_index != index && !add_to_cache {
                    continue;
                }

                let stmt = row.as_ref();
                let mut map = serde_json::Map::new();

                for column in stmt.column_names() {
                    let value: rusqlite::types::Value = row.get(column).unwrap();
                    let value = match value {
                        rusqlite::types::Value::Null => serde_json::Value::Null,
                        rusqlite::types::Value::Integer(n) => serde_json::Value::Number(n.into()),
                        rusqlite::types::Value::Real(n) => {
                            serde_json::Value::Number(Number::from_f64(n).unwrap())
                        }
                        rusqlite::types::Value::Text(s) => serde_json::Value::String(s),
                        rusqlite::types::Value::Blob(items) => {
                            serde_json::Value::Array(items.into_iter().map(Into::into).collect())
                        }
                    };
                    map.insert(column.into(), value);
                }

                let mut item: I = serde_json::from_value(map.into())
                    .expect("Deserialization should be successful");
                if add_to_cache {
                    item = self.cache.set(index, item);
                }
                if retrieved_index == index {
                    selected_item = Some(item);
                }
            }
        }

        self.conn_queue.push(conn);
        selected_item
    }

    fn len(&self) -> usize {
        let conn = self.get_conn();
        let len = conn
            .prepare_cached(&self.len_sql)
            .unwrap()
            .query_one((), |row| row.get("len"))
            .unwrap();
        self.conn_queue.push(conn);
        len
    }
}

impl<I: DeserializeOwned, C: ItemCache<I>> Dataset<I> for &SqliteDataset<I, C> {
    fn get(&self, index: usize) -> Option<I> {
        <SqliteDataset<I, C> as Dataset<I>>::get(self, index)
    }

    fn len(&self) -> usize {
        <SqliteDataset<I, C> as Dataset<I>>::len(self)
    }
}
