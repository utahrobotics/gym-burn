use std::{cell::{OnceCell, RefCell}, num::NonZeroUsize, thread::LocalKey};

use lru::LruCache;
use rusqlite::params;
use rustc_hash::FxBuildHasher;

use crate::{FromSqlRow, SqliteDataset};

pub struct Cache<I> {
    cache: LruCache<usize, I, FxBuildHasher>
}

impl<I> Cache<I> {
    pub fn push(&mut self, index: usize, item: I) {
        self.cache.push(index, item);
    }

    pub fn peek(&self, index: usize) -> Option<&I> {
        self.cache.peek(&index)
    }
}

pub type LocalCache<I> = LocalKey<OnceCell<RefCell<Cache<I>>>>;

impl SqliteDataset {
    pub fn get_cached<I: Clone + FromSqlRow>(&self, index: usize, cache: &'static LocalCache<I>, line_size: NonZeroUsize) -> Option<I> {
        cache.with(|cell| {
            let cell = cell.get_or_init(|| RefCell::new(Cache{ cache: LruCache::with_hasher(DEFAULT_CACHE_SIZE, Default::default())}));
            let mut cache = cell.borrow_mut();
            if let Some(x) = cache.peek(index) {
                Some(x.clone())
            } else {
                self.with_conn(|conn| {
                    let mut stmt = conn.prepare_cached(&self.get_sql).unwrap();
                    let mut rows = stmt.query(params![index, line_size]).unwrap();
                    while let Some(row) = rows.next().unwrap() {
                        cache.push(row.get("row_id").unwrap(), I::from(row));
                    }
                });
                cache.peek(index).cloned()
            }
        })
    }
}

pub const DEFAULT_CACHE_SIZE: NonZeroUsize = NonZeroUsize::new(256).unwrap();

#[macro_export]
macro_rules! def_cache {
    ($vis: vis $ident: ident $item: ty) => {
        thread_local! {
            $vis static $ident: std::cell::OnceCell<std::cell::RefCell<$crate::cache::Cache<$item>>> = const { std::cell::OnceCell::new() };
        }
    };
}

pub fn initialize_cache<I>(cache: &'static LocalCache<I>, size: NonZeroUsize) {
    cache.with(|x| {
        x.get_or_init(|| {
            RefCell::new(Cache { cache: LruCache::with_hasher(size, Default::default()) })
        });
    });
}

def_cache!(pub TEST u32);