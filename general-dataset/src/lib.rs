use std::path::{Path, PathBuf};

use burn::{config::Config};
use crossbeam::queue::SegQueue;
use parking_lot::Mutex;
use rand::Rng;
use rusqlite::{Connection, Row, params};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[cfg(feature = "burn_dataset")]
pub mod burn_dataset;
#[cfg(feature = "cache")]
pub mod cache;
pub mod presets;

#[derive(Debug, Serialize, Deserialize)]
pub struct SqliteDatasetConfig {
    pub db_file: PathBuf,
    pub get_sql: String,
    pub len_sql: String,
    #[serde(default)]
    pub shuffle_sqls: Vec<String>,
}

impl Config for SqliteDatasetConfig {}

pub trait FromSqlRow {
    fn from(row: &Row) -> Self;
}

#[macro_export]
macro_rules! sql_object {
    (
        pub struct $name: ident {
            $(
                pub $field_name: ident : $ty: ty,
            )+
        }
    ) => {
        #[derive(Debug, Clone)]
        pub struct $name {
            $(
                pub $field_name : $ty,
            )+
        }

        impl crate::FromSqlRow for $name {
            fn from(row: &rusqlite::Row) -> Self {
                Self {
                    $(
                        $field_name: row.get(stringify!($field_name)).unwrap(),
                    )+
                }
            }
        }
    };
}

pub struct SqliteDataset {
    conn: Mutex<Connection>,
    extra_conns: SegQueue<Connection>,
    db_file: PathBuf,
    get_sql: String,
    shuffle_sqls: Vec<String>,
    len: usize,
}

impl SqliteDataset {
    pub fn new(
        db_file: impl AsRef<Path>,
        get_sql: String,
        len_sql: String,
        shuffle_sqls: Vec<String>
    ) -> rusqlite::Result<Self> {
        let db_file = db_file.as_ref().to_path_buf();

        let conn = Connection::open(&db_file)?;
        let len;
        {
            let stmt = conn.prepare_cached(&get_sql)?;
            assert!(
                stmt.column_names()
                    .iter()
                    .find(|name| **name == "row_id")
                    .is_some(),
                "get_sql does not output a `row_id` column"
            );
            assert_eq!(
                stmt.parameter_count(),
                2,
                "get_sql must have exactly 2 parameters: row_id and count"
            );

            let mut stmt = conn.prepare_cached(&len_sql)?;
            assert!(
                stmt.column_names()
                    .iter()
                    .find(|name| **name == "len")
                    .is_some(),
                "len_sql does not output a `len` column"
            );
            assert_eq!(stmt.parameter_count(), 0, "len_sql must have no parameters");
            len = stmt.query_one((), |row| row.get("len")).unwrap();

            for sql in &shuffle_sqls {
                let stmt = conn.prepare_cached(sql)?;
                assert_eq!(stmt.parameter_count(), 0, "each statement in shuffle_sqls must not accept parameters");
            }
        }

        Ok(Self {
            conn: Mutex::new(conn),
            get_sql,
            len,
            db_file,
            shuffle_sqls,
            extra_conns: SegQueue::new(),
        })
    }
}

#[derive(Error, Debug)]
pub enum SqliteConfigError {
    #[error("Failed to read the SQL file {path}: {error}")]
    ReadSQLFile { path: String, error: std::io::Error },
    #[error("{0}")]
    RusqliteError(#[from] rusqlite::Error),
}

impl TryFrom<SqliteDatasetConfig> for SqliteDataset {
    type Error = SqliteConfigError;

    fn try_from(value: SqliteDatasetConfig) -> Result<Self, Self::Error> {
        let get_sql = if let Some(path) = value.get_sql.strip_prefix('@') {
            std::fs::read_to_string(path).map_err(|error| SqliteConfigError::ReadSQLFile {
                path: path.into(),
                error,
            })?
        } else {
            value.get_sql
        };
        Self::new(value.db_file, get_sql, value.len_sql, value.shuffle_sqls).map_err(Into::into)
    }
}
pub trait StatefulBatcher<I, O> {
    fn reset(&mut self);
    fn ingest(&mut self, item: I);
    fn finish(&mut self) -> O;
}

impl<I, O, T: StatefulBatcher<I, O>> StatefulBatcher<I, O> for &mut T {
    fn reset(&mut self) {
        T::reset(self);
    }

    fn ingest(&mut self, item: I) {
        T::ingest(self, item);
    }

    fn finish(&mut self) -> O {
        T::finish(self)
    }
}

impl SqliteDataset {
    pub fn query<I: FromSqlRow, O>(
        &self,
        index: usize,
        limit: usize,
        mut batcher: impl StatefulBatcher<I, O>,
    ) -> O {
        batcher.reset();
        self.with_conn(|conn| {
            let mut stmt = conn.prepare_cached(&self.get_sql).unwrap();
            let mut rows = stmt.query(params![index, limit]).unwrap();
            while let Some(row) = rows.next().unwrap() {
                batcher.ingest(I::from(row));
            }
        });
        batcher.finish()
    }

    pub fn shuffle(&self) {
        if self.shuffle_sqls.is_empty() {
            return;
        }
        self.with_conn(|conn| {
            let tx = conn.transaction().unwrap();
            for sql in &self.shuffle_sqls {
                tx.prepare_cached(sql).unwrap().execute(()).unwrap();
            }
            tx.commit().unwrap();
        });
    }

    pub fn get_batch_count(&self, batch_size: usize) -> usize {
        self.len.div_ceil(batch_size)
    }

    pub fn pick_random<I: FromSqlRow>(&self, rng: &mut impl Rng) -> I {
        self.get(rng.random_range(0..self.len))
    }

    pub fn with_conn<T>(&self, f: impl FnOnce(&mut Connection) -> T) -> T {
        match self.conn.try_lock() {
            Some(mut x) => f(&mut x),
            None => {
                let mut conn = self.extra_conns.pop().unwrap_or_else(|| {
                    Connection::open(&self.db_file).expect("Expected db file to be valid")
                });
                let result = f(&mut conn);
                self.extra_conns.push(conn);
                result
            }
        }
    }

    pub fn get<I: FromSqlRow>(&self, index: usize) -> I {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare_cached(&self.get_sql).unwrap();
            stmt.query_one(params![index, 1], |row| Ok(I::from(row)))
                .unwrap()
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }
}
