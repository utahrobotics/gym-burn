use std::path::{Path, PathBuf};

use burn::config::Config;
use rand::Rng;
use rusqlite::{Connection, Row, params};
use thiserror::Error;

pub mod presets;

#[derive(Debug, Config)]
pub struct SqliteDatasetConfig {
    pub db_file: PathBuf,
    pub get_sql: String,
    pub len_sql: String,
}

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
    conn: Connection,
    get_sql: String,
    len_sql: String,
}

impl SqliteDataset {
    pub fn new(
        db_file: impl AsRef<Path>,
        get_sql: String,
        len_sql: String,
    ) -> rusqlite::Result<Self> {
        let db_file = db_file.as_ref().to_path_buf();

        let conn = Connection::open(&db_file)?;
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

            let stmt = conn.prepare_cached(&len_sql)?;
            assert!(
                stmt.column_names()
                    .iter()
                    .find(|name| **name == "len")
                    .is_some(),
                "len_sql does not output a `len` column"
            );
            assert_eq!(stmt.parameter_count(), 0, "len_sql must have no parameters");
        }

        Ok(Self {
            conn,
            get_sql,
            len_sql,
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
        Self::new(value.db_file, get_sql, value.len_sql).map_err(Into::into)
    }
}
pub trait StatefulBatcher<I, O> {
    fn reset(&mut self);
    fn ingest(&mut self, item: I);
    fn shuffle(&mut self, rng: &mut impl Rng);
    fn finish(&mut self) -> O;
}

impl<I, O, T: StatefulBatcher<I, O>> StatefulBatcher<I, O> for &mut T {
    fn reset(&mut self) {
        T::reset(self);
    }

    fn ingest(&mut self, item: I) {
        T::ingest(self, item);
    }

    fn shuffle(&mut self, rng: &mut impl Rng) {
        T::shuffle(self, rng);
    }

    fn finish(&mut self) -> O {
        T::finish(self)
    }
}

impl SqliteDataset {
    pub fn query<I: FromSqlRow, O>(
        &mut self,
        index: usize,
        limit: usize,
        rng: &mut impl Rng,
        mut batcher: impl StatefulBatcher<I, O>,
    ) -> O {
        batcher.reset();
        let mut stmt = self.conn.prepare_cached(&self.get_sql).unwrap();
        let mut rows = stmt.query(params![index, limit]).unwrap();
        while let Some(row) = rows.next().unwrap() {
            batcher.ingest(I::from(row));
        }
        batcher.shuffle(rng);
        batcher.finish()
    }

    /// Queries the database to determine the number of rows. Since
    /// this makes an SQL query, the result should be cached.
    pub fn len(&self) -> usize {
        self.conn
            .prepare_cached(&self.len_sql)
            .unwrap()
            .query_one((), |row| row.get("len"))
            .unwrap()
    }
}

// impl<I: DeserializeOwned, C: ItemCache<I>> Dataset<I> for SqliteDataset<I, C> {
//     fn get(&self, mut index: usize) -> Option<I> {
//         // sqlite starts at 1
//         index += 1;
//         self.reads.fetch_add(1, Ordering::Relaxed);
//         if let Some(item) = self.cache.get(index) {
//             self.cache_hits.fetch_add(1, Ordering::Relaxed);
//             return Some(item);
//         }
//         let conn = &self.conn;
//         let mut selected_item = None;
//         {
//             let mut stmt = conn.prepare_cached(&self.get_sql).unwrap();
//             let mut rows = stmt.query(params![index]).unwrap();

//             while let Some(row) = rows.next().unwrap() {
//                 let retrieved_index: usize = row.get("row_id").unwrap();

//                 let add_to_cache = !self.cache.noop() && !self.cache.has(retrieved_index);
//                 if retrieved_index != index && !add_to_cache {
//                     continue;
//                 }

//                 let stmt = row.as_ref();
//                 let mut map = serde_json::Map::new();

//                 for column in stmt.column_names() {
//                     let value: rusqlite::types::Value = row.get(column).unwrap();
//                     let value = match value {
//                         rusqlite::types::Value::Null => serde_json::Value::Null,
//                         rusqlite::types::Value::Integer(n) => serde_json::Value::Number(n.into()),
//                         rusqlite::types::Value::Real(n) => {
//                             serde_json::Value::Number(Number::from_f64(n).unwrap())
//                         }
//                         rusqlite::types::Value::Text(s) => serde_json::Value::String(s),
//                         rusqlite::types::Value::Blob(items) => {
//                             serde_json::Value::Array(items.into_iter().map(Into::into).collect())
//                         }
//                     };
//                     map.insert(column.into(), value);
//                 }

//                 let mut item: I = serde_json::from_value(map.into())
//                     .expect("Deserialization should be successful");
//                 if add_to_cache {
//                     item = self.cache.set(retrieved_index, item);
//                 }
//                 if retrieved_index == index {
//                     selected_item = Some(item);
//                 }
//             }
//         }

//         self.conn_queue.push(conn);
//         selected_item
//     }

//     fn len(&self) -> usize {
//         let conn = self.get_conn();
//         let len = conn
//             .prepare_cached(&self.len_sql)
//             .unwrap()
//             .query_one((), |row| row.get("len"))
//             .unwrap();
//         self.conn_queue.push(conn);
//         len
//     }
// }
