use std::{marker::PhantomData, path::{Path, PathBuf}};

use burn::{config::Config, data::dataset::Dataset};
use crossbeam::queue::SegQueue;
use rusqlite::{params, Connection, OptionalExtension};
use serde::de::DeserializeOwned;
use serde_json::Number;

#[derive(Debug, Config)]
pub struct SqliteDatasetConfig {
    pub db_file: PathBuf,
    pub get_sql: String,
    pub len_sql: String,
}

pub struct SqliteDataset<I> {
    conn_queue: SegQueue<Connection>,
    db_file: PathBuf,
    get_sql: String,
    len_sql: String,
    _phantom: PhantomData<fn() -> I>
}

impl<I> SqliteDataset<I> {
    pub fn new(db_file: impl AsRef<Path>, get_sql: String, len_sql: String) -> rusqlite::Result<Self> {
        let db_file = db_file.as_ref().to_path_buf();
        let conn_queue = SegQueue::new();
        
        let conn = Connection::open(&db_file)?;
        conn.prepare_cached(&get_sql)?;
        conn.prepare_cached(&len_sql)?;
        
        Ok(Self {
            conn_queue,
            db_file,
            get_sql,
            len_sql,
            _phantom: PhantomData
        })
    }
    
    fn get_conn(&self) -> Connection {
        self.conn_queue.pop().unwrap_or_else(|| Connection::open(&self.db_file).expect("Sqlite database should still be accessible"))
    }
}
impl<I> TryFrom<SqliteDatasetConfig> for SqliteDataset<I> {
    type Error = rusqlite::Error;

    fn try_from(value: SqliteDatasetConfig) -> Result<Self, Self::Error> {
        Self::new(value.db_file, value.get_sql, value.len_sql)
    }
}

impl<I: DeserializeOwned> Dataset<I> for SqliteDataset<I> {
    fn get(&self, index: usize) -> Option<I> {
        let conn = self.get_conn();
        let item = conn
            .prepare_cached(&self.get_sql)
            .unwrap()
            .query_one(params![index], |row| {
                let stmt = row.as_ref();
                let mut map = serde_json::Map::new();
                for column in stmt.column_names() {
                    let value: rusqlite::types::Value = row.get(column).unwrap();
                    let value = match value {
                        rusqlite::types::Value::Null => serde_json::Value::Null,
                        rusqlite::types::Value::Integer(n) => serde_json::Value::Number(n.into()),
                        rusqlite::types::Value::Real(n) => serde_json::Value::Number(Number::from_f64(n).unwrap()),
                        rusqlite::types::Value::Text(s) => serde_json::Value::String(s),
                        rusqlite::types::Value::Blob(items) => serde_json::Value::Array(items.into_iter().map(Into::into).collect()),
                    };
                    map.insert(
                        column.into(),
                        value
                    );
                }
                Ok(serde_json::from_value(map.into()).expect("Deserialization should be successful"))
            })
            .optional()
            .unwrap();
        self.conn_queue.push(conn);
        item
    }

    fn len(&self) -> usize {
        let conn = self.get_conn();
        let len = conn.prepare_cached(&self.len_sql).unwrap().query_one((), |row| row.get("len")).unwrap();
        self.conn_queue.push(conn);
        len
    }
}