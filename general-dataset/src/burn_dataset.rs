use std::num::NonZeroUsize;

use burn::data::dataset::Dataset;

use crate::{FromSqlRow, SqliteDataset, cache::LocalCache};

pub struct SqliteBurnDataset<I: 'static> {
    sqlite: SqliteDataset,
    cache: &'static LocalCache<I>,
    line_size: NonZeroUsize
}

impl<I: Clone + FromSqlRow> Dataset<I> for SqliteBurnDataset<I> {
    fn get(&self, index: usize) -> Option<I> {
        self.sqlite.get_cached(index, self.cache, self.line_size)
    }

    fn len(&self) -> usize {
        self.sqlite.len()
    }
}