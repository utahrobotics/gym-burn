use burn::{data::dataloader::batcher::Batcher, prelude::Backend};
use crossbeam::queue::SegQueue;
use parking_lot::Mutex;
use std::num::NonZeroUsize;

use burn::data::dataset::Dataset;

use crate::{FromSqlRow, SqliteDataset, cache::LocalCache};

use crate::StatefulBatcher;

pub struct BurnBatcher<T> {
    batcher: Mutex<T>,
    extra_batchers: SegQueue<T>,
    new_batcher: Box<dyn Fn() -> T + Send + Sync>,
}

impl<T> BurnBatcher<T> {
    pub fn new(new_batcher: impl Fn() -> T + Send + Sync + 'static) -> Self {
        Self {
            batcher: Mutex::new(new_batcher()),
            new_batcher: Box::new(new_batcher),
            extra_batchers: SegQueue::new(),
        }
    }

    fn with_batcher<O>(&self, f: impl FnOnce(&mut T) -> O) -> O {
        match self.batcher.try_lock() {
            Some(mut x) => f(&mut x),
            None => {
                let mut batcher = self.extra_batchers.pop().unwrap_or_else(&self.new_batcher);
                let result = f(&mut batcher);
                self.extra_batchers.push(batcher);
                result
            }
        }
    }
}

impl<B: Backend, I, O, T: StatefulBatcher<I, O> + Send> Batcher<B, I, O> for BurnBatcher<T> {
    fn batch(&self, items: Vec<I>, _device: &<B as Backend>::Device) -> O {
        self.with_batcher(|batcher| {
            batcher.reset();
            for item in items {
                batcher.ingest(item);
            }
            batcher.finish()
        })
    }
}

pub struct SqliteBurnDataset<I: 'static> {
    sqlite: SqliteDataset,
    cache: &'static LocalCache<I>,
    line_size: NonZeroUsize,
}

impl<I> SqliteBurnDataset<I> {
    pub fn new(
        sqlite: SqliteDataset,
        cache: &'static LocalCache<I>,
        line_size: NonZeroUsize,
    ) -> Self {
        Self {
            sqlite,
            cache,
            line_size,
        }
    }
}

impl<I: Clone + FromSqlRow> Dataset<I> for SqliteBurnDataset<I> {
    fn get(&self, index: usize) -> Option<I> {
        self.sqlite.get_cached(index, self.cache, self.line_size)
    }

    fn len(&self) -> usize {
        self.sqlite.len()
    }
}
