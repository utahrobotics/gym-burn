#![recursion_limit = "256"]
use std::{fmt::Debug, marker::PhantomData};

use burn::{prelude::Backend, record::Record};
use serde::{Deserialize, Serialize};

#[cfg(feature = "app")]
pub mod app;
pub mod batches;
pub mod dataset;
pub mod regression;
pub mod training_loop;

pub struct TrainableModelRecord<T, P> {
    pub model_record: T,
    pub phantom: PhantomData<fn() -> P>,
}

impl<T: Debug, P> Debug for TrainableModelRecord<T, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.model_record.fmt(f)
    }
}

impl<T: Clone, P> Clone for TrainableModelRecord<T, P> {
    fn clone(&self) -> Self {
        Self {
            model_record: self.model_record.clone(),
            phantom: self.phantom,
        }
    }
}

#[derive(Deserialize, Serialize)]
pub struct TrainableModelRecordItem<T, P> {
    pub model_item: T,
    pub phantom: PhantomData<fn() -> P>,
}

impl<T: Debug, P> Debug for TrainableModelRecordItem<T, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.model_item.fmt(f)
    }
}

impl<T: Clone, P> Clone for TrainableModelRecordItem<T, P> {
    fn clone(&self) -> Self {
        Self {
            model_item: self.model_item.clone(),
            phantom: self.phantom,
        }
    }
}

impl<B: Backend, T: Record<B>, P> Record<B> for TrainableModelRecord<T, P> {
    type Item<S: burn::record::PrecisionSettings> = TrainableModelRecordItem<T::Item<S>, P>;

    fn into_item<S: burn::record::PrecisionSettings>(self) -> Self::Item<S> {
        TrainableModelRecordItem {
            model_item: self.model_record.into_item(),
            phantom: PhantomData,
        }
    }

    fn from_item<S: burn::record::PrecisionSettings>(
        item: Self::Item<S>,
        device: &<B as Backend>::Device,
    ) -> Self {
        Self {
            model_record: Record::from_item(item.model_item, device),
            phantom: PhantomData,
        }
    }
}

pub struct ConstUsizeTuple<const N_I: usize, const N_D: usize>;
