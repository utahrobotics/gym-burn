use std::{fmt::Debug, marker::PhantomData};

use burn::{
    Tensor,
    module::AutodiffModule,
    nn::loss::{MseLoss, Reduction},
    prelude::Backend,
    record::Record,
    tensor::backend::AutodiffBackend,
    train::RegressionOutput,
};
use general_models::{SimpleForwardable, autoencoder::SimpleAutoEncoder};
use serde::{Deserialize, Serialize};

pub mod batches;
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

pub trait RegressionTrainable<B: AutodiffBackend, const N_I: usize, const N_O: usize>:
    AutodiffModule<B>
{
    fn forward_regression(
        &self,
        input: Tensor<B, N_I>,
        expected: Tensor<B, N_O>,
    ) -> RegressionOutput<B>;
}

impl<B, const N_I: usize, const N_D: usize, E, D> RegressionTrainable<B, N_I, N_I>
    for SimpleAutoEncoder<B, E, D, N_I, N_D>
where
    B: AutodiffBackend,
    Self: SimpleForwardable<B, N_I, N_I> + AutodiffModule<B>,
{
    fn forward_regression(
        &self,
        input: Tensor<B, N_I>,
        expected: Tensor<B, N_I>,
    ) -> RegressionOutput<B> {
        let batch_size = input.dims()[0];
        let actual = self.forward(input.clone());
        let loss = MseLoss::new().forward(actual.clone(), expected.clone(), Reduction::Mean);

        RegressionOutput::new(
            loss,
            actual.reshape([batch_size as i32, -1]),
            expected.reshape([batch_size as i32, -1]),
        )
    }
}

pub struct ConstUsizeTuple<const N_I: usize, const N_D: usize>;
