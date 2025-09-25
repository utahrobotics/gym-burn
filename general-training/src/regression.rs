use std::{fmt::Debug, marker::PhantomData};

use burn::{
    module::{AutodiffModule, Module}, nn::loss::{MseLoss, Reduction}, prelude::Backend, tensor::backend::AutodiffBackend, train::{RegressionOutput, TrainOutput, TrainStep, ValidStep}, Tensor
};
use general_models::{autoencoder::{LinearImageAutoEncoder, SimpleAutoEncoder}, SimpleForwardable};

use crate::{TrainableModelRecord, batches::AutoEncoderImageBatch};

pub trait RegressionTrainable<B: Backend, const N_I: usize, const N_O: usize>:
    Module<B>
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
    B: Backend,
    Self: SimpleForwardable<B, N_I, N_I>,
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

pub struct RegressionTrainableModel<T, P = ()> {
    pub model: T,
    pub phantom: PhantomData<fn() -> P>,
}

impl<T: Debug, P> Debug for RegressionTrainableModel<T, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.model.fmt(f)
    }
}

impl<T: Clone, P> Clone for RegressionTrainableModel<T, P> {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            phantom: self.phantom,
        }
    }
}

impl<B: Backend, T: Module<B>, P> Module<B> for RegressionTrainableModel<T, P> {
    type Record = TrainableModelRecord<T::Record, P>;

    fn collect_devices(&self, devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
        self.model.collect_devices(devices)
    }

    fn fork(self, device: &<B as Backend>::Device) -> Self {
        Self {
            model: self.model.fork(device),
            phantom: PhantomData,
        }
    }

    fn to_device(self, device: &<B as Backend>::Device) -> Self {
        Self {
            model: self.model.to_device(device),
            phantom: PhantomData,
        }
    }

    fn visit<Visitor: burn::module::ModuleVisitor<B>>(&self, visitor: &mut Visitor) {
        self.model.visit(visitor);
    }

    fn map<Mapper: burn::module::ModuleMapper<B>>(self, mapper: &mut Mapper) -> Self {
        Self {
            model: self.model.map(mapper),
            phantom: PhantomData,
        }
    }

    fn load_record(self, record: Self::Record) -> Self {
        Self {
            model: self.model.load_record(record.model_record),
            phantom: PhantomData,
        }
    }

    fn into_record(self) -> Self::Record {
        TrainableModelRecord {
            model_record: self.model.into_record(),
            phantom: PhantomData,
        }
    }
}

impl<B: AutodiffBackend, T: AutodiffModule<B>, P> AutodiffModule<B>
    for RegressionTrainableModel<T, P>
{
    type InnerModule = RegressionTrainableModel<T::InnerModule>;

    fn valid(&self) -> Self::InnerModule {
        RegressionTrainableModel { model: self.model.valid(), phantom: PhantomData }
    }
}

impl<B: AutodiffBackend> TrainStep<AutoEncoderImageBatch<B>, RegressionOutput<B>>
    for RegressionTrainableModel<LinearImageAutoEncoder<B>>
{
    fn step(&self, batch: AutoEncoderImageBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.model.forward_regression(batch.input, batch.expected);

        TrainOutput::new(&self.model, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<AutoEncoderImageBatch<B>, RegressionOutput<B>>
    for RegressionTrainableModel<LinearImageAutoEncoder<B>>
{
    fn step(&self, batch: AutoEncoderImageBatch<B>) -> RegressionOutput<B> {
        self.model.forward_regression(batch.input, batch.expected)
    }
}
