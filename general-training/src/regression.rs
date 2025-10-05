use std::fmt::Debug;

use burn::{
    module::{AutodiffModule, Module},
    nn::loss::{MseLoss, Reduction},
    prelude::Backend,
    tensor::backend::AutodiffBackend,
    train::{RegressionOutput, TrainOutput, TrainStep, ValidStep},
};
use general_models::{
    SimpleInfer, SimpleTrain, common::PhantomModule, composite::autoencoder::AutoEncoderModel,
};

use crate::batches::AutoEncoderImageBatch;

#[derive(Debug, Module)]
pub struct RegressionTrainableModel<B: Backend, T> {
    pub model: T,
    _phantom: PhantomModule<B>,
}

impl<B: AutodiffBackend, E, D> TrainStep<AutoEncoderImageBatch<B>, RegressionOutput<B>>
    for RegressionTrainableModel<B, AutoEncoderModel<B, E, D>>
where
    AutoEncoderModel<B, E, D>: SimpleTrain<B, 4, 4> + AutodiffModule<B>,
{
    fn step(&self, batch: AutoEncoderImageBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let batch_size = batch.input.dims()[0];
        let actual = self.model.train(batch.input.clone());
        let loss = MseLoss::new().forward(actual.clone(), batch.expected.clone(), Reduction::Mean);
        let item = RegressionOutput::new(
            loss,
            actual.reshape([batch_size as i32, -1]),
            batch.expected.reshape([batch_size as i32, -1]),
        );

        TrainOutput::new(&self.model, item.loss.backward(), item)
    }
}

impl<B: Backend, E, D> ValidStep<AutoEncoderImageBatch<B>, RegressionOutput<B>>
    for RegressionTrainableModel<B, AutoEncoderModel<B, E, D>>
where
    AutoEncoderModel<B, E, D>: SimpleInfer<B, 4, 4>,
{
    fn step(&self, batch: AutoEncoderImageBatch<B>) -> RegressionOutput<B> {
        let batch_size = batch.input.dims()[0];
        let actual = self.model.infer(batch.input.clone());
        let loss = MseLoss::new().forward(actual.clone(), batch.expected.clone(), Reduction::Mean);
        RegressionOutput::new(
            loss,
            actual.reshape([batch_size as i32, -1]),
            batch.expected.reshape([batch_size as i32, -1]),
        )
    }
}
