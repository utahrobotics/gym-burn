use std::fmt::Debug;

use burn::{
    module::{AutodiffModule, Module},
    nn::loss::{MseLoss, Reduction},
    prelude::Backend,
    tensor::backend::AutodiffBackend,
    train::{RegressionOutput, TrainOutput, TrainStep, ValidStep},
};
use general_dataset::presets::autoencoder::AutoEncoderImageBatch;
use general_models::{
    SimpleInfer, SimpleTrain,
    common::PhantomBackend,
    composite::autoencoder::{AutoEncoderModel, vae::VariationalEncoder},
};


pub const STANDARD: usize = 0;
pub const SPECIALIZED: usize = 1;

#[derive(Debug)]
pub struct RegressionTrainableModel<B: Backend, T, const X: usize> {
    pub model: T,
    _phantom: PhantomBackend<B>,
}

impl<B: Backend, T, const X: usize> From<T> for RegressionTrainableModel<B, T, X> {
    fn from(model: T) -> Self {
        Self {
            model,
            _phantom: Default::default(),
        }
    }
}

impl<B: AutodiffBackend, M> TrainStep<AutoEncoderImageBatch<B>, RegressionOutput<B>>
    for RegressionTrainableModel<B, M, STANDARD>
where
    M: SimpleTrain<B, 4, 4> + AutodiffModule<B>,
{
    fn step(&self, batch: AutoEncoderImageBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let actual = self.model.train(batch.input.clone());
        let loss = MseLoss::new().forward(actual.clone(), batch.expected.clone(), Reduction::Mean);
        let item = RegressionOutput::new(loss, actual.flatten(1, 3), batch.expected.flatten(1, 3));

        TrainOutput::new(&self.model, item.loss.backward(), item)
    }
}

impl<B: Backend, M, const X: usize> ValidStep<AutoEncoderImageBatch<B>, RegressionOutput<B>>
    for RegressionTrainableModel<B, M, X>
where
    M: SimpleInfer<B, 4, 4>,
{
    fn step(&self, batch: AutoEncoderImageBatch<B>) -> RegressionOutput<B> {
        let actual = self.model.infer(batch.input.clone());
        let loss = MseLoss::new().forward(actual.clone(), batch.expected.clone(), Reduction::Mean);
        RegressionOutput::new(loss, actual.flatten(1, 3), batch.expected.flatten(1, 3))
    }
}

impl<B: AutodiffBackend, E, D> TrainStep<AutoEncoderImageBatch<B>, RegressionOutput<B>>
    for RegressionTrainableModel<B, AutoEncoderModel<B, VariationalEncoder<B, E>, D>, SPECIALIZED>
where
    E: SimpleTrain<B, 4, 2> + AutodiffModule<B>,
    D: SimpleTrain<B, 2, 4> + AutodiffModule<B>,
    AutoEncoderModel<B, VariationalEncoder<B, E>, D>: AutodiffModule<B>,
{
    fn step(&self, batch: AutoEncoderImageBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let (actual_mean, actual_logvar) = self.model.encoder.train(batch.input.clone());
        let sampled_latent = self
            .model
            .encoder
            .reparameterize(actual_mean.clone(), actual_logvar.clone());
        let actual_reconstructed = self.model.decoder.train(sampled_latent);
        let mse_loss = MseLoss::new().forward(
            actual_reconstructed.clone(),
            batch.expected.clone(),
            Reduction::Mean,
        );

        let kld_element = actual_mean
            .powf_scalar(2.0)
            .add(actual_logvar.clone().exp())
            .sub_scalar(1.0)
            .sub(actual_logvar);
        let kld = kld_element.sum_dim(1).mean().mul_scalar(0.5);

        let total_loss = mse_loss + (kld * self.model.encoder.get_kld_weight());

        let item = RegressionOutput::new(
            total_loss,
            actual_reconstructed.flatten(1, 3),
            batch.expected.flatten(1, 3),
        );

        TrainOutput::new(&self.model, item.loss.backward(), item)
    }
}
