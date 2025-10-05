use std::fmt::Debug;

use burn::{
    module::{AutodiffModule, Module},
    nn::loss::{MseLoss, Reduction},
    prelude::Backend,
    tensor::backend::AutodiffBackend,
    train::{RegressionOutput, TrainOutput, TrainStep, ValidStep},
};
use general_models::{
    common::PhantomBackend, composite::autoencoder::{vae::VariationalEncoder, AutoEncoderModel}, SimpleInfer, SimpleTrain
};

use crate::batches::AutoEncoderImageBatch;

pub const STANDARD: usize = 0;
pub const NON_STANDARD_1: usize = 1;

#[derive(Debug, Module)]
pub struct RegressionTrainableModel<B: Backend, T, const X: usize> {
    pub model: T,
    _phantom: PhantomBackend<B>,
}

impl<B: Backend, T, const X: usize> From<T> for RegressionTrainableModel<B, T, X> {
    fn from(model: T) -> Self {
        Self {
            model,
            _phantom: Default::default()
        }
    }
}

impl<B: AutodiffBackend, M> TrainStep<AutoEncoderImageBatch<B>, RegressionOutput<B>>
    for RegressionTrainableModel<B, M, STANDARD>
where
    M: SimpleTrain<B, 4, 4> + AutodiffModule<B>,
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

impl<B: Backend, M, const X: usize> ValidStep<AutoEncoderImageBatch<B>, RegressionOutput<B>>
    for RegressionTrainableModel<B, M, X>
where
    M: SimpleInfer<B, 4, 4>,
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

impl<B: AutodiffBackend, E, D> TrainStep<AutoEncoderImageBatch<B>, RegressionOutput<B>>
    for RegressionTrainableModel<B, AutoEncoderModel<B, VariationalEncoder<B, E>, D>, NON_STANDARD_1>
where
    E: SimpleTrain<B, 4, 2> + AutodiffModule<B>,
    D: SimpleTrain<B, 2, 4> + AutodiffModule<B>,
    AutoEncoderModel<B, VariationalEncoder<B, E>, D>: AutodiffModule<B>
{
    fn step(&self, batch: AutoEncoderImageBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let batch_size = batch.input.dims()[0];
        let (actual_mean, actual_logvar) = self.model.encoder.train(batch.input.clone());
        let sampled_latent = self.model.encoder.reparameterize(actual_mean.clone(), actual_logvar.clone());
        let actual_reconstructed = self.model.decoder.train(sampled_latent);
        let mse_loss = MseLoss::new().forward(actual_reconstructed.clone(), batch.expected.clone(), Reduction::Mean);
        
        let kld_element = actual_logvar.clone().exp().add(actual_mean.powf_scalar(2.0)).sub_scalar(1.0).sub(actual_logvar);
        let kld = kld_element.sum().mul_scalar(-0.5);
        
        let total_loss = mse_loss + kld;
        
        // // 2. KL Divergence Loss
        // // logvar.exp() is sigma^2
        // let kld_element = logvar.exp().add(mu.powf(2.0)).sub_scalar(1.0).sub(logvar);
        // let kld = kld_element.sum().mul_scalar(-0.5);
    
        // // 3. Total VAE Loss (Negative ELBO)
        // // Note: If you want a beta-VAE, you would multiply the kld term by beta.
        // bce.add(kld)
        
        let item = RegressionOutput::new(
            total_loss,
            actual_reconstructed.reshape([batch_size as i32, -1]),
            batch.expected.reshape([batch_size as i32, -1]),
        );

        TrainOutput::new(&self.model, item.loss.backward(), item)
    }
}
