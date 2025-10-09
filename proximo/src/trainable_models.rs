use burn::{Tensor, nn::loss::{MseLoss, Reduction}, tensor::backend::AutodiffBackend};
use general_dataset::presets::autoencoder::AutoEncoderImageBatch;
use general_models::{SimpleTrain, composite::autoencoder::{AutoEncoderModel, vae::VariationalEncoderModel}};
use serde::de::DeserializeOwned;

pub mod apply_gradients;


pub struct Blanket;
pub struct Specialized;


pub trait TrainableModel<B: AutodiffBackend, I, T = Blanket> {
    type Loss;
    type LossConfig: DeserializeOwned;
    type TrainingConfig;

    fn batch_train(&self, batch: I, loss: &Self::Loss, config: &Self::TrainingConfig) -> Tensor<B, 1>;
}

impl<B: AutodiffBackend, E, D> TrainableModel<B, AutoEncoderImageBatch<B>, Blanket> for AutoEncoderModel<B, E, D>
where 
    Self: SimpleTrain<B, 4, 4>
{
    type Loss = MseLoss;
    type LossConfig = ();
    type TrainingConfig = ();

    fn batch_train(&self, batch: AutoEncoderImageBatch<B>, loss: &Self::Loss, (): &Self::TrainingConfig) -> Tensor<B, 1> {
        loss.forward(
            self.train(batch.input),
            batch.expected,
            Reduction::Auto
        )
    }
}

pub struct VariationalEncoderModelTrainingConfig {
    pub kld_weight: f64
}

impl<B: AutodiffBackend, E, D> TrainableModel<B, AutoEncoderImageBatch<B>, Specialized> for AutoEncoderModel<B, VariationalEncoderModel<B, E>, D>
where 
    E: SimpleTrain<B, 4, 2>,
    D: SimpleTrain<B, 2, 4>
{
    type Loss = MseLoss;
    type LossConfig = ();
    type TrainingConfig = VariationalEncoderModelTrainingConfig;

    fn batch_train(&self, batch: AutoEncoderImageBatch<B>, loss: &Self::Loss, config: &Self::TrainingConfig) -> Tensor<B, 1> {
        let (actual_mean, actual_logvar) = self.encoder.train(batch.input.clone());
        let sampled_latent = self
            .encoder
            .reparameterize(actual_mean.clone(), actual_logvar.clone());
        let actual_reconstructed = self.decoder.train(sampled_latent);
        let mse_loss = loss.forward(
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

        mse_loss + (kld * config.kld_weight)
    }
}
