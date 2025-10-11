use burn::{
    Tensor,
    nn::loss::{BinaryCrossEntropyLoss, MseLoss, Reduction},
    prelude::Backend,
    tensor::backend::AutodiffBackend,
};
use general_dataset::presets::autoencoder::AutoEncoderImageBatch;
use general_models::{
    SimpleInfer, SimpleTrain,
    composite::autoencoder::{AutoEncoderModel, vae::VariationalEncoderModel},
};

pub mod apply_gradients;

pub struct Blanket;
pub struct Specialized;

pub trait ValidatableModel<B: Backend, I, L, T = Blanket> {
    fn batch_valid(&self, batch: I, loss: &L) -> Tensor<B, 1>;
}

pub trait TrainableModel<B: AutodiffBackend, I, L, T = Blanket> {
    type TrainingConfig;

    fn batch_train(&self, batch: I, loss: &L, config: &Self::TrainingConfig) -> Tensor<B, 1>;
}

impl<B: Backend, E, D> ValidatableModel<B, AutoEncoderImageBatch<B>, MseLoss>
    for AutoEncoderModel<B, E, D>
where
    Self: SimpleInfer<B, 4, 4>,
{
    fn batch_valid(&self, batch: AutoEncoderImageBatch<B>, loss: &MseLoss) -> Tensor<B, 1> {
        loss.forward(self.infer(batch.input), batch.expected, Reduction::Auto)
    }
}

impl<B: AutodiffBackend, E, D> TrainableModel<B, AutoEncoderImageBatch<B>, MseLoss>
    for AutoEncoderModel<B, E, D>
where
    Self: SimpleTrain<B, 4, 4>,
{
    type TrainingConfig = ();

    fn batch_train(
        &self,
        batch: AutoEncoderImageBatch<B>,
        loss: &MseLoss,
        (): &Self::TrainingConfig,
    ) -> Tensor<B, 1> {
        loss.forward(self.train(batch.input), batch.expected, Reduction::Auto)
    }
}

impl<B: Backend, E, D> ValidatableModel<B, AutoEncoderImageBatch<B>, BinaryCrossEntropyLoss<B>>
    for AutoEncoderModel<B, E, D>
where
    Self: SimpleInfer<B, 4, 4>,
{
    fn batch_valid(
        &self,
        batch: AutoEncoderImageBatch<B>,
        loss: &BinaryCrossEntropyLoss<B>,
    ) -> Tensor<B, 1> {
        loss.forward::<1>(
            self.infer(batch.input).flatten(0, 3),
            batch.expected.round().int().flatten(0, 3),
        )
    }
}

impl<B: AutodiffBackend, E, D>
    TrainableModel<B, AutoEncoderImageBatch<B>, BinaryCrossEntropyLoss<B>>
    for AutoEncoderModel<B, E, D>
where
    Self: SimpleTrain<B, 4, 4>,
{
    type TrainingConfig = ();

    fn batch_train(
        &self,
        batch: AutoEncoderImageBatch<B>,
        loss: &BinaryCrossEntropyLoss<B>,
        (): &Self::TrainingConfig,
    ) -> Tensor<B, 1> {
        loss.forward::<1>(
            self.train(batch.input).flatten(0, 3),
            batch.expected.round().int().flatten(0, 3),
        )
    }
}

pub struct VariationalEncoderModelTrainingConfig {
    pub kld_weight: f64,
}

impl<B: Backend, E, D> ValidatableModel<B, AutoEncoderImageBatch<B>, MseLoss, Specialized>
    for AutoEncoderModel<B, VariationalEncoderModel<B, E>, D>
where
    Self: SimpleInfer<B, 4, 4>,
{
    fn batch_valid(&self, batch: AutoEncoderImageBatch<B>, loss: &MseLoss) -> Tensor<B, 1> {
        loss.forward(self.infer(batch.input), batch.expected, Reduction::Auto)
    }
}

impl<B: AutodiffBackend, E, D> TrainableModel<B, AutoEncoderImageBatch<B>, MseLoss, Specialized>
    for AutoEncoderModel<B, VariationalEncoderModel<B, E>, D>
where
    E: SimpleTrain<B, 4, 2>,
    D: SimpleTrain<B, 2, 4>,
{
    type TrainingConfig = VariationalEncoderModelTrainingConfig;

    fn batch_train(
        &self,
        batch: AutoEncoderImageBatch<B>,
        loss: &MseLoss,
        config: &Self::TrainingConfig,
    ) -> Tensor<B, 1> {
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
