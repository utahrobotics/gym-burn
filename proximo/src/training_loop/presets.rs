use burn::{Tensor, lr_scheduler::LrScheduler, module::ModuleDisplay, nn::loss::MseLoss, tensor::backend::AutodiffBackend};
use general_dataset::{SqliteDataset, presets::autoencoder::AutoEncoderImageBatcher};
use general_models::{SimpleTrain, composite::autoencoder::{AutoEncoderModel, vae::VariationalEncoderModel}};
use rand::Rng;

use crate::{
    trainable_models::{VariationalEncoderModelTrainingConfig, apply_gradients::{ApplyGradients, autoencoder::{AutoEncoderModelPlan, VariationalEncoderModelPlan}}},
    training_loop::train_epoch,
};

pub fn train_epoch_image_autoencoder<B, E, D, S>(
    model: &mut AutoEncoderModel<B, E, D>,
    dataset: &mut SqliteDataset,
    dataset_len: usize,
    batch_size: usize,
    max_batch_count: usize,
    batcher: &mut AutoEncoderImageBatcher<B>,
    lr_scheduler: &mut impl LrScheduler,
    grads_plan: &mut AutoEncoderModelPlan<E::Plan, D::Plan>,
    rng: &mut (impl Rng + Send),
    post_batch: impl FnMut(Tensor<B, 1>, f64) + Send,
) where
    AutoEncoderModel<B, E, D>: Send,
    E: ApplyGradients<B> + SimpleTrain<B, 4, 2> + ModuleDisplay,
    D: ApplyGradients<B> + SimpleTrain<B, 2, 4> + ModuleDisplay,
    E::Plan: Send,
    D::Plan: Send,
    B: AutodiffBackend,
{
    train_epoch(
        model,
        dataset,
        dataset_len,
        batch_size,
        max_batch_count,
        batcher,
        &(),
        &MseLoss::new(),
        lr_scheduler,
        grads_plan,
        rng,
        post_batch,
    );
}

pub fn train_epoch_image_variational_autoencoder<B, E, D, S>(
    model: &mut AutoEncoderModel<B, VariationalEncoderModel<B, E>, D>,
    dataset: &mut SqliteDataset,
    dataset_len: usize,
    batch_size: usize,
    max_batch_count: usize,
    batcher: &mut AutoEncoderImageBatcher<B>,
    training_config: &VariationalEncoderModelTrainingConfig,
    lr_scheduler: &mut impl LrScheduler,
    grads_plan: &mut AutoEncoderModelPlan<VariationalEncoderModelPlan<B, E::Plan>, D::Plan>,
    rng: &mut (impl Rng + Send),
    post_batch: impl FnMut(Tensor<B, 1>, f64) + Send,
) where
    AutoEncoderModel<B, E, D>: Send,
    E: ApplyGradients<B> + SimpleTrain<B, 4, 2> + ModuleDisplay,
    D: ApplyGradients<B> + SimpleTrain<B, 2, 4> + ModuleDisplay,
    E::Plan: Send,
    D::Plan: Send,
    B: AutodiffBackend,
{
    train_epoch(
        model,
        dataset,
        dataset_len,
        batch_size,
        max_batch_count,
        batcher,
        training_config,
        &MseLoss::new(),
        lr_scheduler,
        grads_plan,
        rng,
        post_batch,
    );
}
