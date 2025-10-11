use burn::{
    Tensor, lr_scheduler::LrScheduler, module::{AutodiffModule, ModuleDisplay}, tensor::backend::AutodiffBackend
};
use general_dataset::{SqliteDataset, presets::autoencoder::{AutoEncoderImageBatch, AutoEncoderImageBatcher}};
use general_models::{
    composite::autoencoder::{AutoEncoderModel, vae::VariationalEncoderModel},
};
use rand::Rng;

use crate::{
    trainable_models::{
        TrainableModel, VariationalEncoderModelTrainingConfig, apply_gradients::{
            ApplyGradients,
            autoencoder::{AutoEncoderModelPlan, VariationalEncoderModelPlan},
        }
    },
    training_loop::train_epoch,
};

pub fn train_epoch_image_autoencoder<B, E, D, L, S>(
    model: &mut AutoEncoderModel<B, E, D>,
    dataset: &mut SqliteDataset,
    batch_size: usize,
    max_batch_count: usize,
    batcher: &mut AutoEncoderImageBatcher<B>,
    lr_scheduler: &mut impl LrScheduler,
    grads_plan: &mut AutoEncoderModelPlan<E::Plan, D::Plan>,
    rng: &mut (impl Rng + Send),
    loss: &L,
    post_batch: impl FnMut(Tensor<B, 1>, f64) -> bool + Send,
) where
    AutoEncoderModel<B, E, D>: Send,
    E: ApplyGradients<B>,
    D: ApplyGradients<B>,
    E::Plan: Send,
    D::Plan: Send,
    B: AutodiffBackend,
    L: Send + Sync,
    AutoEncoderModel<B, E, D>: TrainableModel<B, AutoEncoderImageBatch<B>, L, S, TrainingConfig = ()> + AutodiffModule<B>
{
    train_epoch(
        model,
        dataset,
        batch_size,
        max_batch_count,
        batcher,
        &(),
        loss,
        lr_scheduler,
        grads_plan,
        rng,
        post_batch,
    );
}

// pub fn train_epoch_image_variational_autoencoder<B, E, D, L, S>(
//     model: &mut AutoEncoderModel<B, VariationalEncoderModel<B, E>, D>,
//     dataset: &mut SqliteDataset,
//     batch_size: usize,
//     max_batch_count: usize,
//     batcher: &mut AutoEncoderImageBatcher<B>,
//     training_config: &VariationalEncoderModelTrainingConfig,
//     lr_scheduler: &mut impl LrScheduler,
//     grads_plan: &mut AutoEncoderModelPlan<VariationalEncoderModelPlan<B, E::Plan>, D::Plan>,
//     rng: &mut (impl Rng + Send),
//     loss: &L,
//     post_batch: impl FnMut(Tensor<B, 1>, f64) -> bool + Send,
// ) where
//     E: ApplyGradients<B>,
//     D: ApplyGradients<B>,
//     E::Plan: Send,
//     D::Plan: Send,
//     B: AutodiffBackend,
//     L: Send + Sync,
//     AutoEncoderModel<B, VariationalEncoderModel<B, E>, D>: TrainableModel<B, AutoEncoderImageBatch<B>, L, S, TrainingConfig = VariationalEncoderModelTrainingConfig> + Send
// {
//     train_epoch(
//         model,
//         dataset,
//         batch_size,
//         max_batch_count,
//         batcher,
//         training_config,
//         loss,
//         lr_scheduler,
//         grads_plan,
//         rng,
//         post_batch,
//     );
// }
