use std::sync::Mutex;

use burn::{Tensor, lr_scheduler::LrScheduler, prelude::Backend, tensor::backend::AutodiffBackend};
use general_dataset::{FromSqlRow, SqliteDataset, StatefulBatcher};
use rand::{Rng, seq::SliceRandom};
use rayon::join;

use crate::trainable_models::{TrainableModel, ValidatableModel, apply_gradients::ApplyGradients};

pub mod presets;

// pub fn train_batch<B, M, Row, Item, S>(
//     model: &mut M,
//     dataset: &mut SqliteDataset,
//     index: usize,
//     batch_size: usize,
//     batcher: &mut impl StatefulBatcher<Row, Item>,
//     training_config: &M::TrainingConfig,
//     loss: &M::Loss,
//     lr: f64,
//     grads_plan: &mut M::Plan,
//     rng: &mut impl Rng,
// ) -> Tensor<B, 1>
// where
//     B: AutodiffBackend,
//     Row: FromSqlRow,
//     M: TrainableModel<B, Item, S> + ApplyGradients<B>,
// {
//     let batch = dataset.query(index, batch_size, rng, batcher);
//     let loss = model.batch_train(batch, loss, training_config);
//     let mut grads = loss.backward();
//     model.apply_gradients(lr, &mut grads, grads_plan);
//     loss
// }

// pub fn train_epoch<B, M, Row, Item, S>(
//     model: &mut M,
//     dataset: &mut SqliteDataset,
//     dataset_len: usize,
//     batch_size: usize,
//     batcher: &mut impl StatefulBatcher<Row, Item>,
//     training_config: &M::TrainingConfig,
//     loss: &M::Loss,
//     lr_scheduler: &mut impl LrScheduler,
//     grads_plan: &mut M::Plan,
//     rng: &mut impl Rng,
//     mut post_batch: impl FnMut(&mut M, Tensor<B, 1>, f64),
// ) where
//     B: AutodiffBackend,
//     Row: FromSqlRow,
//     M: TrainableModel<B, Item, S> + ApplyGradients<B>,
// {
//     let mut block_indices: Vec<_> = (0..(dataset_len.div_ceil(batch_size)))
//         .map(|x| x * batch_size)
//         .collect();
//     block_indices.shuffle(rng);
//     for index in block_indices {
//         let lr = lr_scheduler.step();
//         let loss = train_batch(
//             model,
//             dataset,
//             index,
//             batch_size,
//             batcher,
//             training_config,
//             loss,
//             lr,
//             grads_plan,
//             rng,
//         );
//         post_batch(model, loss, lr);
//     }
// }

pub fn train_epoch<B, M, Row, Item, S>(
    model: &mut M,
    dataset: &mut SqliteDataset,
    batch_size: usize,
    max_batch_count: usize,
    batcher: &mut (impl StatefulBatcher<Row, Item> + Send),
    training_config: &M::TrainingConfig,
    loss: &M::Loss,
    lr_scheduler: &mut impl LrScheduler,
    grads_plan: &mut M::Plan,
    rng: &mut (impl Rng + Send),
    mut post_batch: impl FnMut(Tensor<B, 1>, f64) + Send,
) where
    M: Send,
    B: AutodiffBackend,
    Row: FromSqlRow,
    M: TrainableModel<B, Item, S> + ApplyGradients<B>,
    Item: Send,
    M::Loss: Send + Sync,
    M::Plan: Send,
    M::TrainingConfig: Sync,
{
    let mut block_indices: Vec<_> = (0..dataset.get_batch_count(batch_size))
        .map(|x| x * batch_size)
        .collect();
    block_indices.shuffle(rng);
    block_indices.truncate(max_batch_count);
    let mut block_indices = block_indices.into_iter();
    let Some(first_index) = block_indices.next() else {
        return;
    };
    let mut batch = dataset.query(first_index, batch_size, rng, &mut *batcher);
    let mut last_results = None;

    for next_index in block_indices {
        let ((next_batch, ()), (loss, lr)) = join(
            || {
                join(
                    || dataset.query(next_index, batch_size, rng, &mut *batcher),
                    || {
                        if let Some((loss, lr)) = last_results {
                            post_batch(loss, lr);
                        }
                    },
                )
            },
            || {
                let loss = model.batch_train(batch, loss, training_config);
                let mut grads = loss.backward();
                let lr = lr_scheduler.step();
                model.apply_gradients(lr, &mut grads, grads_plan);
                (loss, lr)
            },
        );
        last_results = Some((loss, lr));
        batch = next_batch;
    }
    let ((), (loss, lr)) = join(
        || {
            let (loss, lr) = last_results.unwrap();
            post_batch(loss, lr);
        },
        || {
            let loss = model.batch_train(batch, loss, training_config);
            let mut grads = loss.backward();
            let lr = lr_scheduler.step();
            model.apply_gradients(lr, &mut grads, grads_plan);
            (loss, lr)
        },
    );
    post_batch(loss, lr);
}


pub fn validate_model<B, M, Row, Item, S>(
    model: &mut M,
    dataset: &mut SqliteDataset,
    batch_size: usize,
    max_batch_count: usize,
    batcher: &mut (impl StatefulBatcher<Row, Item> + Send),
    loss: &M::Loss,
    rng: &mut (impl Rng + Send),
    mut post_batch: impl FnMut(Tensor<B, 1>) + Send,
) where
    M: Send,
    B: Backend,
    Row: FromSqlRow,
    M: ValidatableModel<B, Item, S>,
    Item: Send,
    M::Loss: Send + Sync,
{
    // Sync maker using Mutex
    let mut model = Mutex::new(model);
    let mut block_indices: Vec<_> = (0..dataset.get_batch_count(batch_size))
        .map(|x| x * batch_size)
        .collect();
    block_indices.shuffle(rng);
    block_indices.truncate(max_batch_count);
    let mut block_indices = block_indices.into_iter();
    let Some(first_index) = block_indices.next() else {
        return;
    };
    let mut batch = dataset.query(first_index, batch_size, rng, &mut *batcher);
    let mut last_results = None;

    for next_index in block_indices {
        let ((next_batch, ()), loss) = join(
            || {
                join(
                    || dataset.query(next_index, batch_size, rng, &mut *batcher),
                    || {
                        if let Some(loss) = last_results {
                            post_batch(loss);
                        }
                    },
                )
            },
            || {
                model.get_mut().unwrap().batch_valid(batch, loss)
            },
        );
        last_results = Some(loss);
        batch = next_batch;
    }
    let ((), loss) = join(
        || {
            let loss = last_results.unwrap();
            post_batch(loss);
        },
        || {
            model.get_mut().unwrap().batch_valid(batch, loss)
        },
    );
    post_batch(loss);
}