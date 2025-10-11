use std::sync::Mutex;

use burn::{Tensor, lr_scheduler::LrScheduler, module::AutodiffModule, optim::{AdamConfig, GradientsParams, Optimizer, SgdConfig}, prelude::Backend, tensor::backend::AutodiffBackend};
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

pub fn train_epoch<B, M, Row, Item, L, S>(
    model: &mut M,
    dataset: &mut SqliteDataset,
    batch_size: usize,
    max_batch_count: usize,
    batcher: &mut (impl StatefulBatcher<Row, Item> + Send),
    training_config: &M::TrainingConfig,
    loss: &L,
    lr_scheduler: &mut impl LrScheduler,
    grads_plan: &mut M::Plan,
    rng: &mut (impl Rng + Send),
    mut post_batch: impl FnMut(Tensor<B, 1>, f64) -> bool + Send,
) where
    M: Send,
    B: AutodiffBackend,
    Row: FromSqlRow,
    M: TrainableModel<B, Item, L, S> + ApplyGradients<B> + AutodiffModule<B>,
    Item: Send,
    L: Send + Sync,
    M::Plan: Send,
    M::TrainingConfig: Sync,
{
    let model_ptr = model;
    let mut model = unsafe {
        std::ptr::read(model_ptr)
    };
    let mut optimizer = SgdConfig::new().init();
    // let mut grads_accumulator = GradientsAccumulator::new();
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
    // let mut grads_count = 0usize;

    for next_index in block_indices {
        let ((next_batch, end), (loss, lr, tmp)) = join(
            || {
                join(
                    || dataset.query(next_index, batch_size, rng, &mut *batcher),
                    || {
                        if let Some((loss, lr)) = last_results {
                            post_batch(loss, lr)
                        } else {
                            false
                        }
                    },
                )
            },
            || {
                let loss = model.batch_train(batch, loss, training_config);
                let mut grads = loss.backward();
                // grads_count += 1;
                // grads_accumulator.accumulate(model, grads);
                let lr = lr_scheduler.step();
                let grads = GradientsParams::from_grads(grads, &model);
                let model = optimizer.step(lr, model, grads);
                // model.apply_gradients(lr, &mut grads, grads_plan);
                (loss, lr, model)
            },
        );
        model = tmp;
        last_results = Some((loss, lr));
        batch = next_batch;
        if end {
            break;
        }
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
    unsafe {
        std::ptr::write(model_ptr, model);
    }
}

pub fn validate_model<B, M, Row, Item, L, S>(
    model: &mut M,
    dataset: &mut SqliteDataset,
    batch_size: usize,
    max_batch_count: usize,
    batcher: &mut (impl StatefulBatcher<Row, Item> + Send),
    loss: &L,
    rng: &mut (impl Rng + Send),
    mut post_batch: impl FnMut(Tensor<B, 1>) -> bool + Send,
) where
    M: Send,
    B: Backend,
    Row: FromSqlRow,
    M: ValidatableModel<B, Item, L, S>,
    Item: Send,
    L: Send + Sync,
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
        let ((next_batch, end), loss) = join(
            || {
                join(
                    || dataset.query(next_index, batch_size, rng, &mut *batcher),
                    || {
                        if let Some(loss) = last_results {
                            post_batch(loss)
                        } else {
                            false
                        }
                    },
                )
            },
            || model.get_mut().unwrap().batch_valid(batch, loss),
        );
        last_results = Some(loss);
        batch = next_batch;
        if end {
            break;
        }
    }
    let ((), loss) = join(
        || {
            let loss = last_results.unwrap();
            post_batch(loss);
        },
        || model.get_mut().unwrap().batch_valid(batch, loss),
    );
    post_batch(loss);
}
