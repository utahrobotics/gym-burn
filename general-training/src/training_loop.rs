use std::{
    fmt::{Debug, Display},
    path::Path,
};

use burn::{
    config::Config,
    data::{
        dataloader::{DataLoaderBuilder, batcher::Batcher},
        dataset::Dataset,
    },
    module::AutodiffModule,
    optim::AdamConfig,
    prelude::Backend,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        LearnerBuilder, RegressionOutput, TrainStep, ValidStep,
        metric::{Adaptor, ItemLazy, LossInput, LossMetric},
    },
};

#[derive(Config, Debug)]
pub struct SimpleTrainingConfig {
    pub optimizer: Option<AdamConfig>,
    #[config(default = 20)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

pub fn simple_training_loop<
    B,
    M,
    BatchTraining,
    BatchValid,
    I,
    OutputTraining,
    OutputValid,
    BSync,
>(
    model: M,
    training_config: SimpleTrainingConfig,
    batcher: impl Batcher<B, I, BatchTraining>
    + Batcher<B::InnerBackend, I, BatchValid>
    + Clone
    + 'static,
    training_dataset: impl Dataset<I> + 'static,
    validation_dataset: impl Dataset<I> + 'static,
    artifact_dir: impl AsRef<Path>,
    device: &B::Device,
) -> M::InnerModule
where
    B: AutodiffBackend,
    BSync: Backend,
    I: Clone + Debug + Send + Sync + 'static,
    BatchTraining: Clone + Debug + Send + 'static,
    BatchValid: Clone + Debug + Send + 'static,
    OutputTraining: ItemLazy + 'static,
    OutputValid: ItemLazy + 'static,
    M: AutodiffModule<B> + Display + TrainStep<BatchTraining, OutputTraining> + 'static,
    M::Record: 'static,
    M::InnerModule: ValidStep<BatchValid, OutputValid>,
    <OutputTraining as ItemLazy>::ItemSync: Adaptor<LossInput<BSync>>,
    <OutputValid as ItemLazy>::ItemSync: Adaptor<LossInput<BSync>>,
{
    B::seed(&device, training_config.seed);

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(training_config.batch_size)
        .shuffle(training_config.seed)
        .num_workers(training_config.num_workers)
        .build(training_dataset);

    let dataloader_valid = DataLoaderBuilder::new(batcher)
        .batch_size(training_config.batch_size)
        .shuffle(training_config.seed)
        .num_workers(training_config.num_workers)
        .build(validation_dataset);

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        // .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(training_config.num_epochs)
        .summary()
        .build(
            model,
            training_config
                .optimizer
                .unwrap_or_else(|| AdamConfig::new())
                .init(),
            training_config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_valid);
    model_trained.model
}

pub fn simple_regression_training_loop<B, M, BatchTraining, BatchValid, I>(
    model: M,
    training_config: SimpleTrainingConfig,
    batcher: impl Batcher<B, I, BatchTraining>
    + Batcher<B::InnerBackend, I, BatchValid>
    + Clone
    + 'static,
    training_dataset: impl Dataset<I> + 'static,
    validation_dataset: impl Dataset<I> + 'static,
    artifact_dir: impl AsRef<Path>,
    device: &B::Device,
) -> M::InnerModule
where
    B: AutodiffBackend,
    I: Clone + Debug + Send + Sync + 'static,
    BatchTraining: Clone + Debug + Send + 'static,
    BatchValid: Clone + Debug + Send + 'static,
    M: AutodiffModule<B> + Display + TrainStep<BatchTraining, RegressionOutput<B>> + 'static,
    M::Record: 'static,
    M::InnerModule: ValidStep<BatchValid, RegressionOutput<B::InnerBackend>>,
{
    simple_training_loop(
        model,
        training_config,
        batcher,
        training_dataset,
        validation_dataset,
        artifact_dir,
        device,
    )
}
