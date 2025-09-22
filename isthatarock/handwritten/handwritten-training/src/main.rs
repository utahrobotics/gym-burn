use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu},
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::SqliteDataset},
    module::Module,
    optim::AdamConfig,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{metric::LossMetric, LearnerBuilder},
};
use handwritten_model::HandwrittenAutoEncoderConfig;

use crate::{data::{dataset_commands, HandwrittenAutoEncoderBatcher, SQLITE_DATABASE}, model::TrainableHandwrittenAutoEncoder};

mod data;
mod model;

#[derive(Config, Debug)]
pub struct AutoEncoderTrainingConfig {
    pub model: HandwrittenAutoEncoderConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
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

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    config: AutoEncoderTrainingConfig,
    device: B::Device,
) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(&device, config.seed);

    let batcher = HandwrittenAutoEncoderBatcher::default();

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(SqliteDataset::from_db_file(SQLITE_DATABASE, "train").unwrap());

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(SqliteDataset::from_db_file(SQLITE_DATABASE, "test").unwrap());

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        // .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            TrainableHandwrittenAutoEncoder {
                model: config.model.init::<B>(&device),
            },
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    model_trained
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}

fn model_config() -> HandwrittenAutoEncoderConfig {
    HandwrittenAutoEncoderConfig::new(10, 64)
}
pub type MyBackend = Wgpu<f32, i32>;
pub type MyAutodiffBackend = Autodiff<MyBackend>;

fn main() {
    if let Some(first_arg) = std::env::args().nth(1) {
        dataset_commands(&first_arg);
        return;
    }

    let device = WgpuDevice::default();
    let artifact_dir = "artifacts/isthatarock/handwritten";
    train::<MyAutodiffBackend>(
        artifact_dir,
        AutoEncoderTrainingConfig::new(
            model_config(),
            AdamConfig::new(),
        ),
        device.clone(),
    );
}
