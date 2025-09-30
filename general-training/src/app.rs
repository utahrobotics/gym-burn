use std::sync::Arc;

use crate::batches::AutoEncoderImageItem;
use crate::regression::RegressionTrainableModel;
use crate::training_loop::simple_regression_training_loop;
use crate::{
    batches::AutoEncoderImageBatcher,
    dataset::{SqliteDataset, SqliteDatasetConfig},
    training_loop::SimpleTrainingConfig,
};
use burn::backend::Autodiff;
use burn::config::Config;
use burn::module::Module;
use burn::record::CompactRecorder;
use general_models::autoencoder::LinearImageAutoEncoderConfig;

// #[derive(Parser, Debug)]
// #[command(version, about, long_about = None)]
// struct Args {
//     #[command(subcommand)]
//     command: Command,
// }

// #[derive(Debug, Subcommand)]
// enum Command {
//     Train,
// }

#[derive(Debug, Config)]
struct ArtifactConfig {
    artifact_dir: String,
}

fn train() {
    #[cfg(feature = "wgpu")]
    type Backend = burn::backend::Wgpu;

    type AutodiffBackend = Autodiff<Backend>;

    #[cfg(feature = "wgpu")]
    let device = general_models::wgpu::get_device();

    let model_config = LinearImageAutoEncoderConfig::load("model.json").unwrap();
    let training_config = SimpleTrainingConfig::load("training.json").unwrap();
    let artifact_config = ArtifactConfig::load("training.json").unwrap();
    let train_dataset_config = SqliteDatasetConfig::load("training-data.json").unwrap();
    let test_dataset_config = SqliteDatasetConfig::load("test-data.json").unwrap();

    std::fs::create_dir_all(&artifact_config.artifact_dir).unwrap();

    let trained = simple_regression_training_loop::<
        AutodiffBackend,
        RegressionTrainableModel<_>,
        _,
        _,
        _,
        _,
        _,
        _,
    >(
        model_config.init::<_, 3, 2, _, _>(device).into(),
        training_config,
        AutoEncoderImageBatcher,
        SqliteDataset::<Arc<AutoEncoderImageItem>, _>::try_from(train_dataset_config).unwrap(),
        SqliteDataset::<Arc<AutoEncoderImageItem>, _>::try_from(test_dataset_config).unwrap(),
        &artifact_config.artifact_dir,
        &device,
    );

    trained
        .model
        .save_file("model.mpk", &CompactRecorder::new())
        .unwrap();
}

pub fn main() {
    train()
}
