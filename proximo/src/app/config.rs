use std::path::PathBuf;

use general_dataset::SqliteDatasetConfig;
use serde::Deserialize;
use utils::default_f;

use crate::trainable_models::apply_gradients::lr_scheduler::LrSchedulerConfig;

#[derive(Debug, Clone, Copy, Deserialize)]
pub enum ModelType {
    #[serde(alias = "image-ae", alias = "img-ae")]
    ImageAutoEncoder,
    #[serde(alias = "image-vae", alias = "img-vae")]
    ImageVariationalAutoEncoder,
}

#[derive(Deserialize, Debug)]
pub struct TrainingConfig {
    pub artifact_dir: PathBuf,
    pub model_type: ModelType,
    #[serde(default)]
    pub viz_command: Vec<String>,
    #[serde(default = "default_num_epochs")]
    pub num_epochs: usize,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_max_batch_count")]
    pub training_max_batch_count: usize,
    #[serde(default = "default_max_batch_count")]
    pub testing_max_batch_count: usize,
    #[serde(default = "default_grad_accumulate_count")]
    pub grad_accumulate_count: usize,
    pub training_dataset: SqliteDatasetConfig,
    pub testing_dataset: SqliteDatasetConfig,
    pub lr_scheduler: LrSchedulerConfig,
    pub seed: Option<u64>,
}

#[derive(Deserialize, Debug)]
pub struct ImageAutoEncoderChallenge {
    #[serde(default)]
    pub challenge_image_count: usize,
}

#[derive(Deserialize, Debug)]
pub struct TrainingGradsPlan<T> {
    pub grads_plan: T,
}

default_f!(default_num_epochs, usize, 10);
default_f!(default_batch_size, usize, 64);
default_f!(default_grad_accumulate_count, usize, 8);
default_f!(default_max_batch_count, usize, usize::MAX);
