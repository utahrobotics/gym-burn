use burn::record::RecorderError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LoadModelError {
    #[error("Error loading model config: {0}")]
    ConfigError(#[from] serde_json::Error),
    #[error("Error loading model weights: {0}")]
    WeightsError(#[from] RecorderError),
}
