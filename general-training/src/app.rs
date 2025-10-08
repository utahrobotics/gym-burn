use std::path::PathBuf;
use std::time::SystemTime;

use crate::app::infer_plans::infer_image_autoencoder;
use crate::app::training_plans::{train_image_autoencoder, train_image_v_autoencoder};
use crate::{ training_loop::SimpleTrainingConfig};
use burn::backend::Autodiff;
use burn::config::Config;
use clap::{Parser, Subcommand, ValueEnum};
use general_models::composite::autoencoder::AutoEncoderModelConfig;
use general_models::composite::autoencoder::vae::VariationalEncoderConfig;
use general_models::composite::image::{ConvLinearModelConfig, LinearConvTransposedModelConfig};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use utils::parse_json_file;

mod infer_plans;
mod training_plans;

pub fn time_rng() -> SmallRng {
    rand::rngs::SmallRng::seed_from_u64(
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    )
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ModelType {
    #[clap(name = "ae")]
    AutoEncoder,
    #[clap(name = "vae")]
    VariationalAutoEncoder,
}

#[derive(Debug, Subcommand)]
enum Command {
    Train {
        #[arg(short, long, value_enum)]
        model_type: ModelType,
    },
    Infer {
        #[arg(short, long, value_enum)]
        model_type: ModelType,
        #[arg(short, long)]
        weights_path: PathBuf,
        #[arg(long, default_value = "model.json")]
        config_path: PathBuf,
        #[arg(short, long, default_value = "5")]
        count: usize,
    },
}

#[derive(Debug, Config)]
struct ArtifactConfig {
    artifact_dir: PathBuf,
}

pub fn main() {
    #[cfg(feature = "wgpu")]
    type Backend = general_models::wgpu::WgpuBackend;

    #[cfg(feature = "rocm")]
    type Backend = general_models::rocm::RocmBackend;

    #[cfg(feature = "cuda")]
    type Backend = general_models::cuda::CudaBackend;

    type AutodiffBackend = Autodiff<Backend>;

    let args = Args::parse();

    #[cfg(feature = "wgpu")]
    let device = general_models::wgpu::get_device();

    #[cfg(feature = "rocm")]
    let device = general_models::rocm::get_device();

    #[cfg(feature = "cuda")]
    let device = general_models::cuda::get_device();

    match args.command {
        Command::Train { model_type } => {
            let training_config: SimpleTrainingConfig = parse_json_file("training").unwrap();
            let artifact_config: ArtifactConfig = parse_json_file("training").unwrap();
            let train_dataset_config: SqliteDatasetConfig =
                parse_json_file("training-data").unwrap();
            let test_dataset_config: SqliteDatasetConfig = parse_json_file("test-data").unwrap();

            let _ = std::fs::remove_dir_all(&artifact_config.artifact_dir);
            std::fs::create_dir_all(&artifact_config.artifact_dir).unwrap();

            match model_type {
                ModelType::AutoEncoder => {
                    train_image_autoencoder::<AutodiffBackend>(
                        train_dataset_config,
                        test_dataset_config,
                        training_config,
                        artifact_config,
                        device,
                    );
                }
                ModelType::VariationalAutoEncoder => {
                    train_image_v_autoencoder::<AutodiffBackend>(
                        train_dataset_config,
                        test_dataset_config,
                        training_config,
                        artifact_config,
                        device,
                    );
                }
            }
        }
        Command::Infer {
            model_type,
            weights_path,
            config_path,
            count,
        } => match model_type {
            ModelType::AutoEncoder => {
                infer_image_autoencoder::<
                    Backend,
                    _,
                    AutoEncoderModelConfig<ConvLinearModelConfig, LinearConvTransposedModelConfig>,
                >(
                    count,
                    weights_path,
                    config_path,
                    |m| m.encoder.get_input_channels(),
                    device,
                );
            }
            ModelType::VariationalAutoEncoder => {
                infer_image_autoencoder::<
                    Backend,
                    _,
                    AutoEncoderModelConfig<
                        VariationalEncoderConfig<ConvLinearModelConfig>,
                        LinearConvTransposedModelConfig,
                    >,
                >(
                    count,
                    weights_path,
                    config_path,
                    |m| m.encoder.get_model().get_input_channels(),
                    device,
                );
            }
        },
    }
}
