use std::{process::Stdio, time::SystemTime};

use burn::{backend::Autodiff, module::AutodiffModule, nn::loss::MseLoss};
use clap::{Parser, Subcommand};
use general_dataset::{SqliteDataset, presets::autoencoder::AutoEncoderImageBatcher};
use general_models::{
    Init,
    composite::{
        autoencoder::{AutoEncoderModel, AutoEncoderModelConfig},
        image::{
            Conv2dLinearModelConfig, ConvLinearModel, LinearConvTranspose2dModel,
            LinearConvTranspose2dModelConfig,
        },
    },
};
use rand::{SeedableRng, rngs::SmallRng};
use serde_json::json;
use tracing::info;
use utils::parse_json_file;

use crate::{
    app::config::{TrainingConfig, TrainingGradsPlan},
    trainable_models::{Blanket, apply_gradients::{
        ApplyGradients,
        autoencoder::AutoEncoderModelPlanConfig,
        image::{Conv2dLinearModelPlanConfig, LinearConvTranspose2dModelPlanConfig},
    }},
    training_loop::{presets::train_epoch_image_autoencoder, validate_model},
};

pub mod config;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Train,
}

pub fn train() {
    #[cfg(feature = "wgpu")]
    type Backend = general_models::wgpu::WgpuBackend;

    #[cfg(feature = "rocm")]
    type Backend = general_models::rocm::RocmBackend;

    #[cfg(feature = "cuda")]
    type Backend = general_models::cuda::CudaBackend;

    type AutodiffBackend = Autodiff<Backend>;

    #[cfg(feature = "wgpu")]
    let device = general_models::wgpu::get_device();

    #[cfg(feature = "rocm")]
    let device = general_models::rocm::get_device();

    #[cfg(feature = "cuda")]
    let device = general_models::cuda::get_device();

    let training_config: TrainingConfig =
        parse_json_file("training").expect("Expected valid training.json");
    let mut training_dataset: SqliteDataset = training_config
        .training_dataset
        .try_into()
        .expect("Expected valid training dataset config");
    let training_dataset_len = training_dataset.len();

    let mut testing_dataset: SqliteDataset = training_config
        .testing_dataset
        .try_into()
        .expect("Expected valid training dataset config");
    let testing_dataset_len = testing_dataset.len();
    let mut lr_scheduler = training_config.lr_scheduler.init();

    let mut rng = SmallRng::seed_from_u64(training_config.seed.unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }));

    let mut viz_command = training_config.viz_command.into_iter();
    let mut child = viz_command.next().map(|cmd| {
        std::process::Command::new(cmd)
            .args(viz_command)
            .stdin(Stdio::piped())
            .spawn()
            .expect("Expected valid viz command")
    });

    match training_config.model_type {
        config::ModelType::ImageAutoEncoder => {
            let model_config: AutoEncoderModelConfig<
                Conv2dLinearModelConfig,
                LinearConvTranspose2dModelConfig,
            > = parse_json_file("model").expect("Expected valid model.json");
            type Model = AutoEncoderModel<
                AutodiffBackend,
                ConvLinearModel<AutodiffBackend>,
                LinearConvTranspose2dModel<AutodiffBackend>,
            >;
            let mut model: Model = model_config.init(device);
            let grads_plan: TrainingGradsPlan<AutoEncoderModelPlanConfig<
                Conv2dLinearModelPlanConfig,
                LinearConvTranspose2dModelPlanConfig,
            >> = parse_json_file("training").expect("Expected valid training.json");
            let mut grads_plan = Model::config_to_plan(grads_plan.grads_plan);

            let mut training_batcher = AutoEncoderImageBatcher::new(
                model.encoder.conv.get_input_channels(),
                device.clone(),
            );
            let mut testing_batcher = AutoEncoderImageBatcher::new(
                model.encoder.conv.get_input_channels(),
                device.clone(),
            );

            for epoch in 0..training_config.num_epochs {
                let mut batch_i = 0usize;
                info!("Training Epoch {epoch}");
                train_epoch_image_autoencoder::<_, _, _, Blanket>(
                    &mut model,
                    &mut training_dataset,
                    training_dataset_len,
                    training_config.batch_size,
                    training_config.training_max_batch_count,
                    &mut training_batcher,
                    &mut lr_scheduler,
                    &mut grads_plan,
                    &mut rng,
                    |loss, lr| {
                        let (variance, mean) = loss.var_mean(0);
                        let variance = variance.into_scalar();
                        let mean = mean.into_scalar();
                        if let Some(child) = &mut child {
                            serde_json::to_writer(
                                child.stdin.as_mut().unwrap(),
                                &json!({
                                    "batch_i": batch_i,
                                    "epoch": epoch,
                                    "loss": mean,
                                    "loss_variance": variance,
                                    "lr": lr
                                })
                            ).expect("Expected child process to be alive");
                        } else {
                            info!("Batch {batch_i}; Loss: {mean:.4}; Variance: {variance:.4}; LR: {lr:.4}");
                        }
                        batch_i += 1;
                    },
                );
                info!("Testing Epoch {epoch}");
                batch_i = 0;
                validate_model(
                    &mut model.valid(),
                    &mut testing_dataset,
                    testing_dataset_len,
                    training_config.batch_size,
                    training_config.testing_max_batch_count,
                    &mut testing_batcher,
                    &MseLoss::new(),
                    &mut rng,
                    |loss| {
                        let (variance, mean) = loss.var_mean(0);
                        let variance = variance.into_scalar();
                        let mean = mean.into_scalar();
                        if let Some(child) = &mut child {
                            serde_json::to_writer(
                                child.stdin.as_mut().unwrap(),
                                &json!({
                                    "batch_i": batch_i,
                                    "epoch": epoch,
                                    "loss": mean,
                                    "loss_variance": variance
                                })
                            ).expect("Expected child process to be alive");
                        } else {
                            info!("Batch {batch_i}; Loss: {mean:.4}; Variance: {variance:.4}");
                        }
                        batch_i += 1;
                    },
                );
            }
        }
        config::ModelType::ImageVariationalAutoEncoder => todo!(),
    }
}

pub fn main() {
    let args = Args::parse();
    tracing_subscriber::fmt().init();

    match args.command {
        Command::Train => train(),
    }
}
