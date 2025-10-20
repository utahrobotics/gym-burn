use std::{io::Cursor, process::Stdio, sync::atomic::AtomicBool, time::SystemTime};

use base64::{Engine, prelude::BASE64_STANDARD};
use burn::{
    module::{AutodiffModule, DisplaySettings, Module, ModuleDisplay},
    nn::loss::MseLoss,
    prelude::Backend,
    record::CompactRecorder,
};
use clap::{Parser, Subcommand};
use general_dataset::{
    SqliteDataset, StatefulBatcher,
    presets::autoencoder::{AutoEncoderImageBatch, AutoEncoderImageBatcher, AutoEncoderImageItem},
};
use general_models::{
    Init, SimpleInfer, SimpleTrain,
    composite::{
        autoencoder::{AutoEncoderModel, AutoEncoderModelConfig},
        image::{
            Conv2dLinearModel, Conv2dLinearModelConfig, LinearConvTranspose2dModel,
            LinearConvTranspose2dModelConfig,
        },
    },
};
use image::{
    ImageBuffer, ImageDecoder, ImageFormat, Luma, Rgb, buffer::ConvertBuffer,
    codecs::webp::WebPDecoder,
};
use rand::{Rng, SeedableRng, rngs::SmallRng};
use rayon::iter::{IndexedParallelIterator, ParallelBridge, ParallelIterator};
use serde_json::json;
use tracing::info;
use utils::parse_json_file;

use crate::{
    app::{
        config::{ImageAutoEncoderChallenge, ModelType, TrainingConfig, TrainingGradsPlan},
        loss::bce_float_loss,
    },
    trainable_models::{
        AdHocLossModel,
        apply_gradients::{
            AdHocTrainingPlanConfig, ApplyAllGradients, ApplyGradients,
            autoencoder::AutoEncoderModelPlanConfig,
            image::{Conv2dLinearModelPlanConfig, LinearConvTranspose2dModelPlanConfig},
        },
    },
    training_loop::{train_epoch, validate_model},
};

use rayon::iter::ParallelDrainRange;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

pub mod config;
pub mod loss;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Train,
    Clean,
}

pub fn train() {
    let clock = quanta::Clock::new();
    let init_start_time = clock.now();

    let ctrlc_pressed: &_ = Box::leak(Box::new(AtomicBool::new(false)));

    ctrlc::set_handler(move || {
        ctrlc_pressed.store(true, std::sync::atomic::Ordering::Relaxed);
        println!("Cancelling due to Ctrl-C ...");
    })
    .expect("Error setting Ctrl-C handler");

    #[cfg(all(feature = "wgpu", not(feature = "tracking-backend")))]
    type Backend = general_models::wgpu::WgpuBackend;

    #[cfg(feature = "tracking-backend")]
    type AutodiffBackend = tracking_backend::TrackingBackend;
    #[cfg(feature = "tracking-backend")]
    type Backend =
        <tracking_backend::TrackingBackend as burn::tensor::backend::AutodiffBackend>::InnerBackend;

    #[cfg(feature = "rocm")]
    type Backend = general_models::rocm::RocmBackend;

    #[cfg(feature = "cuda")]
    type Backend = general_models::cuda::CudaBackend;

    #[cfg(not(feature = "tracking-backend"))]
    type AutodiffBackend = burn::backend::Autodiff<Backend>;

    #[cfg(feature = "wgpu")]
    let device = general_models::wgpu::get_device();

    #[cfg(feature = "rocm")]
    let device = general_models::rocm::get_device();

    #[cfg(feature = "cuda")]
    let device = general_models::cuda::get_device();

    let training_config: TrainingConfig =
        parse_json_file("training").expect("Expected valid training.json");

    let secs = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let artifact_dir = training_config.artifact_dir.join(secs.to_string());
    std::fs::create_dir_all(&artifact_dir).expect("Expected artifact dir to be creatable");

    #[cfg(feature = "tracking-backend")]
    tracking_backend::set_artifact_dir(artifact_dir.clone());

    let mut training_dataset: SqliteDataset = training_config
        .training_dataset
        .try_into()
        .expect("Expected valid training dataset config");

    let mut testing_dataset: SqliteDataset = training_config
        .testing_dataset
        .try_into()
        .expect("Expected valid training dataset config");
    let mut lr_scheduler = training_config.lr_scheduler.init();

    let mut rng = SmallRng::seed_from_u64(training_config.seed.unwrap_or(secs));
    Backend::seed(device, rng.random());

    let mut viz_command = training_config.viz_command.into_iter();
    let mut child = viz_command.next().map(|cmd| {
        std::process::Command::new(cmd)
            .args(viz_command)
            .env("ARTIFACT_DIR", &artifact_dir)
            .env("TRAINING_BATCH_COUNT", &artifact_dir)
            .stdin(Stdio::piped())
            .spawn()
            .expect("Expected valid viz command")
    });

    match training_config.model_type {
        ModelType::ImageAutoEncoder => {
            let challenge_config: ImageAutoEncoderChallenge =
                parse_json_file("training").expect("Expected valid training.json");
            let model_config: AutoEncoderModelConfig<
                Conv2dLinearModelConfig,
                LinearConvTranspose2dModelConfig,
            > = parse_json_file("model").expect("Expected valid model.json");
            type AutodiffModel = AutoEncoderModel<
                AutodiffBackend,
                Conv2dLinearModel<AutodiffBackend>,
                LinearConvTranspose2dModel<AutodiffBackend>,
            >;
            type Model = AutoEncoderModel<
                Backend,
                Conv2dLinearModel<Backend>,
                LinearConvTranspose2dModel<Backend>,
            >;
            let mut model: AutodiffModel = model_config.init(device);
            std::fs::write(
                artifact_dir.join("model.txt"),
                model.format(DisplaySettings::new()).as_bytes(),
            )
            .expect("Expected model.txt to be writable in artifact dir");

            let grads_plan: TrainingGradsPlan<
                AdHocTrainingPlanConfig<
                    AutoEncoderModelPlanConfig<
                        Conv2dLinearModelPlanConfig,
                        LinearConvTranspose2dModelPlanConfig,
                    >,
                >,
            > = parse_json_file("training").expect("Expected valid training.json");
            let mut grads_plan = AdHocLossModel::<_, ()>::config_to_plan(grads_plan.grads_plan);

            let mut training_batcher = AutoEncoderImageBatcher::<AutodiffBackend>::new(
                model.encoder.get_input_channels(),
                device.clone(),
            );
            let mut testing_batcher = AutoEncoderImageBatcher::<Backend>::new(
                model.encoder.get_input_channels(),
                device.clone(),
            );

            let mut input_images = vec![];

            info!(
                "Initialized in {:.3}s",
                init_start_time.elapsed().as_secs_f32()
            );
            let training_start_time = clock.now();
            for epoch in 0..training_config.num_epochs {
                let epoch_start_time = clock.now();
                let mut batch_i = 0usize;
                info!("Training Epoch {epoch}");

                let mut trainable_model = AdHocLossModel::new(
                    model,
                    |model: &AutodiffModel, item: AutoEncoderImageBatch<AutodiffBackend>| {
                        MseLoss::new().forward(
                            model.train(item.input),
                            item.expected,
                            burn::nn::loss::Reduction::Auto,
                        )
                        // bce_float_loss(
                        //     model.train(item.input),
                        //     item.expected,
                        // )
                    },
                );

                trainable_model = train_epoch::<AutodiffBackend, _, _, _>(
                    trainable_model,
                    &mut training_dataset,
                    training_config.batch_size,
                    training_config.training_max_batch_count,
                    &mut training_batcher,
                    &mut lr_scheduler,
                    &mut grads_plan,
                    &mut rng,
                    device,
                    |loss, lr| {
                        let ctrlc_pressed =
                            ctrlc_pressed.load(std::sync::atomic::Ordering::Relaxed);
                        let loss = loss.into_scalar();
                        if let Some(child) = &mut child {
                            let result = serde_json::to_writer(
                                child.stdin.as_mut().unwrap(),
                                &json!({
                                    "batch_i": batch_i,
                                    "epoch": epoch,
                                    "loss": loss,
                                    "lr": lr
                                }),
                            );
                            if !ctrlc_pressed {
                                result.expect("Expected child process to be alive");
                            }
                        } else {
                            info!("Batch {batch_i}; Loss: {loss:.4}; LR: {lr:.4}");
                        }
                        batch_i += 1;
                        ctrlc_pressed
                    },
                );

                model = trainable_model.unwrap();

                model
                    .clone()
                    .save_file(
                        artifact_dir.join(format!("model-{epoch}.mpk")),
                        &CompactRecorder::new(),
                    )
                    .expect("Expected model to be saveable to artifact dir");

                testing_batcher.reset();
                let mut output_width = 0usize;
                let mut output_height = 0usize;
                for _ in 0..challenge_config.challenge_image_count {
                    let item: AutoEncoderImageItem = testing_dataset.pick_random(&mut rng);
                    output_width = item.expected_width;
                    output_height = item.expected_height;
                    input_images.push(item.webp_input.clone());
                    testing_batcher.ingest(item);
                }
                let batch = testing_batcher.finish();
                let model: Model = model.valid();
                let reconstructed = model.infer(batch.input.clone());

                let reconstructed_images: Vec<_> = match model.encoder.get_input_channels() {
                    1 => reconstructed
                        .iter_dim(0)
                        .par_bridge()
                        .map(|tensor| {
                            let [_, _, width, height] = tensor.dims();
                            let buf = tensor.into_data().into_vec::<f32>().unwrap();
                            let img = ImageBuffer::<Luma<f32>, _>::from_raw(
                                width as u32,
                                height as u32,
                                buf,
                            )
                            .unwrap();
                            let img: ImageBuffer<Rgb<u8>, Vec<_>> = img.convert();

                            img
                        })
                        .collect(),
                    _ => todo!(),
                };

                if let Some(child) = &mut child {
                    let images: Vec<_> = input_images
                        .par_drain(..)
                        .zip(reconstructed_images)
                        .map(|(input, output)| {
                            let mut output_bytes = vec![];
                            output
                                .write_to(&mut Cursor::new(&mut output_bytes), ImageFormat::WebP)
                                .unwrap();
                            (
                                BASE64_STANDARD.encode(input),
                                BASE64_STANDARD.encode(output_bytes),
                            )
                        })
                        .collect();
                    serde_json::to_writer(
                        child.stdin.as_mut().unwrap(),
                        &json!({
                            "epoch": epoch,
                            "challenge_images": images,
                        }),
                    )
                    .expect("Expected child process to be alive");
                } else {
                    let mosaic_width = output_width as u32 * 2;
                    let mosaic_height =
                        output_height as u32 * challenge_config.challenge_image_count as u32;
                    let mut pixels =
                        Vec::with_capacity(mosaic_width as usize * mosaic_height as usize);
                    let mut input_buf = vec![];
                    input_buf.resize(output_width * output_height * 3, 0);

                    input_images
                        .iter()
                        .zip(reconstructed_images.iter())
                        .for_each(|(input, output)| {
                            WebPDecoder::new(Cursor::new(input))
                                .unwrap()
                                .read_image(&mut input_buf)
                                .unwrap();
                            let iter = input_buf
                                .chunks(output_width * 3)
                                .zip(output.chunks(output_width * 3));
                            for (input_row, output_row) in iter {
                                pixels.extend_from_slice(input_row);
                                pixels.extend_from_slice(output_row);
                            }
                        });
                    input_images.clear();
                    ImageBuffer::<Rgb<u8>, _>::from_raw(mosaic_width, mosaic_height, pixels)
                        .unwrap()
                        .save(artifact_dir.join(format!("infer-{epoch}.webp")))
                        .expect("Expected inference image to be saveable");
                }

                if ctrlc_pressed.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }

                let mut validatable_model = AdHocLossModel::new(
                    model,
                    |model: &Model, item: AutoEncoderImageBatch<Backend>| {
                        bce_float_loss(
                            item.expected,
                            model.infer(item.input),
                            // 1.0,
                            // 0.1
                        )
                    },
                );

                info!("Testing Epoch {epoch}");
                batch_i = 0;
                validate_model::<Backend, _, _, _>(
                    &mut validatable_model,
                    &mut testing_dataset,
                    training_config.batch_size,
                    training_config.testing_max_batch_count,
                    &mut testing_batcher,
                    &mut rng,
                    |loss| {
                        let ctrlc_pressed =
                            ctrlc_pressed.load(std::sync::atomic::Ordering::Relaxed);
                        let loss = loss.into_scalar();
                        if let Some(child) = &mut child {
                            let result = serde_json::to_writer(
                                child.stdin.as_mut().unwrap(),
                                &json!({
                                    "batch_i": batch_i,
                                    "epoch": epoch,
                                    "loss": loss,
                                }),
                            );
                            if !ctrlc_pressed {
                                result.expect("Expected child process to be alive");
                            }
                        } else {
                            info!("Batch {batch_i}; Loss: {loss:.4}");
                        }
                        batch_i += 1;
                        ctrlc_pressed
                    },
                );
                let epoch_duration = epoch_start_time.elapsed();
                info!(
                    "Epoch Duration: {:.1}s; Remaining: {:.1}s",
                    epoch_duration.as_secs_f32(),
                    training_start_time.elapsed().as_secs_f32()
                        * (training_config.num_epochs as f32 / (epoch + 1) as f32 - 1.0)
                );
            }

            info!(
                "Total Duration: {:.1}s",
                training_start_time.elapsed().as_secs_f32()
            );
        }
        ModelType::ImageVariationalAutoEncoder => todo!(),
    }

    #[cfg(feature = "tracking-backend")]
    tracking_backend::wait_until_paused();
}

pub fn main() {
    #[cfg(feature = "dhat-ad-hoc")]
    let _profiler = dhat::Profiler::new_ad_hoc();
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    let args = Args::parse();
    tracing_subscriber::fmt().init();

    match args.command {
        Command::Train => train(),
        Command::Clean => {
            let training_config: TrainingConfig =
                parse_json_file("training").expect("Expected valid training.json");
            std::fs::remove_dir_all(&training_config.artifact_dir)
                .expect("Expected artifact dir to be removable");
        }
    }
}
