use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;

use crate::batches::{AutoEncoderImageBatch, AutoEncoderImageItem};
use crate::regression::RegressionTrainableModel;
use crate::training_loop::simple_regression_training_loop;
use crate::{
    batches::AutoEncoderImageBatcher,
    dataset::{SqliteDataset, SqliteDatasetConfig},
    training_loop::SimpleTrainingConfig,
};
use burn::backend::Autodiff;
use burn::config::Config;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::module::Module;
use burn::record::{CompactRecorder, Recorder};
use clap::{Parser, Subcommand, ValueEnum};
use general_models::SimpleForwardable;
use general_models::autoencoder::{LinearImageAutoEncoder, LinearImageAutoEncoderConfig};
use image::{DynamicImage, ImageBuffer, Luma, LumaA, Rgb, Rgba};
use rand::{Rng, SeedableRng};
use rustc_hash::FxHashSet;
use utils::parse_json_file;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ModelType {
    #[clap(name = "auto-encoder")]
    AutoEncoder,
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
        #[arg(long, default_value = "1")]
        channels: NonZeroUsize,
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

            macro_rules! epilogue {
                ($training_dataset: ident, $test_dataset: ident, $trained: ident) => {{
                    println!(
                        "Training Cache Performance: {} / {}",
                        $training_dataset.get_cache_hits(),
                        $training_dataset.get_reads()
                    );
                    println!(
                        "Testing Cache Performance:  {} / {}",
                        $test_dataset.get_cache_hits(),
                        $test_dataset.get_reads()
                    );
                    $trained
                        .model
                        .save_file(
                            artifact_config.artifact_dir.join("model.mpk"),
                            &CompactRecorder::new(),
                        )
                        .unwrap();
                }};
            }

            match model_type {
                ModelType::AutoEncoder => {
                    let model_config: LinearImageAutoEncoderConfig =
                        parse_json_file("model").unwrap();
                    let training_dataset = Arc::new(
                        SqliteDataset::<Arc<AutoEncoderImageItem>, _>::try_from(
                            train_dataset_config,
                        )
                        .unwrap(),
                    );
                    let test_dataset = Arc::new(
                        SqliteDataset::<Arc<AutoEncoderImageItem>, _>::try_from(
                            test_dataset_config,
                        )
                        .unwrap(),
                    );

                    let trained = simple_regression_training_loop::<
                        AutodiffBackend,
                        RegressionTrainableModel<LinearImageAutoEncoder<_>>,
                        _,
                        _,
                        _,
                    >(
                        model_config.init(device).into(),
                        training_config,
                        AutoEncoderImageBatcher,
                        training_dataset.clone(),
                        test_dataset.clone(),
                        &artifact_config.artifact_dir,
                        &device,
                    );

                    trained
                        .model
                        .encoder
                        .clone()
                        .save_file(
                            artifact_config.artifact_dir.join("encoder.mpk"),
                            &CompactRecorder::new(),
                        )
                        .unwrap();
                    trained
                        .model
                        .decoder
                        .clone()
                        .save_file(
                            artifact_config.artifact_dir.join("decoder.mpk"),
                            &CompactRecorder::new(),
                        )
                        .unwrap();
                    epilogue!(training_dataset, test_dataset, trained);
                }
            }
        }
        Command::Infer {
            model_type,
            weights_path,
            config_path,
            count,
            channels,
        } => {
            let channels = channels.get();
            assert!(channels < 5, "Too many channels");
            let test_dataset_config: SqliteDatasetConfig = parse_json_file("test-data").unwrap();
            let mut rng = rand::rngs::SmallRng::seed_from_u64(
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            );

            match model_type {
                ModelType::AutoEncoder => {
                    let model_config: LinearImageAutoEncoderConfig =
                        parse_json_file(config_path).unwrap();
                    let mut model: LinearImageAutoEncoder<Backend> = model_config.init(device);
                    model = model.load_record(
                        CompactRecorder::new()
                            .load(weights_path, device)
                            .expect("Failed to load weights"),
                    );
                    let test_dataset = Arc::new(
                        SqliteDataset::<Arc<AutoEncoderImageItem>, _>::try_from(
                            test_dataset_config,
                        )
                        .unwrap(),
                    );
                    let mut set = FxHashSet::default();

                    while set.len() < count {
                        set.insert(rng.random_range(0..test_dataset.len()));
                    }

                    let batcher = AutoEncoderImageBatcher;
                    let items: Vec<_> = set.iter().map(|&i| test_dataset.get(i).unwrap()).collect();

                    let width = items.first().unwrap().input_width as u32;
                    let mosaic_width = width as u32 * 2;
                    let height = items.first().unwrap().input_height as u32;
                    let mosaic_height = height * count as u32;

                    let batch: AutoEncoderImageBatch<Backend> = batcher.batch(items, device);
                    let now = quanta::Instant::now();
                    let actual = model.forward(batch.input.clone()).clamp(0.0, 1.0);
                    let elapsed = now.elapsed();
                    println!("Forwarded in {:.2}s", elapsed.as_secs_f32());

                    let mut pixels = vec![];

                    for (actual, input) in actual.iter_dim(0).zip(batch.input.iter_dim(0)) {
                        let input = input.into_data();
                        let actual = actual.into_data();
                        let mut input_iter = input
                            .iter::<f32>()
                            .map(|x| (x * 255.0).round().clamp(0.0, 255.0) as u8);
                        let mut actual_iter = actual
                            .iter::<f32>()
                            .map(|x| (x * 255.0).round().clamp(0.0, 255.0) as u8);
                        for _ in 0..height {
                            pixels.extend((&mut input_iter).take(width as usize));
                            pixels.extend((&mut actual_iter).take(width as usize));
                        }
                    }

                    let img: Option<DynamicImage> = match channels {
                        1 => ImageBuffer::<Luma<u8>, _>::from_raw(
                            mosaic_width,
                            mosaic_height,
                            pixels,
                        )
                        .map(Into::into),
                        2 => ImageBuffer::<LumaA<u8>, _>::from_raw(
                            mosaic_width,
                            mosaic_height,
                            pixels,
                        )
                        .map(Into::into),
                        3 => {
                            ImageBuffer::<Rgb<u8>, _>::from_raw(mosaic_width, mosaic_height, pixels)
                                .map(Into::into)
                        }
                        4 => ImageBuffer::<Rgba<u8>, _>::from_raw(
                            mosaic_width,
                            mosaic_height,
                            pixels,
                        )
                        .map(Into::into),
                        _ => unreachable!(),
                    };

                    img
                        .expect("Dimensions mismatch. The auto encoder must accept only one size of image and output that same size")
                        .into_rgb8()
                        .save("inference.webp")
                        .unwrap();
                }
            }
        }
    }
}
