use std::path::PathBuf;

use crate::app::training_plans::{train_image_autoencoder, train_image_v_autoencoder};
use crate::{dataset::SqliteDatasetConfig, training_loop::SimpleTrainingConfig};
use burn::backend::Autodiff;
use burn::config::Config;
use clap::{Parser, Subcommand, ValueEnum};
use utils::parse_json_file;

mod training_plans;

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
        } => {
            // let channels = channels.get();
            // assert!(channels < 5, "Too many channels");
            // let test_dataset_config: SqliteDatasetConfig = parse_json_file("test-data").unwrap();
            // let mut rng = rand::rngs::SmallRng::seed_from_u64(
            //     SystemTime::now()
            //         .duration_since(SystemTime::UNIX_EPOCH)
            //         .unwrap()
            //         .as_secs(),
            // );

            // match model_type {
            //     ModelType::AutoEncoder => {
            //         let model_config: LinearImageAutoEncoderConfig =
            //             parse_json_file(config_path).unwrap();
            //         let mut model: LinearImageAutoEncoder<Backend> = model_config.init(device);
            //         model = model.load_record(
            //             CompactRecorder::new()
            //                 .load(weights_path, device)
            //                 .expect("Failed to load weights"),
            //         );
            //         let test_dataset = Arc::new(
            //             SqliteDataset::<Arc<AutoEncoderImageItem>, _>::try_from(
            //                 test_dataset_config,
            //             )
            //             .unwrap(),
            //         );
            //         let mut set = FxHashSet::default();

            //         while set.len() < count {
            //             set.insert(rng.random_range(0..test_dataset.len()));
            //         }

            //         let batcher = AutoEncoderImageBatcher;
            //         let items: Vec<_> = set.iter().map(|&i| test_dataset.get(i).unwrap()).collect();

            //         let width = items.first().unwrap().input_width as u32;
            //         let mosaic_width = width as u32 * 2;
            //         let height = items.first().unwrap().input_height as u32;
            //         let mosaic_height = height * count as u32;

            //         let batch: AutoEncoderImageBatch<Backend> = batcher.batch(items, device);
            //         let now = quanta::Instant::now();
            //         let actual = model.forward(batch.input.clone()).clamp(0.0, 1.0);
            //         let elapsed = now.elapsed();
            //         println!("Forwarded in {:.2}s", elapsed.as_secs_f32());

            //         let mut pixels = vec![];

            //         for (actual, input) in actual.iter_dim(0).zip(batch.input.iter_dim(0)) {
            //             let input = input.into_data();
            //             let actual = actual.into_data();
            //             let mut input_iter = input
            //                 .iter::<f32>()
            //                 .map(|x| (x * 255.0).round().clamp(0.0, 255.0) as u8);
            //             let mut actual_iter = actual
            //                 .iter::<f32>()
            //                 .map(|x| (x * 255.0).round().clamp(0.0, 255.0) as u8);
            //             for _ in 0..height {
            //                 pixels.extend((&mut input_iter).take(width as usize));
            //                 pixels.extend((&mut actual_iter).take(width as usize));
            //             }
            //         }

            //         let img: Option<DynamicImage> = match channels {
            //             1 => ImageBuffer::<Luma<u8>, _>::from_raw(
            //                 mosaic_width,
            //                 mosaic_height,
            //                 pixels,
            //             )
            //             .map(Into::into),
            //             2 => ImageBuffer::<LumaA<u8>, _>::from_raw(
            //                 mosaic_width,
            //                 mosaic_height,
            //                 pixels,
            //             )
            //             .map(Into::into),
            //             3 => {
            //                 ImageBuffer::<Rgb<u8>, _>::from_raw(mosaic_width, mosaic_height, pixels)
            //                     .map(Into::into)
            //             }
            //             4 => ImageBuffer::<Rgba<u8>, _>::from_raw(
            //                 mosaic_width,
            //                 mosaic_height,
            //                 pixels,
            //             )
            //             .map(Into::into),
            //             _ => unreachable!(),
            //         };

            //         img
            //             .expect("Dimensions mismatch. The auto encoder must accept only one size of image and output that same size")
            //             .into_rgb8()
            //             .save("inference.webp")
            //             .unwrap();
            //     }
            // }
        }
    }
}
