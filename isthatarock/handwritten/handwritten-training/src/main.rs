use std::collections::{BTreeMap, btree_map::Entry};

use burn::{
    backend::{Autodiff, Wgpu, wgpu::WgpuDevice},
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::SqliteDataset},
    module::Module,
    optim::AdamConfig,
    prelude::Backend,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{LearnerBuilder, metric::LossMetric},
};
use handwritten_model::{HandwrittenAutoEncoder, HandwrittenAutoEncoderConfig};

use image::{DynamicImage, imageops::FilterType};
use linfa::traits::*;
use linfa_clustering::Dbscan;
use ndarray::Array2;

use crate::{data::HandwrittenAutoEncoderBatcher, model::TrainableHandwrittenAutoEncoder};

mod data;
mod model;

pub const SQLITE_DATABASE: &str = "isthatarock-handwritten-dataset.sqlite";

#[derive(Config, Debug)]
pub struct AutoEncoderTrainingConfig {
    pub model: HandwrittenAutoEncoderConfig,
    pub optimizer: AdamConfig,
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

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

fn train<B: AutodiffBackend>(
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
        .clone()
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");

    cluster(&model_trained.model.model);
}

fn model_config() -> HandwrittenAutoEncoderConfig {
    HandwrittenAutoEncoderConfig::new(10, 64)
}

pub type MyBackend = Wgpu<f32, i32>;
pub type MyAutodiffBackend = Autodiff<MyBackend>;

fn cluster<B: Backend>(model: &HandwrittenAutoEncoder<B>) {
    // let device = WgpuDevice::default();
    // let mut model = model_config().init::<MyBackend>(&device);
    // model.load_compact_record_file("artifacts/isthatarock/handwritten/model.mpk".into(), &device);

    let batcher = HandwrittenAutoEncoderBatcher::default();

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(64)
        .num_workers(4)
        .build(SqliteDataset::from_db_file(SQLITE_DATABASE, "train").unwrap());

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(64)
        .num_workers(4)
        .build(SqliteDataset::from_db_file(SQLITE_DATABASE, "test").unwrap());

    let mut latent_points: Vec<[f32; 10]> = vec![];
    for batch in dataloader_train.iter().chain(dataloader_test.iter()) {
        let latent_batch = model.encode(batch.images);
        for tensor in latent_batch.iter_dim(0) {
            let point: Vec<_> = tensor.into_data().iter().collect();
            latent_points.push(point.try_into().unwrap());
        }
    }
    let min_points = 3;
    let clusters = Dbscan::params::<f32>(min_points)
        .tolerance(1e-2)
        .transform(&Array2::from(latent_points))
        .unwrap();
    let mut ids = BTreeMap::default();
    let mut noise = 0usize;
    for id in clusters {
        let Some(id) = id else {
            noise += 1;
            continue;
        };
        match ids.entry(id) {
            Entry::Occupied(mut entry) => *entry.get_mut() += 1,
            Entry::Vacant(entry) => {
                entry.insert(1usize);
            }
        }
    }
    for (id, count) in ids {
        println!("{id}: {count}");
    }
    println!("noise: {noise}");
}

// fn validate_color(img: &mut DynamicImage) {
//     let mut luma_bytes = img.to_luma8().into_vec();
//     luma_bytes.sort_unstable();
//     if luma_bytes[luma_bytes.len() / 2] > 127 {
//         img.invert();
//     }
// }

fn main() {
    if let Some(command) = std::env::args().nth(1) {
        match command.as_str() {
            "infer" => {
                let Some(path) = std::env::args().nth(2) else {
                    eprintln!("Missing path");
                    return;
                };
                let device = WgpuDevice::default();
                let mut model = model_config().init::<MyBackend>(&device);
                model.load_compact_record_file(
                    "artifacts/isthatarock/handwritten/model.mpk".into(),
                    &device,
                );
                let mut image = image::open(path).unwrap();
                image = image.resize_exact(28, 28, FilterType::CatmullRom);
                // validate_color(&mut image);
                let mut image = image.to_luma32f();
                let mut output = vec![];
                model.forward_slice(&image, &mut output, &device);
                image.copy_from_slice(&output);
                DynamicImage::from(image)
                    .to_rgba8()
                    .save("valid.webp")
                    .unwrap();
            }
            _ => {
                eprintln!("Unknown command");
            }
        }
        return;
    }
    let device = WgpuDevice::default();
    let artifact_dir = "artifacts/isthatarock/handwritten";
    train::<MyAutodiffBackend>(
        artifact_dir,
        AutoEncoderTrainingConfig::new(model_config(), AdamConfig::new()),
        device.clone(),
    );
}
