use std::{collections::{BTreeMap, btree_map::Entry}, io::Cursor, path::Path};

use burn::{
    backend::wgpu::WgpuDevice, data::{dataloader::{DataLoaderBuilder, batcher::Batcher}, dataset::SqliteDataset}, prelude::Backend, tensor::{Tensor, TensorData}
};
use crossbeam::queue::SegQueue;
use image::{imageops::FilterType, DynamicImage, ImageFormat};
use linfa::traits::*;
use linfa_clustering::Dbscan;
use ndarray::Array2;
use rayon::iter::{ParallelBridge, ParallelIterator};
use rusqlite::{Connection, ErrorCode, params};
use serde::{Deserialize, Serialize};

use crate::{MyBackend, model_config};

#[derive(Clone, Default)]
pub struct HandwrittenAutoEncoderBatcher {}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct HandwrittenAutoEncoderItem {
    pub image_blob: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct HandwrittenAutoEncoderBatch<B: Backend> {
    pub images: Tensor<B, 3>,
}

impl<B: Backend> Batcher<B, HandwrittenAutoEncoderItem, HandwrittenAutoEncoderBatch<B>>
    for HandwrittenAutoEncoderBatcher
{
    fn batch(
        &self,
        items: Vec<HandwrittenAutoEncoderItem>,
        device: &B::Device,
    ) -> HandwrittenAutoEncoderBatch<B> {
        let images = items
            .iter()
            .map(|item|
                image::load_from_memory_with_format(&item.image_blob, ImageFormat::WebP).unwrap()
            )
            .map(|image| image.to_luma32f())
            .map(|image| TensorData::from(&*image))
            .map(|data| Tensor::<B, 1>::from_data(data, device))
            .map(|tensor| tensor.reshape([1, 28, 28]))
            // // Normalize: scale between [0,1] and make the mean=0 and std=1
            // // values mean=0.1307,std=0.3081 are from the PyTorch MNIST example
            // // https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/mnist/main.py#L122
            // .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)
            .collect();

        let images = Tensor::cat(images, 0);

        HandwrittenAutoEncoderBatch { images }
    }
}

fn recursive_iter(path: &Path, f: &(impl Fn(&Path, Box<dyn FnOnce() -> Vec<u8> + 'static>) + Sync)) {
    if path.is_dir() {
        std::fs::read_dir(&path)
            .unwrap()
            .par_bridge()
            .for_each(|result| {
                let entry = result.unwrap();
                recursive_iter(&entry.path(), f);
            });
    } else {
        let Some(extension) = path.extension().map(|s| s.to_str().unwrap()) else {
            return;
        };
        if !matches!(extension, "png" | "webp" | "jpg") {
            return;
        }
        let next = path.to_path_buf();
        let blob_fn = Box::new(move || {
            let mut img = image::open(next).unwrap();
            img = img.resize_exact(28, 28, FilterType::CatmullRom);
            validate_color(&mut img);
            let mut blob = vec![];
            img.write_to(&mut Cursor::new(&mut blob), ImageFormat::WebP).unwrap();
            blob
        });
        f(&path, blob_fn);
    }
}

fn validate_color(img: &mut DynamicImage) {
    let mut luma_bytes = img.to_luma8().into_vec();
    luma_bytes.sort_unstable();
    if luma_bytes[luma_bytes.len() / 2] > 127 {
        img.invert();
    }
}

pub const SQLITE_DATABASE: &str = "isthatarock-handwritten-dataset.sqlite";

pub fn dataset_commands(command: &str) {
    let conn = Connection::open(SQLITE_DATABASE).unwrap();
    conn.execute(
        r#"
        CREATE TABLE IF NOT EXISTS `train` (
            `row_id` INTEGER PRIMARY KEY,
            `hash` TEXT NOT NULL UNIQUE,
            `image_blob` BLOB NOT NULL,
            `noisy` INTEGER NOT NULL,
            `scaled` INTEGER NOT NULL
        ) STRICT"#,
        (),
    ).unwrap();
    conn.execute(
        r#"
        CREATE TABLE IF NOT EXISTS `test` (
            `row_id` INTEGER PRIMARY KEY,
            `hash` TEXT NOT NULL UNIQUE,
            `image_blob` BLOB NOT NULL,
            `noisy` INTEGER NOT NULL,
            `scaled` INTEGER NOT NULL
        ) STRICT"#,
        (),
    ).unwrap();
    let conn_queue = SegQueue::new();
    conn_queue.push(conn);

    let second_arg = std::env::args().nth(2);
    let third_arg = std::env::args().nth(3);
    match command {
        "append" => {
            let Some(split) = second_arg else {
                eprintln!("No split provided");
                return;
            };
            let Some(path) = third_arg else {
                eprintln!("No path provided");
                return;
            };
            if !matches!(split.as_str(), "train" | "test") {
                eprintln!("Invalid split. Must be 'train' or 'test'");
                return;
            }
            recursive_iter(
                path.as_ref(),
                &|path, blob_fn| {
                    let conn = conn_queue.pop().unwrap_or_else(|| Connection::open(SQLITE_DATABASE).unwrap());
                    let md5_hash = hash_file(&path);
                    let md5_hash = std::str::from_utf8(&md5_hash).unwrap();
                    let insert_cmd = format!("INSERT OR IGNORE INTO {split} (`hash`, `image_blob`, `noisy`, `scaled`) VALUES (?1, ?2, 0, ?3)");
                    let blob = blob_fn();
                    loop {
                        if let Err(e) = conn.execute(
                            &insert_cmd,
                            params![md5_hash, blob, 0]
                        ) {
                            if let Some(ErrorCode::DatabaseBusy) = e.sqlite_error_code() {
                                std::thread::sleep(std::time::Duration::from_millis(100));
                                continue;
                            }
                            panic!("{e}");
                        }
                        break;
                    }
                    conn_queue.push(conn);
                }
            );
        }
        "infer" => {
            let Some(path) = second_arg else {
                eprintln!("No path provided");
                return;
            };
            let device = WgpuDevice::default();
            let mut model = model_config().init::<MyBackend>(&device);
            model.load_compact_record_file("artifacts/isthatarock/handwritten/model.mpk".into(), &device);
            let mut image = image::open(path).unwrap();
            image = image.resize_exact(28, 28, FilterType::CatmullRom);
            validate_color(&mut image);
            let mut image = image.to_luma32f();
            let mut output = vec![];
            model.forward_slice(&image, &mut output, &device);
            image.copy_from_slice(&output);
            DynamicImage::from(image).to_rgba8().save("valid.webp").unwrap();
        }
        "cluster" => {
            let device = WgpuDevice::default();
            let mut model = model_config().init::<MyBackend>(&device);
            model.load_compact_record_file("artifacts/isthatarock/handwritten/model.mpk".into(), &device);

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
                    latent_points.push(
                        point.try_into().unwrap()
                    );
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
                    Entry::Vacant(entry) => { entry.insert(1usize); },
                }
            }
            for (id, count) in ids {
                println!("{id}: {count}");
            }
            println!("noise: {noise}");
        }
        _ => {
            eprintln!("Unknown command");
            return;
        }
    }
}

fn hash_file(file: &Path) -> [u8; 32] {
    let mut md5_ctx = md5::Context::new();
    std::io::copy(&mut std::fs::File::open(file).unwrap(), &mut md5_ctx).unwrap();
    let md5_hash = md5_ctx.finalize().0;
    let mut out = [0u8; 32];
    hex::encode_to_slice(md5_hash, &mut out).unwrap();
    out
}
