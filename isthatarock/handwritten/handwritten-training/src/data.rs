use std::{io::Cursor, path::Path};

use burn::{
    data::dataloader::batcher::Batcher,
    prelude::Backend,
    tensor::{Tensor, TensorData},
};
use crossbeam::queue::SegQueue;
use image::{imageops::FilterType, ImageFormat};
use rayon::iter::{ParallelBridge, ParallelIterator};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

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
            let mut blob = vec![];
            img.write_to(&mut Cursor::new(&mut blob), ImageFormat::WebP).unwrap();
            blob
        });
        f(&path, blob_fn);
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
                    let mut md5_ctx = md5::Context::new();
                    std::io::copy(&mut std::fs::File::open(path).unwrap(), &mut md5_ctx).unwrap();
                    let md5_hash = md5_ctx.finalize().0;
                    let md5_hash = hex::encode(md5_hash);
                    let insert_cmd = format!("INSERT OR IGNORE INTO {split} (`hash`, `image_blob`, `noisy`, `scaled`) VALUES (?1, ?2, 0, ?3)");
                    conn.execute(
                        &insert_cmd,
                        params![md5_hash, blob_fn(), 0]
                    ).unwrap();
                    conn_queue.push(conn);
                }
            );
        }
        _ => {
            eprintln!("Unknown command");
            return;
        }
    }
}
