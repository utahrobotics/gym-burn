use std::{path::PathBuf, sync::Arc, time::SystemTime};

use burn::{data::{dataloader::batcher::Batcher, dataset::Dataset}, prelude::*, record::{CompactRecorder, Recorder}};
use general_models::{Init, SimpleInfer};
use image::{DynamicImage, ImageBuffer, Luma, LumaA, Rgb, Rgba};
use rand::{Rng, SeedableRng};
use rustc_hash::FxHashSet;
use serde::de::DeserializeOwned;
use utils::parse_json_file;

use crate::{batches::{AutoEncoderImageBatch, AutoEncoderImageBatcher, AutoEncoderImageItem}, dataset::{SqliteDataset, SqliteDatasetConfig}};

pub fn infer_image_autoencoder<B, M, C>(
    count: usize,
    weights_path: PathBuf,
    config_path: PathBuf,
    get_input_channels: impl FnOnce(&M) -> usize,
    device: &B::Device
)
where 
    B: Backend,
    M: SimpleInfer<B, 4, 4>,
    C: Init<B, Output = M> + DeserializeOwned
{
    let test_dataset_config: SqliteDatasetConfig = parse_json_file("test-data").unwrap();
    let mut rng = rand::rngs::SmallRng::seed_from_u64(
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    );

    let model_config: C =
        parse_json_file(config_path).unwrap();
    let mut model: M = model_config.init(device);
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

    let batcher = AutoEncoderImageBatcher {
        channels: get_input_channels(&model)
    };
    let items: Vec<_> = set.iter().map(|&i| test_dataset.get(i).unwrap()).collect();

    let width = items.first().unwrap().input_width as u32;
    let mosaic_width = width as u32 * 2;
    let height = items.first().unwrap().input_height as u32;
    let mosaic_height = height * count as u32;

    let batch: AutoEncoderImageBatch<B> = batcher.batch(items, device);
    let now = quanta::Instant::now();
    let actual = model.infer(batch.input.clone()).clamp(0.0, 1.0);
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

    let img: Option<DynamicImage> = match batcher.channels {
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
        x => unreachable!("Unexpected number of input channels: {x}"),
    };

    img
        .expect("Dimensions mismatch. The auto encoder must accept only one size of image and output that same size")
        .into_rgb8()
        .save("inference.webp")
        .unwrap();
}
