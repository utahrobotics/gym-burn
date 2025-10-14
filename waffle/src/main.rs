use std::{io::Cursor, num::NonZeroUsize, path::PathBuf, time::SystemTime};

use burn::{
    Tensor, backend::Autodiff, data::dataloader::DataLoaderBuilder, module::Module, nn::loss::{MseLoss, Reduction}, optim::AdamConfig, prelude::Backend, record::CompactRecorder, tensor::backend::AutodiffBackend, train::{
        LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep, metric::LossMetric,
    }
};
use general_dataset::{
    SqliteDataset, SqliteDatasetConfig, StatefulBatcher,
    burn_dataset::{BurnBatcher, SqliteBurnDataset},
    cache::initialize_cache,
    def_cache,
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
    ImageBuffer, ImageDecoder, Luma, Rgb, buffer::ConvertBuffer, codecs::webp::WebPDecoder,
};
use serde::Deserialize;
use utils::parse_json_file;

#[derive(Deserialize, Debug)]
pub struct TrainingConfig {
    pub artifact_dir: PathBuf,
    pub num_epochs: usize,
    pub batch_size: usize,
    pub training_dataset: SqliteDatasetConfig,
    pub testing_dataset: SqliteDatasetConfig,
    pub seed: Option<u64>,
}

type Model<B> = AutoEncoderModel<B, Conv2dLinearModel<B>, LinearConvTranspose2dModel<B>>;

const CHALLENGE_COUNT: usize = 5;

#[derive(Module, Debug)]
pub struct Trainable<B: Backend> {
    model: Model<B>,
}

const EPSILON: f64 = 1e-7;

pub fn bce_float_loss<B: Backend, const D: usize>(
    expected: Tensor<B, D>,
    mut actual: Tensor<B, D>,
) -> Tensor<B, 1> {
    actual = actual.clamp(EPSILON, 1.0 - EPSILON);
    let loss = expected.clone() * actual.clone().log() + (-expected + 1.0) * (-actual + 1.0).log();
    -loss.mean()
}

impl<B: AutodiffBackend> TrainStep<AutoEncoderImageBatch<B>, RegressionOutput<B>> for Trainable<B> {
    fn step(&self, batch: AutoEncoderImageBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let actual = self.model.train(batch.input);
        let actual = actual.flatten(1, 3);
        let expected = batch.expected.flatten(1, 3);
        let item = RegressionOutput::new(
            // bce_float_loss(actual.clone(), expected.clone()),
            MseLoss::new().forward(actual.clone(), expected.clone(), Reduction::Auto),
            actual,
            expected,
        );

        TrainOutput::new(&self.model, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<AutoEncoderImageBatch<B>, RegressionOutput<B>> for Trainable<B> {
    fn step(&self, batch: AutoEncoderImageBatch<B>) -> RegressionOutput<B> {
        let actual = self.model.infer(batch.input);
        let actual = actual.flatten(1, 3);
        let expected = batch.expected.flatten(1, 3);
        let item = RegressionOutput::new(
            // bce_float_loss(actual.clone(), expected.clone()),
            MseLoss::new().forward(actual.clone(), expected.clone(), Reduction::Auto),
            actual,
            expected,
        );
        item
    }
}

def_cache!(TRAINING_CACHE AutoEncoderImageItem);
def_cache!(TESTING_CACHE AutoEncoderImageItem);

fn main() {
    type Backend = general_models::wgpu::WgpuBackend;
    type AutodiffBackend = Autodiff<Backend>;

    let device = general_models::wgpu::get_device();
    let training_config: TrainingConfig =
        parse_json_file("training").expect("Expected valid training.json");

    let model_config: AutoEncoderModelConfig<
        Conv2dLinearModelConfig,
        LinearConvTranspose2dModelConfig,
    > = parse_json_file("model").expect("Expected valid model.json");

    let secs = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let artifact_dir = training_config.artifact_dir.join(secs.to_string());
    std::fs::create_dir_all(&artifact_dir).expect("Expected artifact dir to be creatable");

    let training_dataset: SqliteDataset = training_config
        .training_dataset
        .try_into()
        .expect("Expected valid training dataset config");

    let testing_dataset: SqliteDataset = training_config
        .testing_dataset
        .clone()
        .try_into()
        .expect("Expected valid training dataset config");

    let model: Model<AutodiffBackend> = model_config.clone().init(device);

    let seed = training_config.seed.unwrap_or(secs);
    <Backend as burn::prelude::Backend>::seed(device, seed);

    initialize_cache(&TRAINING_CACHE, NonZeroUsize::new(1024).unwrap());
    initialize_cache(&TESTING_CACHE, NonZeroUsize::new(1024).unwrap());

    let dataloader_train = DataLoaderBuilder::new(BurnBatcher::new(|| {
        AutoEncoderImageBatcher::new(1, device.clone())
    }))
    .batch_size(training_config.batch_size)
    .shuffle(seed)
    .num_workers(4)
    .build(SqliteBurnDataset::new(
        training_dataset,
        &TRAINING_CACHE,
        NonZeroUsize::new(128).unwrap(),
    ));

    let dataloader_test = DataLoaderBuilder::new(BurnBatcher::new(|| {
        AutoEncoderImageBatcher::new(1, device.clone())
    }))
    .batch_size(training_config.batch_size)
    .shuffle(seed)
    .num_workers(4)
    .build(SqliteBurnDataset::new(
        testing_dataset,
        &TESTING_CACHE,
        NonZeroUsize::new(128).unwrap(),
    ));

    let optim = AdamConfig::new().init();

    let learner = LearnerBuilder::new(&artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(training_config.num_epochs)
        .summary()
        .build(Trainable { model }, optim.clone(), 0.0001);
    let training_result = learner.fit(dataloader_train, dataloader_test);
    let model = training_result.model.model;

    let mut testing_batcher = AutoEncoderImageBatcher::new(1, device.clone());
    let output_width = 28usize;
    let output_height = 28usize;
    let mut rng = rand::rng();
    let mut input_images = vec![];

    let testing_dataset: SqliteDataset =
        training_config.testing_dataset.clone().try_into().unwrap();

    for _ in 0..CHALLENGE_COUNT {
        let item: AutoEncoderImageItem = testing_dataset.pick_random(&mut rng);
        input_images.push(item.webp_input.clone());
        testing_batcher.ingest(item);
    }
    let batch = testing_batcher.finish();
    let reconstructed = model.infer(batch.input.clone());

    model
        .save_file(artifact_dir.join("model.mpk"), &CompactRecorder::new())
        .unwrap();

    let reconstructed_images: Vec<_> = reconstructed
        .iter_dim(0)
        .map(|tensor| {
            let [_, _, width, height] = tensor.dims();
            let buf = tensor.into_data().into_vec::<f32>().unwrap();
            let img =
                ImageBuffer::<Luma<f32>, _>::from_raw(width as u32, height as u32, buf).unwrap();
            let img: ImageBuffer<Rgb<u8>, Vec<_>> = img.convert();

            img
        })
        .collect();

    let mosaic_width = output_width as u32 * 2;
    let mosaic_height = output_height as u32 * CHALLENGE_COUNT as u32;
    let mut pixels = Vec::with_capacity(mosaic_width as usize * mosaic_height as usize);
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
        .save(artifact_dir.join(format!("infer.webp")))
        .expect("Expected inference image to be saveable");
}
