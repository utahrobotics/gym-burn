#![recursion_limit = "256"]
use std::path::Path;

use burn::{
    Tensor,
    module::Module,
    prelude::Backend,
    record::{CompactRecorder, Recorder},
    tensor::TensorData,
};
use general_models::{Init, SimpleInfer, composite::{autoencoder::{AutoEncoderModelConfig, AutoEncoderModelRecord, vae::{VariationalEncoderModel, VariationalEncoderModelConfig, VariationalEncoderModelRecord}}, image::{Conv2dLinearModel, Conv2dLinearModelConfig, Conv2dLinearModelRecord, LinearConvTranspose2dModel, LinearConvTranspose2dModelConfig}}, error::LoadModelError};
use image::{ImageBuffer, Luma, buffer::ConvertBuffer};
use utils::parse_json_file;

pub type VaeModel<B> = VariationalEncoderModel<B, Conv2dLinearModel<B>>;
pub type VaeConfig = VariationalEncoderModelConfig<Conv2dLinearModelConfig>;
pub type VaeRecord<B> = VariationalEncoderModelRecord<B, Conv2dLinearModelRecord<B>>;

pub type Model<B> = Conv2dLinearModel<B>;
pub type Config = Conv2dLinearModelConfig;
pub type Record<B> = Conv2dLinearModelRecord<B>;

pub struct ImageEncoder<B: Backend, M> {
    encoder: M,
    device: B::Device
}

impl<B: Backend> ImageEncoder<B, Model<B>> {
    pub fn load(
        autoencoder_config: impl AsRef<Path>,
        encoder_weights: impl AsRef<Path>,
        device: &B::Device,
    ) -> Result<Self, LoadModelError> {
        let autoencoder_config: AutoEncoderModelConfig<Config, LinearConvTranspose2dModelConfig> =
            parse_json_file(autoencoder_config)?;
        let mut encoder = autoencoder_config.encoder.init(device);
        let record: AutoEncoderModelRecord<B, Model<B>, LinearConvTranspose2dModel<B>> = CompactRecorder::new().load(encoder_weights.as_ref().into(), device)?;
        encoder = encoder.load_record(record.encoder);
        Ok(Self {
            encoder,
            device: device.clone(),
        })
    }
}

// impl<B: Backend> ImageEncoder<B, VaeModel<B>> {
//     pub fn load(
//         autoencoder_config: impl AsRef<Path>,
//         encoder_weights: impl AsRef<Path>,
//         device: &B::Device,
//     ) -> Result<Self, LoadModelError> {
//         let autoencoder_config: AutoEncoderModelConfig<VaeConfig, LinearConvTranspose2dModelConfig> =
//             parse_json_file(autoencoder_config)?;
//         let mut encoder = autoencoder_config.encoder.init(device);
//         encoder = encoder
//             .load_record(CompactRecorder::new().load(encoder_weights.as_ref().into(), device)?);
//         Ok(Self {
//             encoder,
//             device: device.clone(),
//         })
//     }
// }

impl<B: Backend, M> ImageEncoder<B, M> {
    pub fn get_encoder(&self) -> &M {
        &self.encoder
    }
}

impl<B: Backend, M: SimpleInfer<B, 4, 2>> ImageEncoder<B, M> {
    /// Encodes the given Luma images with edge-detection already performed.
    ///
    /// # Panic
    /// Panics if a slice is not exactly 28 * 28 floats.
    pub fn encode_sized_edges<T>(&self, iter: impl IntoIterator<Item = T>) -> Encodings
    where
        T: AsRef<[f32]>,
    {
        let tensors = iter
            .into_iter()
            .map(|x| TensorData::from(x.as_ref()))
            .map(|data| Tensor::<B, 1>::from_data(data, &self.device))
            .map(|tensor| tensor.reshape([1, 1, 28, 28]))
            .collect();
        let tensor = Tensor::cat(tensors, 0);
        let latent_dims = self.encoder.infer(tensor);
        let [_, latent_size] = latent_dims.dims();
        Encodings {
            data: latent_dims.into_data().into_vec::<f32>().unwrap(),
            latent_size,
        }
    }

    /// Encodes the given Luma images, after performing edge-detection using canny.
    ///
    /// This step *may* be quite slow. There are a lot of heap allocations.
    ///
    /// # Panic
    /// Panics if a slice is not exactly 28 * 28 floats.
    pub fn encode_sized<T>(
        &self,
        iter: impl IntoIterator<Item = T>,
        low_threshold: f32,
        high_threshold: f32,
    ) -> Encodings
    where
        T: Into<Vec<f32>>,
    {
        self.encode_sized_edges(
            iter.into_iter()
                .map(Into::into)
                .map(|buf| ImageBuffer::<Luma<f32>, _>::from_raw(28, 28, buf).unwrap())
                .map(|img| img.convert())
                .map(|img| imageproc::edges::canny(&img, low_threshold, high_threshold))
                .map(|img| img.convert())
                .map(|img: ImageBuffer<Luma<f32>, Vec<_>>| img.into_vec()),
        )
    }
}

#[derive(Debug, Clone)]
pub struct Encodings {
    data: Vec<f32>,
    latent_size: usize,
}

impl Encodings {
    pub fn get_latent_size(&self) -> usize {
        self.latent_size
    }

    pub fn iter(&self) -> std::slice::Chunks<'_, f32> {
        self.data.chunks(self.latent_size)
    }
}

impl<'a> IntoIterator for &'a Encodings {
    type Item = &'a [f32];

    type IntoIter = std::slice::Chunks<'a, f32>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
