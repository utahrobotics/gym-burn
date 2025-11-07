#![recursion_limit = "256"]

use std::{num::NonZeroUsize, path::Path};

use burn::{
    Tensor, module::Module, nn::{interpolate::{Interpolate2dConfig, InterpolateMode}, loss::{MseLoss, Reduction}}, prelude::Backend, record::{CompactRecorder, Recorder}, tensor::Bool,
};
use efficient_pca::PCA;
use general_models::{
    Init, SimpleInfer,
    composite::{
        autoencoder::{AutoEncoderModel, AutoEncoderModelConfig},
        image::{
            Conv2dLinearModel, Conv2dLinearModelConfig, LinearConvTranspose2dModel,
            LinearConvTranspose2dModelConfig,
        },
    },
    error::LoadModelError,
    wgpu::WgpuBackend,
};
use ndarray::Array2;
use utils::parse_json_file;

pub use general_models::wgpu;
pub use burn;

use crate::pca::load_pca;

pub mod pca;

const IMAGE_WIDTH: usize = 28;

pub struct Detector<B: Backend = WgpuBackend> {
    model: AutoEncoderModel<B, Conv2dLinearModel<B>, LinearConvTranspose2dModel<B>>,
    // device: B::Device,
    pca: Option<PCA>,
    pub batch_size: NonZeroUsize
}

#[derive(Debug)]
pub struct Feature<B: Backend> {
    pub feature_size: usize,
    pub batched: Vec<Tensor<B, 4>>,
    pub latents: Vec<Tensor<B, 2>>,
    pub latents_pca: Vec<Array2<f64>>
}

#[derive(Debug)]
pub struct EncodingOutput<B: Backend> {
    pub features: Vec<Feature<B>>
}

impl<B: Backend> Detector<B> {
    pub fn load(
        autoencoder_config: impl AsRef<Path>,
        encoder_weights: impl AsRef<Path>,
        device: &B::Device,
    ) -> Result<Self, LoadModelError> {
        let autoencoder_config: AutoEncoderModelConfig<
            Conv2dLinearModelConfig,
            LinearConvTranspose2dModelConfig,
        > = parse_json_file(autoencoder_config)?;
        let mut model = autoencoder_config.init(device);
        let record = CompactRecorder::new().load(encoder_weights.as_ref().into(), device)?;
        model = model.load_record(record);
        Ok(Self {
            model,
            pca: None,
            batch_size: NonZeroUsize::new(256).unwrap()
            // device: device.clone(),
        })
    }

    /// Encodes the given tensor by sliding the model across it like a kernel. There will be a kernel for each feature size.
    /// 
    /// The tensor's dimension should be [image width, image height, number of color channels]
    pub fn encode_tensor(&self, tensor: Tensor<B, 3>, feature_sizes: impl IntoIterator<Item = usize>) -> EncodingOutput<B> {
        let [channels, width, height] = tensor.dims();
        let interp = Interpolate2dConfig::new().with_mode(InterpolateMode::Linear).with_output_size(Some([IMAGE_WIDTH, IMAGE_WIDTH])).init();

        let mut features = vec![];
        for feature_size in feature_sizes {
            let mut item_iter = 
                (0..width - feature_size + 1)
                    .into_iter()
                    .flat_map(move |x| {
                        (0..height - feature_size + 1)
                            .into_iter()
                            .map(move |y| (x, y, feature_size))
                    })
                    .map(|(x, y, feature_size)| {
                        tensor.clone().slice(burn::tensor::s![
                            ..,
                            x..x + feature_size,
                            y..y + feature_size,
                        ])
                        .reshape([1, channels, feature_size, feature_size])
                    })
                    .map(|tensor| interp.forward(tensor));
            
            let mut latents_vec = vec![];
            let mut latents_pca_vec = vec![];
            let mut batched_vec = vec![];
            while let Some(item) = item_iter.next() {
                let mut tensors: Vec<_> = (&mut item_iter)
                    .take(self.batch_size.get() - 1)
                    .collect();
                tensors.push(item);
                let batched = Tensor::cat(tensors, 0);

                let (latents, latents_pca) = self.encode_tensor_batch(batched.clone());
                if let Some(latents_pca) = latents_pca {
                    latents_pca_vec.push(latents_pca);
                }
                latents_vec.push(latents);
                batched_vec.push(batched);
            }

            features.push(Feature { feature_size, batched: batched_vec, latents: latents_vec, latents_pca: latents_pca_vec });
        }

        EncodingOutput { features }
    }

    pub fn encode_tensor_batch_raw(&self, tensor: Tensor<B, 4>) -> Tensor<B, 2> {
        self.model.encoder.infer(tensor)
    }

    pub fn set_pca(&mut self, pca: PCA) {
        self.pca.replace(pca);
    }

    pub fn load_pca(&mut self, file: impl AsRef<Path>) -> serde_json::Result<()> {
        self.pca.replace(load_pca(file)?);
        Ok(())
    }

    /// Encodes the batch of tensors that are already the correct size.
    pub fn encode_tensor_batch(&self, tensor: Tensor<B, 4>) -> (Tensor<B, 2>, Option<Array2<f64>>) {
        let raw_encodings = self.encode_tensor_batch_raw(tensor);
        let after_pca = self.pca.as_ref().map(|pca| {
            let shape = raw_encodings.dims();
            let raw_encodings_vec = raw_encodings.clone().into_data().into_vec::<f32>().unwrap();
            let raw_encodings_vec = raw_encodings_vec.into_iter().map(|x| x as f64).collect();
            let raw_encodings_ndarray =
                Array2::from_shape_vec((shape[0], shape[1]), raw_encodings_vec).unwrap();
            pca
                .transform(raw_encodings_ndarray)
                .expect("Expected PCA to already be fitted")
        });

        (
            raw_encodings,
            after_pca
        )
    }

    pub fn decode_latents(&self, latents: Tensor<B, 2>) -> Tensor<B, 4> {
        self.model.decoder.infer(latents)
    }

    // pub fn decode_batch_ndarray(&self, mut arr: Array2<f64>) -> Tensor<B, 4> {
    //     let pca = self.pca.as_ref().expect("Expected PCA to be provided");
    //     arr = pca.
    //     let &[batch_size, latents] = arr.shape() else {
    //         unreachable!();
    //     };
    //     let floats = if let Some(slice) = arr.as_slice() {
    //         slice.to_vec()
    //     } else {
    //         arr.iter().copied().collect()
    //     };
    //     let tensor = Tensor::<B, 2>::from_data(TensorData::new(floats, [batch_size, latents]), &self.device);
    //     self.decode_batch_tensor(tensor)
    // }

    pub fn get_latents_size(&self) -> usize {
        self.model.encoder.linear.get_output_size()
    }
}

/// Peak Signal-to-Noise Ratio between two float tensors whose values are in [0, 1].
/// 
/// - \>40 dB: Excellent quality, differences barely perceptible
/// - 30-40 dB: Good quality, acceptable for most applications
/// - 20-30 dB: Fair quality, noticeable differences
/// - <20 dB: Poor quality, significant degradation
pub fn psnr<B: Backend<FloatElem = f32>, const N: usize>(a: Tensor<B, N>, b: Tensor<B, N>) -> f32 {
    10.0 * (1.0 / MseLoss::new().forward(a, b, Reduction::Mean).into_scalar()).log10()
}

/// Peak Signal-to-Noise Ratio between two batched float tensors whose values are in [0, 1]. It is assumed
/// that the first axis is the batch axis.
/// 
/// - \>40 dB: Excellent quality, differences barely perceptible
/// - 30-40 dB: Good quality, acceptable for most applications
/// - 20-30 dB: Fair quality, noticeable differences
/// - <20 dB: Poor quality, significant degradation
pub fn psnr_batched<B: Backend<FloatElem = f32>, const N: usize>(a: Tensor<B, N>, b: Tensor<B, N>) -> Tensor<B, 1> {
    assert_eq!(a.shape(), b.shape(), "The shapes of a and b are different");
    let batch_size = a.dims()[0];
    let c = a.sub(b).square().mean_dims((1..N).into_iter().collect::<Vec<_>>().as_slice()).reshape([batch_size]);
    let device = c.device();

    c.recip().log() / (Tensor::ones([batch_size], &device) * 10.0).log() * 10.0
}

pub fn psnr_mask<B: Backend>(psnr_level: Tensor<B, 1>, min_level: f32) -> Tensor<B, 1, Bool> {
    psnr_level.greater_elem(min_level)
}

// pub type VaeModel<B> = VariationalEncoderModel<B, Conv2dLinearModel<B>>;
// pub type VaeConfig = VariationalEncoderModelConfig<Conv2dLinearModelConfig>;
// pub type VaeRecord<B> = VariationalEncoderModelRecord<B, Conv2dLinearModelRecord<B>>;

// pub type Model<B> = Conv2dLinearModel<B>;
// pub type Config = Conv2dLinearModelConfig;
// pub type Record<B> = Conv2dLinearModelRecord<B>;

// pub struct ImageEncoder<B: Backend, M> {
//     encoder: M,
//     device: B::Device,
// }

// impl<B: Backend> ImageEncoder<B, Model<B>> {
//     pub fn load(
//         autoencoder_config: impl AsRef<Path>,
//         encoder_weights: impl AsRef<Path>,
//         device: &B::Device,
//     ) -> Result<Self, LoadModelError> {
//         let autoencoder_config: AutoEncoderModelConfig<Config, LinearConvTranspose2dModelConfig> =
//             parse_json_file(autoencoder_config)?;
//         let mut encoder = autoencoder_config.encoder.init(device);
//         let record: AutoEncoderModelRecord<B, Model<B>, LinearConvTranspose2dModel<B>> =
//             CompactRecorder::new().load(encoder_weights.as_ref().into(), device)?;
//         encoder = encoder.load_record(record.encoder);
//         Ok(Self {
//             encoder,
//             device: device.clone(),
//         })
//     }
// }

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

// impl<B: Backend, M> ImageEncoder<B, M> {
//     pub fn get_encoder(&self) -> &M {
//         &self.encoder
//     }
// }

// impl<B: Backend, M: SimpleInfer<B, 4, 2>> ImageEncoder<B, M> {
//     // /// Encodes the given Luma images with edge-detection already performed.
//     // ///
//     // /// # Panic
//     // /// Panics if a slice is not exactly 28 * 28 floats.
//     // pub fn encode_sized_edges<T>(&self, iter: impl IntoIterator<Item = T>) -> Encodings
//     // where
//     //     T: AsRef<[f32]>,
//     // {
//     //     let tensors = iter
//     //         .into_iter()
//     //         .map(|x| TensorData::from(x.as_ref()))
//     //         .map(|data| Tensor::<B, 1>::from_data(data, &self.device))
//     //         .map(|tensor| tensor.reshape([1, 1, 28, 28]))
//     //         .collect();
//     //     let tensor = Tensor::cat(tensors, 0);
//     //     let latent_dims = self.encoder.infer(tensor);
//     //     let [_, latent_size] = latent_dims.dims();
//     //     Encodings {
//     //         data: latent_dims.into_data().into_vec::<f32>().unwrap(),
//     //         latent_size,
//     //     }
//     // }

//     // /// Encodes the given Luma images, after performing edge-detection using canny.
//     // ///
//     // /// This step *may* be quite slow. There are a lot of heap allocations.
//     // ///
//     // /// # Panic
//     // /// Panics if a slice is not exactly 28 * 28 floats.
//     // pub fn encode_sized<T>(
//     //     &self,
//     //     iter: impl IntoIterator<Item = T>,
//     //     low_threshold: f32,
//     //     high_threshold: f32,
//     // ) -> Encodings
//     // where
//     //     T: Into<Vec<f32>>,
//     // {
//     //     self.encode_sized_edges(
//     //         iter.into_iter()
//     //             .map(Into::into)
//     //             .map(|buf| ImageBuffer::<Luma<f32>, _>::from_raw(28, 28, buf).unwrap())
//     //             .map(|img| img.convert())
//     //             .map(|img| imageproc::edges::canny(&img, low_threshold, high_threshold))
//     //             .map(|img| img.convert())
//     //             .map(|img: ImageBuffer<Luma<f32>, Vec<_>>| img.into_vec()),
//     //     )
//     // }
// }

// #[derive(Debug, Clone)]
// pub struct Encodings {
//     data: Vec<f32>,
//     latent_size: usize,
// }

// impl Encodings {
//     pub fn get_latent_size(&self) -> usize {
//         self.latent_size
//     }

//     pub fn iter(&self) -> std::slice::Chunks<'_, f32> {
//         self.data.chunks(self.latent_size)
//     }
// }

// impl<'a> IntoIterator for &'a Encodings {
//     type Item = &'a [f32];

//     type IntoIter = std::slice::Chunks<'a, f32>;

//     fn into_iter(self) -> Self::IntoIter {
//         self.iter()
//     }
// }
