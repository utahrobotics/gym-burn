use std::{fmt, marker::PhantomData};

use crate::{FromConfig, SimpleForwardable, serde_fix::{Conv2dConfig, ConvTranspose2dConfig, LinearConfig}};

use burn::{
    Tensor,
    config::Config,
    module::{AutodiffModule, Module},
    nn::{
        BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Gelu, Linear, Sigmoid,
        Tanh,
        conv::{Conv2d, ConvTranspose2d},
        interpolate::{Interpolate2d, Interpolate2dConfig, InterpolateMode},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
    },
    prelude::Backend,
    record::Record,
    tensor::backend::AutodiffBackend,
};

use serde::{Deserialize, Serialize};

#[derive(Module, Debug)]
pub struct SimpleLumaImageEncoder<B: Backend> {
    encoder_convolutions: Vec<(Conv2d<B>, Option<BatchNorm<B>>)>,
    adaptive_avg_pooling: Option<AdaptiveAvgPool2d>,
    encoder_linears: Vec<(Linear<B>, Option<BatchNorm<B>>)>,

    gelu: Gelu,
    tanh: Tanh,
    linear_dropout: Dropout,
    conv_dropout: Dropout,
}

impl<B: Backend> FromConfig<B> for SimpleLumaImageEncoder<B> {
    type Config = SimpleLumaImageEncoderConfig;

    fn init(config: Self::Config, device: &B::Device) -> Self {
        config.init(device)
    }
}

impl<B: Backend> SimpleForwardable<B, 3, 2> for SimpleLumaImageEncoder<B> {
    fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, image_width, image_height] = images.dims();

        let mut x = if self.encoder_convolutions.is_empty() {
            images.reshape([batch_size as i32, -1])
        } else {
            let mut x = images.reshape([batch_size, 1, image_width, image_height]);

            for (cnn, norm) in &self.encoder_convolutions {
                x = cnn.forward(x);
                x = self.gelu.forward(x);
                if let Some(norm) = norm {
                    x = norm.forward(x);
                }
                x = self.conv_dropout.forward(x);
            }

            if let Some(adaptive_avg_pooling) = &self.adaptive_avg_pooling {
                x = adaptive_avg_pooling.forward(x);
            }

            x.reshape([batch_size as i32, -1])
        };

        for (i, (linear, norm)) in self.encoder_linears.iter().enumerate() {
            x = linear.forward(x);
            if let Some(norm) = norm {
                x = norm.forward(x);
            }

            if i < self.encoder_linears.len() - 1 {
                x = self.gelu.forward(x);
                x = self.linear_dropout.forward(x);
            } else {
                x = self.tanh.forward(x);
            }
        }

        x
    }
}

#[derive(Debug, Config)]
pub struct SimpleLumaImageEncoderConfig {
    pub encoder_convolutions: Vec<(Conv2dConfig, bool)>,
    pub adaptive_avg_pooling: Option<AdaptiveAvgPool2dConfig>,
    pub encoder_linears: Vec<(LinearConfig, bool)>,

    pub linear_dropout: DropoutConfig,
    pub conv_dropout: DropoutConfig,
}

impl SimpleLumaImageEncoderConfig {
    pub fn init<B: Backend>(self, device: &B::Device) -> SimpleLumaImageEncoder<B> {
        SimpleLumaImageEncoder {
            encoder_convolutions: self
                .encoder_convolutions
                .into_iter()
                .map(|(conv_config, bn_config)| {
                    (
                        conv_config.init(device),
                        bn_config.then(|| BatchNormConfig::new(conv_config.channels[1]).init(device))
                    )
                })
                .collect(),
            adaptive_avg_pooling: self.adaptive_avg_pooling.map(|x| x.init()),
            encoder_linears: self
                .encoder_linears
                .into_iter()
                .map(|(linear_config, bn_config)| {
                    (
                        linear_config.init(device),
                        bn_config.then(|| BatchNormConfig::new(linear_config.d_output).init(device))
                    )
                })
                .collect(),
            gelu: Gelu::new(),
            tanh: Tanh::new(),
            linear_dropout: self.linear_dropout.init(),
            conv_dropout: self.conv_dropout.init(),
        }
    }
}

#[derive(Module, Debug)]
pub struct SimpleLumaImageDecoder<B: Backend> {
    decoder_linears: Vec<(Linear<B>, Option<BatchNorm<B>>)>,
    cnn_input_shape: Option<[usize; 2]>,
    decoder_convolutions: Vec<(ConvTranspose2d<B>, Option<BatchNorm<B>>)>,

    gelu: Gelu,
    sigmoid: Sigmoid,
    linear_dropout: Dropout,
    conv_dropout: Dropout,
    output_size: [usize; 2],
    interpolate: Option<Interpolate2d>,
}

impl<B: Backend> FromConfig<B> for SimpleLumaImageDecoder<B> {
    type Config = SimpleLumaImageDecoderConfig;

    fn init(config: Self::Config, device: &B::Device) -> Self {
        config.init(device)
    }
}

impl<B: Backend> SimpleForwardable<B, 2, 3> for SimpleLumaImageDecoder<B> {
    fn forward(&self, latents: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch_size, _] = latents.dims();
        let mut x = latents;

        for (i, (linear, norm)) in self.decoder_linears.iter().enumerate() {
            x = linear.forward(x);
            if let Some(norm) = norm {
                x = norm.forward(x);
            }
            if i == self.decoder_linears.len() - 1 && self.decoder_convolutions.is_empty() {
                break;
            }
            x = self.gelu.forward(x);
            x = self.linear_dropout.forward(x);
        }

        let mut x = if let Some((first_cnn, _)) = self.decoder_convolutions.first() {
            let mut x = if let Some([width, height]) = self.cnn_input_shape {
                x.reshape([batch_size, first_cnn.channels[0] as usize, width, height])
            } else {
                let width = (x.dims()[1] as f64).sqrt().ceil() as i32;
                x.reshape([batch_size as i32, first_cnn.channels[0] as i32, width, -1])
            };
            for (i, (cnn, norm)) in self.decoder_convolutions.iter().enumerate() {
                x = cnn.forward(x);
                if let Some(norm) = norm {
                    x = norm.forward(x);
                }
                if i == self.decoder_convolutions.len() - 1 {
                    break;
                }
                x = self.gelu.forward(x);
                x = self.linear_dropout.forward(x);
            }
            let [_, channels, width, height] = x.dims();
            assert_eq!(
                channels, 1,
                "Final CNN layer does not output exactly 1 channel"
            );
            x.reshape([batch_size, width, height])
        } else {
            let [width, height] = self.output_size;
            let count = width * height;
            let [_, actual_count] = x.dims();
            let actual_width = (actual_count as f64 / count as f64 * width as f64).round();

            x.reshape([batch_size as i32, actual_width as i32, -1])
        };

        x = self.sigmoid.forward(x);

        if let Some(interpolate) = &self.interpolate {
            let [_, width, height] = x.dims();
            interpolate
                .forward(x.reshape([batch_size, 1, width, height]))
                .reshape([batch_size, width, height])
        } else {
            x
        }
    }
}

#[derive(Debug, Config)]
pub struct SimpleLumaImageDecoderConfig {
    pub decoder_linears: Vec<(LinearConfig, bool)>,
    pub cnn_input_shape: Option<[usize; 2]>,
    pub decoder_convolutions: Vec<(ConvTranspose2dConfig, bool)>,

    pub linear_dropout: DropoutConfig,
    pub conv_dropout: DropoutConfig,
    pub output_size: [usize; 2],
    pub interpolate: Option<InterpolateMode>,
}

impl SimpleLumaImageDecoderConfig {
    pub fn init<B: Backend>(self, device: &B::Device) -> SimpleLumaImageDecoder<B> {
        SimpleLumaImageDecoder {
            decoder_linears: self
                .decoder_linears
                .into_iter()
                .map(|(config, norm)| {
                    let linear = config.init(device);
                    let norm = norm.then(|| BatchNormConfig::new(config.d_output).init(device));
                    (linear, norm)
                })
                .collect(),
            cnn_input_shape: self.cnn_input_shape,
            decoder_convolutions: self
                .decoder_convolutions
                .into_iter()
                .map(|(config, norm)| {
                    let conv = config.init(device);
                    let norm = norm.then(|| BatchNormConfig::new(config.channels[1]).init(device));
                    (conv, norm)
                })
                .collect(),
            gelu: Gelu::new(),
            sigmoid: Sigmoid::new(),
            linear_dropout: self.linear_dropout.init(),
            conv_dropout: self.conv_dropout.init(),
            output_size: self.output_size,
            interpolate: self.interpolate.map(|x| Interpolate2dConfig::new().with_mode(x).with_output_size(Some(self.output_size)).init()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SimpleAutoEncoder<B, E, D, const N_I: usize, const N_D: usize> {
    pub encoder: E,
    pub decoder: D,
    _phantom: PhantomData<B>,
}

impl<B, E, D, const N_I: usize, const N_D: usize> fmt::Display
    for SimpleAutoEncoder<B, E, D, N_I, N_D>
where
    E: fmt::Display,
    D: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SimpleAutoEncoder with input dimension {} and latent dimension {}\nEncoder: {}\nDecoder: {}",
            N_I, N_D, self.encoder, self.decoder
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleAutoEncoderConfig<E, D> {
    pub encoder_config: E,
    pub decoder_config: D,
}

impl<E: Config, D: Config> Config for SimpleAutoEncoderConfig<E, D> {}

impl<X, Y> SimpleAutoEncoderConfig<X, Y> {
    pub fn init<B, const N_I: usize, const N_D: usize, E, D>(
        self,
        device: &B::Device,
    ) -> SimpleAutoEncoder<B, E, D, N_I, N_D>
    where
        B: Backend,
        E: SimpleForwardable<B, N_I, N_D, Config = X>,
        D: SimpleForwardable<B, N_D, N_I, Config = Y>,
    {
        SimpleAutoEncoder::init(self, device)
    }
}

/// An Image AutoEncoder with a linear Latent Space
pub type LinearImageAutoEncoder<B> =
    SimpleAutoEncoder<B, SimpleLumaImageEncoder<B>, SimpleLumaImageDecoder<B>, 3, 2>;
pub type LinearImageAutoEncoderConfig =
    SimpleAutoEncoderConfig<SimpleLumaImageEncoderConfig, SimpleLumaImageDecoderConfig>;

impl<
    B: Backend,
    const N_I: usize,
    const N_D: usize,
    E: SimpleForwardable<B, N_I, N_D>,
    D: SimpleForwardable<B, N_D, N_I>,
> FromConfig<B> for SimpleAutoEncoder<B, E, D, N_I, N_D>
{
    type Config = SimpleAutoEncoderConfig<E::Config, D::Config>;

    fn init(config: Self::Config, device: &B::Device) -> Self {
        SimpleAutoEncoder {
            encoder: E::init(config.encoder_config, device),
            decoder: D::init(config.decoder_config, device),
            _phantom: PhantomData,
        }
    }
}
impl<
    B: Backend,
    const N_I: usize,
    const N_D: usize,
    E: SimpleForwardable<B, N_I, N_D>,
    D: SimpleForwardable<B, N_D, N_I>,
> SimpleForwardable<B, N_I, N_I> for SimpleAutoEncoder<B, E, D, N_I, N_D>
{
    fn forward(&self, images: Tensor<B, N_I>) -> Tensor<B, N_I> {
        self.decoder.forward(self.encoder.forward(images))
    }
}

#[derive(Debug, Clone)]
pub struct AutoEncoderRecord<E, D> {
    pub encoder_record: E,
    pub decoder_record: D,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct AutoEncoderRecordItem<E, D> {
    pub encoder_item: E,
    pub decoder_item: D,
}

impl<B: Backend, E: Record<B>, D: Record<B>> Record<B> for AutoEncoderRecord<E, D> {
    type Item<S: burn::record::PrecisionSettings> = AutoEncoderRecordItem<E::Item<S>, D::Item<S>>;

    fn into_item<S: burn::record::PrecisionSettings>(self) -> Self::Item<S> {
        AutoEncoderRecordItem {
            encoder_item: self.encoder_record.into_item(),
            decoder_item: self.decoder_record.into_item(),
        }
    }

    fn from_item<S: burn::record::PrecisionSettings>(
        item: Self::Item<S>,
        device: &<B as Backend>::Device,
    ) -> Self {
        Self {
            encoder_record: Record::from_item(item.encoder_item, device),
            decoder_record: Record::from_item(item.decoder_item, device),
        }
    }
}

impl<B: Backend, E: Module<B>, D: Module<B>, const N_I: usize, const N_D: usize> Module<B>
    for SimpleAutoEncoder<B, E, D, N_I, N_D>
{
    type Record = AutoEncoderRecord<E::Record, D::Record>;

    fn collect_devices(&self, mut devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
        devices = self.encoder.collect_devices(devices);
        self.decoder.collect_devices(devices)
    }

    fn fork(self, device: &<B as Backend>::Device) -> Self {
        Self {
            encoder: self.encoder.fork(device),
            decoder: self.decoder.fork(device),
            _phantom: PhantomData,
        }
    }

    fn to_device(self, device: &<B as Backend>::Device) -> Self {
        Self {
            encoder: self.encoder.to_device(device),
            decoder: self.decoder.to_device(device),
            _phantom: PhantomData,
        }
    }

    fn visit<Visitor: burn::module::ModuleVisitor<B>>(&self, visitor: &mut Visitor) {
        self.encoder.visit(visitor);
        self.decoder.visit(visitor);
    }

    fn map<Mapper: burn::module::ModuleMapper<B>>(self, mapper: &mut Mapper) -> Self {
        Self {
            encoder: self.encoder.map(mapper),
            decoder: self.decoder.map(mapper),
            _phantom: PhantomData,
        }
    }

    fn load_record(self, record: Self::Record) -> Self {
        Self {
            encoder: self.encoder.load_record(record.encoder_record),
            decoder: self.decoder.load_record(record.decoder_record),
            _phantom: PhantomData,
        }
    }

    fn into_record(self) -> Self::Record {
        AutoEncoderRecord {
            encoder_record: self.encoder.into_record(),
            decoder_record: self.decoder.into_record(),
        }
    }
}

impl<
    B: AutodiffBackend,
    E: AutodiffModule<B>,
    D: AutodiffModule<B>,
    const N_I: usize,
    const N_D: usize,
> AutodiffModule<B> for SimpleAutoEncoder<B, E, D, N_I, N_D>
{
    type InnerModule = SimpleAutoEncoder<B::InnerBackend, E::InnerModule, D::InnerModule, N_I, N_D>;

    fn valid(&self) -> Self::InnerModule {
        SimpleAutoEncoder {
            encoder: self.encoder.valid(),
            decoder: self.decoder.valid(),
            _phantom: PhantomData,
        }
    }
}
