use std::marker::PhantomData;

use crate::SimpleForwardable;

use burn::{
    Tensor, module::Module, nn::{
        BatchNorm, Dropout, Gelu, Linear, Tanh, conv::{Conv2d, ConvTranspose2d}, interpolate::Interpolate2d, pool::AdaptiveAvgPool2d
    }, prelude::Backend, record::Record
};

use serde::{Serialize, Deserialize};

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

#[derive(Module, Debug)]
pub struct SimpleLumaImageDecoder<B: Backend> {
    decoder_linears: Vec<(Linear<B>, Option<BatchNorm<B>>)>,
    cnn_input_shape: Option<[usize; 2]>,
    decoder_convolutions: Vec<(ConvTranspose2d<B>, Option<BatchNorm<B>>)>,

    gelu: Gelu,
    linear_dropout: Dropout,
    conv_dropout: Dropout,
    output_size: [usize; 2],
    interpolate: Option<Interpolate2d>,
}

impl<B: Backend> SimpleForwardable<B, 2, 3> for SimpleLumaImageDecoder<B> {
    fn forward(&self, latents: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch_size, _] = latents.dims();
        let mut x = latents;

        for (linear, norm) in &self.decoder_linears {
            x = linear.forward(x);
            if let Some(norm) = norm {
                x = norm.forward(x);
            }
            x = self.gelu.forward(x);
            x = self.linear_dropout.forward(x);
        }

        let x = if let Some((first_cnn, _)) = self.decoder_convolutions.first() {
            let mut x = if let Some([width, height]) = self.cnn_input_shape {
                x.reshape([batch_size, first_cnn.channels[0] as usize, width, height])
            } else {
                let width = (x.dims()[1] as f64).sqrt().ceil() as i32;
                x.reshape([batch_size as i32, first_cnn.channels[0] as i32, width, -1])
            };
            for (cnn, norm) in &self.decoder_convolutions {
                x = cnn.forward(x);
                if let Some(norm) = norm {
                    x = norm.forward(x);
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


#[derive(Debug, Clone)]
pub struct SimpleAutoEncoder<B, E, D, const N_I: usize, const N_D: usize> {
    pub encoder: E,
    pub decoder: D,
    _phantom: PhantomData<B>
}

/// An Image AutoEncoder with a linear Latent Space
pub type LinearImageAutoEncoder<B> = SimpleAutoEncoder<B, Linear<B>, Linear<B>, 3, 2>;


impl<B: Backend, const N_I: usize, const N_D: usize, E: SimpleForwardable<B, N_I, N_D>, D: SimpleForwardable<B, N_D, N_I>> SimpleForwardable<B, N_I, N_I> for SimpleAutoEncoder<B, E, D, N_I, N_D> {
    fn forward(&self, images: Tensor<B, N_I>) -> Tensor<B, N_I> {
        self.decoder.forward(self.encoder.forward(images))
    }
}


#[derive(Debug, Clone)]
pub struct AutoEncoderRecord<E, D> {
    pub encoder_record: E,
    pub decoder_record: D
}


#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct AutoEncoderRecordItem<E, D> {
    pub encoder_item: E,
    pub decoder_item: D
}


impl<B: Backend, E: Record<B>, D: Record<B>> Record<B> for AutoEncoderRecord<E, D> {
    type Item<S: burn::record::PrecisionSettings> = AutoEncoderRecordItem<E::Item<S>, D::Item<S>>;

    fn into_item<S: burn::record::PrecisionSettings>(self) -> Self::Item<S> {
        AutoEncoderRecordItem {
            encoder_item: self.encoder_record.into_item(),
            decoder_item: self.decoder_record.into_item(),
        }
    }

    fn from_item<S: burn::record::PrecisionSettings>(item: Self::Item<S>, device: &<B as Backend>::Device) -> Self {
        Self {
            encoder_record: Record::from_item(item.encoder_item, device),
            decoder_record: Record::from_item(item.decoder_item, device)
        }
    }
}


impl<B: Backend, E: Module<B>, D: Module<B>, const N_I: usize, const N_D: usize> Module<B> for SimpleAutoEncoder<B, E, D, N_I, N_D> {
    type Record = AutoEncoderRecord<E::Record, D::Record>;

    fn collect_devices(&self, mut devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
        devices = self.encoder.collect_devices(devices);
        self.decoder.collect_devices(devices)
    }

    fn fork(self, device: &<B as Backend>::Device) -> Self {
        Self {
            encoder: self.encoder.fork(device),
            decoder: self.decoder.fork(device),
            _phantom: PhantomData
        }
    }

    fn to_device(self, device: &<B as Backend>::Device) -> Self {
        Self {
            encoder: self.encoder.to_device(device),
            decoder: self.decoder.to_device(device),
            _phantom: PhantomData
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
            _phantom: PhantomData
        }
    }

    fn load_record(self, record: Self::Record) -> Self {
        Self {
            encoder: self.encoder.load_record(record.encoder_record),
            decoder: self.decoder.load_record(record.decoder_record),
            _phantom: PhantomData
        }
    }

    fn into_record(self) -> Self::Record {
        AutoEncoderRecord {
            encoder_record: self.encoder.into_record(),
            decoder_record: self.decoder.into_record()
        }
    }
}