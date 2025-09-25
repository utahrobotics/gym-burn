use burn::{
    Tensor,
    module::Module,
    nn::{
        BatchNorm, Dropout, Gelu, Linear, Tanh,
        conv::{Conv2d, ConvTranspose2d},
        interpolate::Interpolate2d,
        pool::AdaptiveAvgPool2d,
    },
    prelude::Backend,
};

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

impl<B: Backend> SimpleLumaImageEncoder<B> {
    pub fn encode(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
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

impl<B: Backend> SimpleLumaImageDecoder<B> {
    pub fn decode(&self, latents: Tensor<B, 2>) -> Tensor<B, 3> {
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
