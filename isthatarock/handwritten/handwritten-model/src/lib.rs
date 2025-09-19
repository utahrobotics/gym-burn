use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig}, pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig}, Dropout, DropoutConfig, Gelu, Linear, LinearConfig, Sigmoid, Tanh
    },
    prelude::*, train::RegressionOutput,
};

#[derive(Module, Debug)]
pub struct HandwrittenEncoder<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    pool: AdaptiveAvgPool2d,
    linear1: Linear<B>,
    linear2: Linear<B>,
    conv_dropout: Dropout,
    linear_dropout: Dropout,
    tanh: Tanh,
    gelu: Gelu,
}

impl<B: Backend> HandwrittenEncoder<B> {
    /// # Shapes
    ///   - Images [batch_size, height, width]
    ///   - Output [batch_size, latent_dim]
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = images.dims();

        // Create a channel at the second dimension.
        let x = images.reshape([batch_size, 1, height, width]);

        let x = self.conv1.forward(x); // [batch_size, 8, _, _]
        let x = self.gelu.forward(x);
        let x = self.conv_dropout.forward(x);

        let x = self.conv2.forward(x); // [batch_size, 16, _, _]
        let x = self.gelu.forward(x);
        let x = self.conv_dropout.forward(x);

        let x = self.pool.forward(x); // [batch_size, 16, 8, 8]
        let x = x.reshape([batch_size, 16 * 8 * 8]);
        let x = self.linear1.forward(x);
        let x = self.gelu.forward(x);
        let x = self.linear_dropout.forward(x);

        let x = self.linear2.forward(x);
        let x = self.tanh.forward(x);

        x
    }
}

#[derive(Config, Debug)]
pub struct HandwrittenEncoderConfig {
    latent_dim: usize,
    hidden_size: usize,
    #[config(default = "0.5")]
    linear_dropout: f64,
    #[config(default = "0.1")]
    conv_dropout: f64,
}

impl HandwrittenEncoderConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> HandwrittenEncoder<B> {
        HandwrittenEncoder {
            conv1: Conv2dConfig::new([1, 8], [5, 5]).init(device),
            conv2: Conv2dConfig::new([8, 16], [3, 3]).init(device),
            pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            gelu: Gelu::new(),
            tanh: Tanh::new(),
            linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.latent_dim).init(device),
            linear_dropout: DropoutConfig::new(self.linear_dropout).init(),
            conv_dropout: DropoutConfig::new(self.conv_dropout).init(),
        }
    }
}

#[derive(Module, Debug)]
pub struct HandwrittenDecoder<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    conv_transpose1: ConvTranspose2d<B>,
    conv_transpose2: ConvTranspose2d<B>,
    linear_dropout: Dropout,
    conv_dropout: Dropout,
    gelu: Gelu,
    sigmoid: Sigmoid,
}

impl<B: Backend> HandwrittenDecoder<B> {
    /// # Shapes
    ///   - Images [batch_size, latent_dim]
    ///   - Output [batch_size, height, width]
    pub fn forward(&self, latent_space: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch_size, _latent_dim] = latent_space.dims();

        // Reverse the encoder: latent -> hidden -> feature map
        let x = self.linear1.forward(latent_space);
        let x = self.gelu.forward(x);
        let x = self.linear_dropout.forward(x);

        let x = self.linear2.forward(x);
        let x = self.gelu.forward(x);
        let x = self.linear_dropout.forward(x);

        // Reshape back to feature map: [batch_size, 16, 8, 8]
        let x = x.reshape([batch_size, 16, 8, 8]);

        // Transpose convolutions to upsample
        let x = self.conv_transpose1.forward(x);
        let x = self.gelu.forward(x);
        let x = self.conv_dropout.forward(x);

        let x = self.conv_transpose2.forward(x);
        let x = self.sigmoid.forward(x); // Output should be in [0, 1] range

        // Remove the channel dimension to get [batch_size, height, width]
        let [_batch_size, _channels, height, width] = x.dims();
        x.reshape([batch_size, height, width])
    }
}

#[derive(Config, Debug)]
pub struct HandwrittenDecoderConfig {
    latent_dim: usize,
    hidden_size: usize,
    output_height: usize,
    output_width: usize,
    #[config(default = "0.5")]
    linear_dropout: f64,
    #[config(default = "0.1")]
    conv_dropout: f64,
}

impl HandwrittenDecoderConfig {
    /// Returns the initialized decoder model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> HandwrittenDecoder<B> {
        HandwrittenDecoder {
            linear1: LinearConfig::new(self.latent_dim, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, 16 * 8 * 8).init(device),
            // ConvTranspose2d to go from [16, 8, 8] back to larger feature maps
            // We need to determine the right parameters to get the desired output size
            conv_transpose1: ConvTranspose2dConfig::new([16, 8], [3, 3])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .init(device),
            conv_transpose2: ConvTranspose2dConfig::new([8, 1], [5, 5])
                .with_stride([2, 2])
                .with_padding([2, 2])
                .init(device),
            linear_dropout: DropoutConfig::new(self.linear_dropout).init(),
            conv_dropout: DropoutConfig::new(self.conv_dropout).init(),
            gelu: Gelu::new(),
            sigmoid: Sigmoid::new(),
        }
    }
}


#[derive(Module, Debug)]
pub struct HandwrittenAutoEncoder<B: Backend> {
    encoder: HandwrittenEncoder<B>,
    decoder: HandwrittenDecoder<B>
}

impl<B: Backend> HandwrittenAutoEncoder<B> {
    /// Forward pass through the autoencoder
    /// # Shapes
    ///   - Input [batch_size, height, width]
    ///   - Output [batch_size, height, width]
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 3> {
        let latent = self.encoder.forward(images);
        self.decoder.forward(latent)
    }

    /// Encode images to latent space
    /// # Shapes
    ///   - Input [batch_size, height, width]
    ///   - Output [batch_size, latent_dim]
    pub fn encode(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        self.encoder.forward(images)
    }

    /// Decode latent representations to images
    /// # Shapes
    ///   - Input [batch_size, latent_dim]
    ///   - Output [batch_size, height, width]
    pub fn decode(&self, latent: Tensor<B, 2>) -> Tensor<B, 3> {
        self.decoder.forward(latent)
    }

    pub fn forward_regression(&self, images: Tensor<B, 3>, expected: Tensor<B, 3>) -> RegressionOutput<B> {
        let [batch_size, ..] = images.dims();
        let actual = self.forward(images);
        let loss = burn::nn::loss::MseLoss::new()
            .forward(actual.clone(), expected.clone(), nn::loss::Reduction::Auto);
        RegressionOutput::new(loss, actual.reshape([batch_size as i32, -1]), expected.reshape([batch_size as i32, -1]))
    }
}

impl<B: AutodiffBackend> TrainStep<MnistBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: MnistBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MnistBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: MnistBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}

#[derive(Config, Debug)]
pub struct HandwrittenAutoEncoderConfig {
    latent_dim: usize,
    hidden_size: usize,
    output_height: usize,
    output_width: usize,
    #[config(default = "0.5")]
    linear_dropout: f64,
    #[config(default = "0.1")]
    conv_dropout: f64,
}

impl HandwrittenAutoEncoderConfig {
    /// Returns the initialized autoencoder model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> HandwrittenAutoEncoder<B> {
        let encoder_config = HandwrittenEncoderConfig {
            latent_dim: self.latent_dim,
            hidden_size: self.hidden_size,
            linear_dropout: self.linear_dropout,
            conv_dropout: self.conv_dropout,
        };

        let decoder_config = HandwrittenDecoderConfig {
            latent_dim: self.latent_dim,
            hidden_size: self.hidden_size,
            output_height: self.output_height,
            output_width: self.output_width,
            linear_dropout: self.linear_dropout,
            conv_dropout: self.conv_dropout,
        };

        HandwrittenAutoEncoder {
            encoder: encoder_config.init(device),
            decoder: decoder_config.init(device),
        }
    }
}