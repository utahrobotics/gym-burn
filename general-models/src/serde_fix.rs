use burn::{
    config::Config,
    nn::{
        Initializer, Linear, LinearLayout, PaddingConfig2d,
        conv::{Conv2d, ConvTranspose2d},
    },
    prelude::Backend,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LinearConfig {
    /// The size of the input features.
    pub d_input: usize,
    /// The size of the output features.
    pub d_output: usize,
    /// If a bias should be applied during the linear transformation.
    #[serde(default = "fn_true")]
    pub bias: bool,
    pub initializer: Option<Initializer>,
    /// The layout in which the linear parameters are stored.
    pub layout: Option<LinearLayout>,
}

impl Config for LinearConfig {}

impl LinearConfig {
    pub(crate) fn into_burn(&self) -> burn::nn::LinearConfig {
        let mut config = burn::nn::LinearConfig::new(self.d_input, self.d_output);
        config.bias = self.bias;
        if let Some(initializer) = self.initializer.clone() {
            config.initializer = initializer;
        }
        if let Some(layout) = self.layout {
            config.layout = layout;
        }
        config
    }

    pub(crate) fn init<B: Backend>(&self, device: &B::Device) -> Linear<B> {
        self.into_burn().init(device)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Conv2dConfig {
    /// The number of channels.
    pub channels: [usize; 2],
    /// The size of the kernel.
    pub kernel_size: [usize; 2],
    /// The stride of the convolution.
    pub stride: Option<[usize; 2]>,
    /// Spacing between kernel elements.
    pub dilation: Option<[usize; 2]>,
    /// Controls the connections between input and output channels.
    #[serde(default = "fn_1")]
    pub groups: usize,
    /// The padding configuration.
    ///
    /// ### Warning
    /// Only symmetric padding is currently supported. As such, using `Same` padding with an even kernel
    /// size is not supported as it will not produce the same output size.
    pub padding: Option<PaddingConfig2d>,
    /// If bias should be added to the output.
    #[serde(default = "fn_true")]
    pub bias: bool,
    /// The type of function used to initialize neural network parameters
    pub initializer: Option<Initializer>,
}

impl Config for Conv2dConfig {}

impl Conv2dConfig {
    pub(crate) fn into_burn(&self) -> burn::nn::conv::Conv2dConfig {
        let mut config = burn::nn::conv::Conv2dConfig::new(self.channels, self.kernel_size);
        if let Some(stride) = self.stride {
            config.stride = stride;
        }
        if let Some(dilation) = self.dilation {
            config.dilation = dilation;
        }
        config.groups = self.groups;
        if let Some(padding) = self.padding.clone() {
            config.padding = padding;
        }
        config.bias = self.bias;
        if let Some(initializer) = self.initializer.clone() {
            config.initializer = initializer;
        }
        config
    }

    pub(crate) fn init<B: Backend>(&self, device: &B::Device) -> Conv2d<B> {
        self.into_burn().init(device)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ConvTranspose2dConfig {
    /// The number of channels.
    pub channels: [usize; 2],
    /// The size of the kernel.
    pub kernel_size: [usize; 2],
    /// The stride of the convolution.
    pub stride: Option<[usize; 2]>,
    /// Spacing between kernel elements.
    pub dilation: Option<[usize; 2]>,
    /// Controls the connections between input and output channels.
    #[serde(default = "fn_1")]
    pub groups: usize,
    /// The padding configuration.
    pub padding: Option<[usize; 2]>,
    /// The padding output configuration.
    pub padding_out: Option<[usize; 2]>,
    /// If bias should be added to the output.
    #[serde(default = "fn_true")]
    pub bias: bool,
    /// The type of function used to initialize neural network parameters
    pub initializer: Option<Initializer>,
}

impl Config for ConvTranspose2dConfig {}

impl ConvTranspose2dConfig {
    pub(crate) fn into_burn(&self) -> burn::nn::conv::ConvTranspose2dConfig {
        let mut config =
            burn::nn::conv::ConvTranspose2dConfig::new(self.channels, self.kernel_size);
        if let Some(stride) = self.stride {
            config.stride = stride;
        }
        if let Some(dilation) = self.dilation {
            config.dilation = dilation;
        }
        config.groups = self.groups;
        if let Some(padding) = self.padding.clone() {
            config.padding = padding;
        }
        if let Some(padding_out) = self.padding_out.clone() {
            config.padding_out = padding_out;
        }
        config.bias = self.bias;
        if let Some(initializer) = self.initializer.clone() {
            config.initializer = initializer;
        }
        config
    }

    pub(crate) fn init<B: Backend>(&self, device: &B::Device) -> ConvTranspose2d<B> {
        self.into_burn().init(device)
    }
}

fn fn_true() -> bool {
    true
}

fn fn_1() -> usize {
    1
}
