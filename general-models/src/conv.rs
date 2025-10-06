use burn::{
    module::{Ignored, Module},
    nn::{
        Dropout, DropoutConfig, PaddingConfig2d,
        activation::Activation,
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
    },
    prelude::*,
};
use serde::{Deserialize, Serialize};

use crate::{
    Init, SimpleInfer, SimpleTrain,
    common::{ActivationConfig, Either, Norm, NormConfig},
    default_f,
};

#[derive(Debug, Module)]
pub struct Conv2dModel<B: Backend> {
    input_channels: Ignored<usize>,
    layers: Vec<(Conv2d<B>, Option<Norm<B>>, Option<Activation<B>>)>,
    dropout: Dropout,
}

impl<B: Backend> Conv2dModel<B> {
    pub fn get_input_channels(&self) -> usize {
        self.input_channels.0
    }
}

impl<B: Backend> SimpleTrain<B, 4, 4> for Conv2dModel<B> {
    fn forward(&self, mut tensor: burn::Tensor<B, 4>) -> burn::Tensor<B, 4> {
        for (i, (conv, norm, activation)) in self.layers.iter().enumerate() {
            tensor = conv.forward(tensor);
            if let Some(norm) = norm {
                tensor = norm.forward(tensor);
            }
            if let Some(activation) = activation {
                tensor = activation.forward(tensor);
            }
            if i < self.layers.len() - 1 {
                tensor = self.dropout.forward(tensor);
            }
        }
        tensor
    }
}

impl<B: Backend> SimpleInfer<B, 4, 4> for Conv2dModel<B> {
    fn forward(&self, mut tensor: burn::Tensor<B, 4>) -> burn::Tensor<B, 4> {
        for (conv, norm, activation) in self.layers.iter() {
            tensor = conv.forward(tensor);
            if let Some(norm) = norm {
                tensor = norm.forward(tensor);
            }
            if let Some(activation) = activation {
                tensor = activation.forward(tensor);
            }
        }
        tensor
    }
}

#[derive(Serialize, Debug, Deserialize, Clone, Copy)]
pub struct Conv2dLayerConfig {
    pub output_channels: usize,
    pub kernel_size: [usize; 2],
    #[serde(default = "default_stride")]
    pub stride: [usize; 2],
    #[serde(default = "default_stride")]
    pub dilation: [usize; 2],
    #[serde(default = "default_groups")]
    pub groups: usize,
    pub padding: Option<[usize; 2]>,
}

#[derive(Serialize, Debug, Deserialize, Clone)]
pub struct Conv2dModelConfig {
    pub input_channels: usize,
    pub default_activation: Option<ActivationConfig>,
    pub default_norm: Option<NormConfig>,
    pub layers: Vec<Either<Conv2dLayerConfig, ActivationConfig, NormConfig>>,
    #[serde(default = "default_dropout")]
    pub dropout: f64,
}

impl<B: Backend> Init<B, Conv2dModel<B>> for Conv2dModelConfig {
    fn init(self, device: &B::Device) -> Conv2dModel<B> {
        let default_activation = self.default_activation.unwrap_or_default();
        let mut input_channels = self.input_channels;
        let mut layers = vec![];
        for (
            Conv2dLayerConfig {
                output_channels,
                kernel_size,
                stride,
                dilation,
                groups,
                padding,
            },
            activation,
            norm,
        ) in self.layers.into_iter().map(Either::into_tuple)
        {
            layers.push((
                Conv2dConfig::new([input_channels, output_channels], kernel_size)
                    .with_bias(norm.is_none())
                    .with_stride(stride)
                    .with_dilation(dilation)
                    .with_groups(groups)
                    .with_padding(
                        padding
                            .map(|[x, y]| PaddingConfig2d::Explicit(x, y))
                            .unwrap_or(PaddingConfig2d::Valid),
                    )
                    .init(device),
                norm.or_else(|| self.default_norm.clone())
                    .map(|norm| norm.init(device, output_channels)),
                activation
                    .unwrap_or_else(|| default_activation.clone())
                    .init(device),
            ));
            input_channels = output_channels;
        }
        Conv2dModel {
            input_channels: Ignored(self.input_channels),
            layers,
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

#[derive(Debug, Module)]
pub struct ConvTranspose2dModel<B: Backend> {
    input_channels: Ignored<usize>,
    layers: Vec<(ConvTranspose2d<B>, Option<Norm<B>>, Option<Activation<B>>)>,
    dropout: Dropout,
}

impl<B: Backend> ConvTranspose2dModel<B> {
    pub fn get_input_channels(&self) -> usize {
        self.input_channels.0
    }
}

impl<B: Backend> SimpleTrain<B, 4, 4> for ConvTranspose2dModel<B> {
    fn forward(&self, mut tensor: burn::Tensor<B, 4>) -> burn::Tensor<B, 4> {
        for (i, (conv, norm, activation)) in self.layers.iter().enumerate() {
            tensor = conv.forward(tensor);
            if let Some(norm) = norm {
                tensor = norm.forward(tensor);
            }
            if let Some(activation) = activation {
                tensor = activation.forward(tensor);
            }
            if i < self.layers.len() - 1 {
                tensor = self.dropout.forward(tensor);
            }
        }
        tensor
    }
}

impl<B: Backend> SimpleInfer<B, 4, 4> for ConvTranspose2dModel<B> {
    fn forward(&self, mut tensor: burn::Tensor<B, 4>) -> burn::Tensor<B, 4> {
        for (conv, norm, activation) in self.layers.iter() {
            tensor = conv.forward(tensor);
            if let Some(norm) = norm {
                tensor = norm.forward(tensor);
            }
            if let Some(activation) = activation {
                tensor = activation.forward(tensor);
            }
        }
        tensor
    }
}

#[derive(Serialize, Debug, Deserialize, Clone, Copy)]
pub struct ConvTranspose2dLayerConfig {
    pub output_channels: usize,
    pub kernel_size: [usize; 2],
    #[serde(default = "default_stride")]
    pub stride: [usize; 2],
    #[serde(default = "default_stride")]
    pub dilation: [usize; 2],
    #[serde(default = "default_groups")]
    pub groups: usize,
    #[serde(default = "default_padding")]
    pub padding: [usize; 2],
    #[serde(default = "default_padding")]
    pub padding_out: [usize; 2],
}

#[derive(Serialize, Debug, Deserialize, Clone)]
pub struct ConvTranspose2dModelConfig {
    pub input_channels: usize,
    pub default_activation: Option<ActivationConfig>,
    pub default_norm: Option<NormConfig>,
    pub layers: Vec<Either<ConvTranspose2dLayerConfig, ActivationConfig, NormConfig>>,
    #[serde(default = "default_dropout")]
    pub dropout: f64,
}

impl<B: Backend> Init<B, ConvTranspose2dModel<B>> for ConvTranspose2dModelConfig {
    fn init(self, device: &B::Device) -> ConvTranspose2dModel<B> {
        let default_activation = self.default_activation.unwrap_or_default();
        let mut input_channels = self.input_channels;
        let mut layers = vec![];
        for (
            ConvTranspose2dLayerConfig {
                output_channels,
                kernel_size,
                stride,
                dilation,
                padding,
                padding_out,
                groups,
            },
            activation,
            norm,
        ) in self.layers.into_iter().map(Either::into_tuple)
        {
            layers.push((
                ConvTranspose2dConfig::new([input_channels, output_channels], kernel_size)
                    .with_bias(norm.is_none())
                    .with_stride(stride)
                    .with_dilation(dilation)
                    .with_padding(padding)
                    .with_padding_out(padding_out)
                    .with_groups(groups)
                    .init(device),
                norm.or_else(|| self.default_norm.clone())
                    .map(|norm| norm.init(device, output_channels)),
                activation
                    .unwrap_or_else(|| default_activation.clone())
                    .init(device),
            ));
            input_channels = output_channels;
        }
        ConvTranspose2dModel {
            input_channels: Ignored(self.input_channels),
            layers,
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

default_f!(default_stride, [usize; 2], [1, 1]);
default_f!(default_groups, usize, 1);
default_f!(default_dropout, f64, 0.0);
default_f!(default_padding, [usize; 2], [0, 0]);
