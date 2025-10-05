use burn::{
    module::Module,
    nn::{
        Dropout, DropoutConfig,
        activation::{Activation, ActivationConfig},
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
    },
    prelude::*,
};
use serde::{Deserialize, Serialize};

use crate::{
    Init, SimpleInfer, SimpleTrain,
    common::{Norm, NormConfig},
    default_f,
};

#[derive(Debug, Module)]
pub struct Conv2dModel<B: Backend> {
    layers: Vec<(Conv2d<B>, Option<Norm<B>>, Activation<B>)>,
    dropout: Dropout,
}

impl<B: Backend> SimpleInfer<B, 4, 4> for Conv2dModel<B> {
    fn forward(&self, mut tensor: burn::Tensor<B, 4>) -> burn::Tensor<B, 4> {
        for (i, (conv, norm, activation)) in self.layers.iter().enumerate() {
            tensor = conv.forward(tensor);
            if let Some(norm) = norm {
                tensor = norm.forward(tensor);
            }
            tensor = activation.forward(tensor);
            if i < self.layers.len() - 1 {
                tensor = self.dropout.forward(tensor);
            }
        }
        tensor
    }
}

impl<B: Backend> SimpleTrain<B, 4, 4> for Conv2dModel<B> {
    fn forward(&self, mut tensor: burn::Tensor<B, 4>) -> burn::Tensor<B, 4> {
        for (conv, norm, activation) in self.layers.iter() {
            tensor = conv.forward(tensor);
            if let Some(norm) = norm {
                tensor = norm.forward(tensor);
            }
            tensor = activation.forward(tensor);
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
}

#[derive(Serialize, Debug, Deserialize, Clone)]
pub struct Conv2dModelConfig {
    pub input_channels: usize,
    pub default_activation: Option<ActivationConfig>,
    pub default_norm: Option<NormConfig>,
    pub layers: Vec<(
        Conv2dLayerConfig,
        Option<NormConfig>,
        Option<ActivationConfig>,
    )>,
    pub dropout: f64,
}

impl<B: Backend> Init<B> for Conv2dModelConfig {
    type Output = Conv2dModel<B>;

    fn init(self, device: &B::Device) -> Self::Output {
        let default_activation = self.default_activation.unwrap_or(ActivationConfig::Gelu);
        let mut input_channels = self.input_channels;
        let mut layers = vec![];
        for (
            Conv2dLayerConfig {
                output_channels,
                kernel_size,
                stride,
                dilation,
                groups,
            },
            norm,
            activation,
        ) in self.layers
        {
            layers.push((
                Conv2dConfig::new([input_channels, output_channels], kernel_size)
                    .with_bias(norm.is_none())
                    .with_stride(stride)
                    .with_dilation(dilation)
                    .with_groups(groups)
                    .init(device),
                norm.map(|norm| norm.init(device, output_channels)),
                activation
                    .unwrap_or_else(|| default_activation.clone())
                    .init(device),
            ));
            input_channels = output_channels;
        }
        Conv2dModel {
            layers,
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

#[derive(Debug, Module)]
pub struct ConvTranspose2dModel<B: Backend> {
    layers: Vec<(ConvTranspose2d<B>, Option<Norm<B>>, Activation<B>)>,
    dropout: Dropout,
}

impl<B: Backend> SimpleInfer<B, 4, 4> for ConvTranspose2dModel<B> {
    fn forward(&self, mut tensor: burn::Tensor<B, 4>) -> burn::Tensor<B, 4> {
        for (i, (conv, norm, activation)) in self.layers.iter().enumerate() {
            tensor = conv.forward(tensor);
            if let Some(norm) = norm {
                tensor = norm.forward(tensor);
            }
            tensor = activation.forward(tensor);
            if i < self.layers.len() - 1 {
                tensor = self.dropout.forward(tensor);
            }
        }
        tensor
    }
}

impl<B: Backend> SimpleTrain<B, 4, 4> for ConvTranspose2dModel<B> {
    fn forward(&self, mut tensor: burn::Tensor<B, 4>) -> burn::Tensor<B, 4> {
        for (conv, norm, activation) in self.layers.iter() {
            tensor = conv.forward(tensor);
            if let Some(norm) = norm {
                tensor = norm.forward(tensor);
            }
            tensor = activation.forward(tensor);
        }
        tensor
    }
}

#[derive(Serialize, Debug, Deserialize, Clone)]
pub struct ConvTranspose2dModelConfig(Conv2dModelConfig);

impl<B: Backend> Init<B> for ConvTranspose2dModelConfig {
    type Output = ConvTranspose2dModel<B>;

    fn init(self, device: &B::Device) -> Self::Output {
        let default_activation = self.0.default_activation.unwrap_or(ActivationConfig::Gelu);
        let mut input_channels = self.0.input_channels;
        let mut layers = vec![];
        for (
            Conv2dLayerConfig {
                output_channels,
                kernel_size,
                stride,
                dilation,
                groups,
            },
            norm,
            activation,
        ) in self.0.layers
        {
            layers.push((
                ConvTranspose2dConfig::new([input_channels, output_channels], kernel_size)
                    .with_bias(norm.is_none())
                    .with_stride(stride)
                    .with_dilation(dilation)
                    .with_groups(groups)
                    .init(device),
                norm.map(|norm| norm.init(device, output_channels)),
                activation
                    .unwrap_or_else(|| default_activation.clone())
                    .init(device),
            ));
            input_channels = output_channels;
        }
        ConvTranspose2dModel {
            layers,
            dropout: DropoutConfig::new(self.0.dropout).init(),
        }
    }
}

default_f!(default_stride, [usize; 2], [1, 1]);
default_f!(default_groups, usize, 1);
