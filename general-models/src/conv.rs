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
use utils::default_f;

use crate::{
    Init, SimpleInfer, SimpleTrain,
    common::{ActivationConfig, Either, Norm, NormConfig, Optional, handle_norm_activation},
};

#[derive(Debug, Module)]
pub struct Conv2dModel<B: Backend> {
    input_channels: Ignored<usize>,
    layers: Vec<(Conv2d<B>, Option<Norm<B>>, Option<Activation<B>>)>,
    dropout: Dropout,
    dropout_last: bool,
}

impl<B: Backend> Conv2dModel<B> {
    pub fn get_input_channels(&self) -> usize {
        self.input_channels.0
    }

    pub fn iter_layers(
        &mut self,
        mut map: impl FnMut(
            Conv2d<B>,
            Option<Norm<B>>,
            Option<Activation<B>>,
        ) -> (Conv2d<B>, Option<Norm<B>>, Option<Activation<B>>),
    ) {
        self.layers = std::mem::take(&mut self.layers)
            .into_iter()
            .map(|(conv, norm, activation)| map(conv, norm, activation))
            .collect();
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
            if i < self.layers.len() - 1 || self.dropout_last {
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
    pub padding: Option<[usize; 2]>
}

#[derive(Serialize, Debug, Deserialize, Clone)]
pub struct Conv2dModelConfig {
    pub input_channels: usize,
    pub default_activation: Option<ActivationConfig>,
    pub default_norm: Option<NormConfig>,
    pub layers: Vec<Either<Conv2dLayerConfig, Optional<ActivationConfig>, Optional<NormConfig>, Option<f64>>>,
    #[serde(default = "default_dropout")]
    pub dropout: f64,
    #[serde(default = "default_dropout_last")]
    pub dropout_last: bool,
}

impl<B: Backend> Init<B, Conv2dModel<B>> for Conv2dModelConfig {
    fn init(self, device: &B::Device) -> Conv2dModel<B> {
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
            weights_gain
        ) in self.layers.into_iter().map(Either::into_tuple)
        {
            let (norm, activation, init) = handle_norm_activation(
                norm,
                activation,
                &self.default_norm,
                &self.default_activation,
                weights_gain,
                output_channels,
                device
            );
            layers.push((
                Conv2dConfig::new([input_channels, output_channels], kernel_size)
                    .with_bias(norm.is_none())
                    .with_stride(stride)
                    .with_initializer(init)
                    .with_dilation(dilation)
                    .with_groups(groups)
                    .with_padding(
                        padding
                            .map(|[x, y]| PaddingConfig2d::Explicit(x, y))
                            .unwrap_or(PaddingConfig2d::Valid),
                    )
                    .init(device),
                norm,
                activation
            ));
            input_channels = output_channels;
        }
        Conv2dModel {
            input_channels: Ignored(self.input_channels),
            layers,
            dropout: DropoutConfig::new(self.dropout).init(),
            dropout_last: self.dropout_last,
        }
    }
}

#[derive(Debug, Module)]
pub struct ConvTranspose2dModel<B: Backend> {
    input_channels: Ignored<usize>,
    layers: Vec<(ConvTranspose2d<B>, Option<Norm<B>>, Option<Activation<B>>)>,
    dropout: Dropout,
    dropout_last: bool,
}

impl<B: Backend> ConvTranspose2dModel<B> {
    pub fn get_input_channels(&self) -> usize {
        self.input_channels.0
    }

    pub fn iter_layers(
        &mut self,
        mut map: impl FnMut(
            ConvTranspose2d<B>,
            Option<Norm<B>>,
            Option<Activation<B>>,
        ) -> (ConvTranspose2d<B>, Option<Norm<B>>, Option<Activation<B>>),
    ) {
        self.layers = std::mem::take(&mut self.layers)
            .into_iter()
            .map(|(conv, norm, activation)| map(conv, norm, activation))
            .collect();
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
            if i < self.layers.len() - 1 || self.dropout_last {
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
    pub layers: Vec<Either<ConvTranspose2dLayerConfig, Optional<ActivationConfig>, Optional<NormConfig>, Option<f64>>>,
    #[serde(default = "default_dropout")]
    pub dropout: f64,
    #[serde(default = "default_dropout_last")]
    pub dropout_last: bool,
}

impl<B: Backend> Init<B, ConvTranspose2dModel<B>> for ConvTranspose2dModelConfig {
    fn init(self, device: &B::Device) -> ConvTranspose2dModel<B> {
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
            weights_gain
        ) in self.layers.into_iter().map(Either::into_tuple)
        {
            let (norm, activation, init) = handle_norm_activation(
                norm,
                activation,
                &self.default_norm,
                &self.default_activation,
                weights_gain,
                output_channels,
                device
            );
            layers.push((
                ConvTranspose2dConfig::new([input_channels, output_channels], kernel_size)
                    .with_bias(norm.is_none())
                    .with_stride(stride)
                    .with_initializer(init)
                    .with_dilation(dilation)
                    .with_padding(padding)
                    .with_padding_out(padding_out)
                    .with_groups(groups)
                    .init(device),
                norm,
                activation
            ));
            input_channels = output_channels;
        }

        ConvTranspose2dModel {
            input_channels: Ignored(self.input_channels),
            layers,
            dropout: DropoutConfig::new(self.dropout).init(),
            dropout_last: self.dropout_last,
        }
    }
}

default_f!(default_stride, [usize; 2], [1, 1]);
default_f!(default_groups, usize, 1);
default_f!(default_dropout, f64, 0.0);
default_f!(default_dropout_last, bool, true);
default_f!(default_padding, [usize; 2], [0, 0]);
