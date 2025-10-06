use burn::{
    module::Ignored,
    nn::{
        interpolate::{Interpolate2d, Interpolate2dConfig, InterpolateMode},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
    },
    prelude::*,
};
use serde::{Deserialize, Serialize};

use crate::{
    Init, SimpleInfer, SimpleTrain,
    conv::{Conv2dModel, Conv2dModelConfig, ConvTranspose2dModel, ConvTranspose2dModelConfig},
    linear::{LinearClassifierModel, LinearClassifierModelConfig, LinearModel, LinearModelConfig},
};

#[derive(Debug, Module)]
pub struct ConvLinearModel<B: Backend> {
    conv: Conv2dModel<B>,
    adaptive_avg_pooling: Option<AdaptiveAvgPool2d>,
    linear: LinearModel<B>,
}

impl<B: Backend> SimpleInfer<B, 4, 2> for ConvLinearModel<B> {
    fn forward(&self, mut tensor: Tensor<B, 4>) -> Tensor<B, 2> {
        tensor = self.conv.infer(tensor);
        if let Some(adaptive_avg_pooling) = &self.adaptive_avg_pooling {
            tensor = adaptive_avg_pooling.forward(tensor);
        }
        self.linear.infer(tensor.flatten(1, 3))
    }
}

impl<B: Backend> SimpleTrain<B, 4, 2> for ConvLinearModel<B> {
    fn forward(&self, mut tensor: Tensor<B, 4>) -> Tensor<B, 2> {
        tensor = self.conv.train(tensor);
        if let Some(adaptive_avg_pooling) = &self.adaptive_avg_pooling {
            tensor = adaptive_avg_pooling.forward(tensor);
        }
        self.linear.train(tensor.flatten(1, 3))
    }
}

impl<B: Backend> ConvLinearModel<B> {
    pub fn get_input_channels(&self) -> usize {
        self.conv.get_input_channels()
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ConvLinearModelConfig {
    pub conv: Conv2dModelConfig,
    pub adaptive_avg_pooling: Option<[usize; 2]>,
    pub linear: LinearModelConfig,
}

impl<B: Backend> Init<B, ConvLinearModel<B>> for ConvLinearModelConfig {
    fn init(self, device: &<B as Backend>::Device) -> ConvLinearModel<B> {
        ConvLinearModel {
            conv: self.conv.init(device),
            adaptive_avg_pooling: self
                .adaptive_avg_pooling
                .map(|x| AdaptiveAvgPool2dConfig::new(x).init()),
            linear: self.linear.init(device),
        }
    }
}

#[derive(Debug, Module)]
pub struct ConvLinearClassifierModel<B: Backend> {
    conv_linear: ConvLinearModel<B>,
    classifier: LinearClassifierModel<B>,
}

impl<B: Backend> SimpleInfer<B, 4, 2> for ConvLinearClassifierModel<B> {
    fn forward(&self, tensor: Tensor<B, 4>) -> Tensor<B, 2> {
        self.classifier.infer(self.conv_linear.infer(tensor))
    }
}

impl<B: Backend> SimpleTrain<B, 4, 2> for ConvLinearClassifierModel<B> {
    fn forward(&self, tensor: Tensor<B, 4>) -> Tensor<B, 2> {
        self.classifier.train(self.conv_linear.train(tensor))
    }
}

impl<B: Backend> ConvLinearClassifierModel<B> {
    pub fn get_input_channels(&self) -> usize {
        self.conv_linear.get_input_channels()
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ConvLinearClassifierModelConfig {
    pub conv_linear: ConvLinearModelConfig,
    pub classifier: LinearClassifierModelConfig,
}

impl<B: Backend> Init<B, ConvLinearClassifierModel<B>> for ConvLinearClassifierModelConfig {
    fn init(self, device: &<B as Backend>::Device) -> ConvLinearClassifierModel<B> {
        ConvLinearClassifierModel {
            conv_linear: self.conv_linear.init(device),
            classifier: self.classifier.init(device),
        }
    }
}

#[derive(Debug, Module)]
pub struct LinearConvTransposedModel<B: Backend> {
    linear: LinearModel<B>,
    conv_input_size: Ignored<[usize; 2]>,
    intermediate_interpolate: Option<Interpolate2d>,
    conv: ConvTranspose2dModel<B>,
    output_interpolate: Option<Interpolate2d>,
}

impl<B: Backend> SimpleInfer<B, 2, 4> for LinearConvTransposedModel<B> {
    fn forward(&self, tensor: Tensor<B, 2>) -> Tensor<B, 4> {
        let batch_size = tensor.dims()[0];
        let tensor = self.conv.infer(self.linear.infer(tensor).reshape([
            batch_size,
            self.conv.get_input_channels(),
            self.conv_input_size[0],
            self.conv_input_size[1],
        ]));
        if let Some(interpolate) = &self.output_interpolate {
            interpolate.forward(tensor)
        } else {
            tensor
        }
    }
}

impl<B: Backend> SimpleTrain<B, 2, 4> for LinearConvTransposedModel<B> {
    fn forward(&self, mut tensor: Tensor<B, 2>) -> Tensor<B, 4> {
        let batch_size = tensor.dims()[0];
        tensor = self.linear.train(tensor);
        let mut tensor = tensor.reshape([
            batch_size,
            self.conv.get_input_channels(),
            self.conv_input_size[0],
            self.conv_input_size[1],
        ]);
        if let Some(interpolate) = &self.intermediate_interpolate {
            tensor = interpolate.forward(tensor);
        }
        let tensor = self.conv.train(tensor);
        if let Some(interpolate) = &self.output_interpolate {
            interpolate.forward(tensor)
        } else {
            tensor
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LinearConvTransposedModelConfig {
    pub linear: LinearModelConfig,
    pub intermediate_interpolate: Option<InterpolateMode>,
    pub conv_input_size: [usize; 2],
    pub conv: ConvTranspose2dModelConfig,
    pub output_interpolate: Option<Interpolate2dConfig>,
}

impl<B: Backend> Init<B, LinearConvTransposedModel<B>> for LinearConvTransposedModelConfig {
    fn init(self, device: &<B as Backend>::Device) -> LinearConvTransposedModel<B> {
        LinearConvTransposedModel {
            linear: self.linear.init(device),
            conv_input_size: Ignored(self.conv_input_size),
            intermediate_interpolate: self.intermediate_interpolate.map(|mode| {
                Interpolate2dConfig::new()
                    .with_mode(mode)
                    .with_output_size(Some([
                        self.conv_input_size[0] as usize,
                        self.conv_input_size[1] as usize,
                    ]))
                    .init()
            }),
            conv: self.conv.init(device),
            output_interpolate: self.output_interpolate.map(|x| x.init()),
        }
    }
}
