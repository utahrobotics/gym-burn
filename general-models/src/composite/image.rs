use burn::{
    nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
    prelude::*,
};
use serde::{Deserialize, Serialize};

use crate::{
    Init, SimpleInfer, SimpleTrain,
    conv::{Conv2dModel, Conv2dModelConfig},
    linear::{LinearClassifierModel, LinearClassifierModelConfig, LinearModel, LinearModelConfig},
};

#[derive(Debug, Module)]
pub struct ConvLinearImageModel<B: Backend> {
    conv: Conv2dModel<B>,
    adaptive_avg_pooling: Option<AdaptiveAvgPool2d>,
    linear: LinearModel<B>,
}

impl<B: Backend> SimpleInfer<B, 4, 2> for ConvLinearImageModel<B> {
    fn forward(&self, mut tensor: Tensor<B, 4>) -> Tensor<B, 2> {
        let batch_size = tensor.dims()[0] as i32;
        tensor = self.conv.infer(tensor);
        if let Some(adaptive_avg_pooling) = &self.adaptive_avg_pooling {
            tensor = adaptive_avg_pooling.forward(tensor);
        }
        self.linear.infer(tensor.reshape([batch_size, -1]))
    }
}

impl<B: Backend> SimpleTrain<B, 4, 2> for ConvLinearImageModel<B> {
    fn forward(&self, mut tensor: Tensor<B, 4>) -> Tensor<B, 2> {
        let batch_size = tensor.dims()[0] as i32;
        tensor = self.conv.train(tensor);
        if let Some(adaptive_avg_pooling) = &self.adaptive_avg_pooling {
            tensor = adaptive_avg_pooling.forward(tensor);
        }
        self.linear.train(tensor.reshape([batch_size, -1]))
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ConvLinearImageModelConfig {
    pub conv: Conv2dModelConfig,
    pub adaptive_avg_pooling: Option<AdaptiveAvgPool2dConfig>,
    pub linear: LinearModelConfig,
}

impl<B: Backend> Init<B> for ConvLinearImageModelConfig {
    type Output = ConvLinearImageModel<B>;

    fn init(self, device: &<B as Backend>::Device) -> Self::Output {
        ConvLinearImageModel {
            conv: self.conv.init(device),
            adaptive_avg_pooling: self.adaptive_avg_pooling.map(|x| x.init()),
            linear: self.linear.init(device),
        }
    }
}

#[derive(Debug, Module)]
pub struct ConvLinearImageClassifierModel<B: Backend> {
    conv_linear: ConvLinearImageModel<B>,
    classifier: LinearClassifierModel<B>,
}

impl<B: Backend> SimpleInfer<B, 4, 2> for ConvLinearImageClassifierModel<B> {
    fn forward(&self, tensor: Tensor<B, 4>) -> Tensor<B, 2> {
        self.classifier.infer(self.conv_linear.infer(tensor))
    }
}

impl<B: Backend> SimpleTrain<B, 4, 2> for ConvLinearImageClassifierModel<B> {
    fn forward(&self, tensor: Tensor<B, 4>) -> Tensor<B, 2> {
        self.classifier.train(self.conv_linear.train(tensor))
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ConvLinearImageClassifierModelConfig {
    pub conv_linear: ConvLinearImageModelConfig,
    pub classifier: LinearClassifierModelConfig,
}

impl<B: Backend> Init<B> for ConvLinearImageClassifierModelConfig {
    type Output = ConvLinearImageClassifierModel<B>;

    fn init(self, device: &<B as Backend>::Device) -> Self::Output {
        ConvLinearImageClassifierModel {
            conv_linear: self.conv_linear.init(device),
            classifier: self.classifier.init(device),
        }
    }
}
