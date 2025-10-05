use burn::{nn::pool::AdaptiveAvgPool2d, prelude::*};

use crate::{conv::Conv2dModel, linear::{LinearClassifierModel, LinearModel}, SimpleInfer, SimpleTrain};

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

#[derive(Debug, Module)]
pub struct ConvLinearImageClassifierModel<B: Backend> {
    conv_linear: ConvLinearImageModel<B>,
    classifier: LinearClassifierModel<B>
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