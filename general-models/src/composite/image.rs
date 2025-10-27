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
pub struct Conv2dLinearModel<B: Backend> {
    pub conv: Conv2dModel<B>,
    adaptive_avg_pooling: Option<AdaptiveAvgPool2d>,
    pub linear: LinearModel<B>,
}

impl<B: Backend> SimpleInfer<B, 4, 2> for Conv2dLinearModel<B> {
    fn forward(&self, mut tensor: Tensor<B, 4>) -> Tensor<B, 2> {
        tensor = self.conv.infer(tensor);
        if let Some(adaptive_avg_pooling) = &self.adaptive_avg_pooling {
            tensor = adaptive_avg_pooling.forward(tensor);
        }
        self.linear.infer(tensor.flatten(1, 3))
    }
}

impl<B: Backend> SimpleTrain<B, 4, 2> for Conv2dLinearModel<B> {
    fn forward(&self, mut tensor: Tensor<B, 4>) -> Tensor<B, 2> {
        tensor = self.conv.train(tensor);
        if let Some(adaptive_avg_pooling) = &self.adaptive_avg_pooling {
            tensor = adaptive_avg_pooling.forward(tensor);
        }
        self.linear.train(tensor.flatten(1, 3))
    }
}

impl<B: Backend> Conv2dLinearModel<B> {
    pub fn get_input_channels(&self) -> usize {
        self.conv.get_input_channels()
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Conv2dLinearModelConfig {
    pub conv: Conv2dModelConfig,
    pub adaptive_avg_pooling: Option<[usize; 2]>,
    pub linear: LinearModelConfig,
}

impl<B: Backend> Init<B, Conv2dLinearModel<B>> for Conv2dLinearModelConfig {
    fn init(self, device: &<B as Backend>::Device) -> Conv2dLinearModel<B> {
        Conv2dLinearModel {
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
    pub conv_linear: Conv2dLinearModel<B>,
    pub classifier: LinearClassifierModel<B>,
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
    pub conv_linear: Conv2dLinearModelConfig,
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
pub struct LinearConvTranspose2dModel<B: Backend> {
    pub linear: LinearModel<B>,
    conv_input_size: Ignored<[usize; 2]>,
    intermediate_interpolate: Option<Interpolate2d>,
    pub conv: ConvTranspose2dModel<B>,
    output_interpolate: Option<Interpolate2d>,
}

/// Source: Claude Sonnet 4.5
fn resize_to_aspect(len: usize, aspect_ratio: f64) -> (usize, usize) {
    let mut best_width = len;
    let mut best_height = 1;
    let mut best_diff = f64::INFINITY;

    for h in 1..=(len as f64).sqrt() as usize + 1 {
        if len % h == 0 {
            let w = len / h;
            let current_aspect = w as f64 / h as f64;
            let diff = (current_aspect - aspect_ratio).abs();

            if diff < best_diff {
                best_diff = diff;
                best_width = w;
                best_height = h;
            }
        }
    }

    (best_width, best_height)
}

impl<B: Backend> SimpleInfer<B, 2, 4> for LinearConvTranspose2dModel<B> {
    fn forward(&self, mut tensor: Tensor<B, 2>) -> Tensor<B, 4> {
        tensor = self.linear.infer(tensor);

        let mut tensor = if let Some(interpolate) = &self.intermediate_interpolate {
            let [batch_size, len] = tensor.dims();
            let aspect = self.conv_input_size[0] as f64 / self.conv_input_size[1] as f64;
            let (width, height) = resize_to_aspect(len / self.conv.get_input_channels(), aspect);
            let tensor =
                tensor.reshape([batch_size, self.conv.get_input_channels(), width, height]);
            interpolate.forward(tensor)
        } else {
            let batch_size = tensor.dims()[0];
            tensor.reshape([
                batch_size,
                self.conv.get_input_channels(),
                self.conv_input_size[0],
                self.conv_input_size[1],
            ])
        };

        tensor = self.conv.infer(tensor);

        if let Some(interpolate) = &self.output_interpolate {
            interpolate.forward(tensor)
        } else {
            tensor
        }
    }
}

impl<B: Backend> SimpleTrain<B, 2, 4> for LinearConvTranspose2dModel<B> {
    fn forward(&self, mut tensor: Tensor<B, 2>) -> Tensor<B, 4> {
        tensor = self.linear.train(tensor);

        let mut tensor = if let Some(interpolate) = &self.intermediate_interpolate {
            let [batch_size, len] = tensor.dims();
            let aspect = self.conv_input_size[0] as f64 / self.conv_input_size[1] as f64;
            let (width, height) = resize_to_aspect(len / self.conv.get_input_channels(), aspect);
            let tensor =
                tensor.reshape([batch_size, self.conv.get_input_channels(), width, height]);
            interpolate.forward(tensor)
        } else {
            let batch_size = tensor.dims()[0];
            tensor.reshape([
                batch_size,
                self.conv.get_input_channels(),
                self.conv_input_size[0],
                self.conv_input_size[1],
            ])
        };

        tensor = self.conv.train(tensor);

        if let Some(interpolate) = &self.output_interpolate {
            interpolate.forward(tensor)
        } else {
            tensor
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LinearConvTranspose2dModelConfig {
    pub linear: LinearModelConfig,
    pub intermediate_interpolate: Option<InterpolateMode>,
    pub conv_input_size: [usize; 2],
    pub conv: ConvTranspose2dModelConfig,
    pub output_interpolate: Option<Interpolate2dConfig>,
}

impl<B: Backend> Init<B, LinearConvTranspose2dModel<B>> for LinearConvTranspose2dModelConfig {
    fn init(self, device: &<B as Backend>::Device) -> LinearConvTranspose2dModel<B> {
        LinearConvTranspose2dModel {
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
