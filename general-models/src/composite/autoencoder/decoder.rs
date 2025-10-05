use burn::{module::Ignored, nn::interpolate::{Interpolate2d, Interpolate2dConfig}, prelude::*};
use serde::{Deserialize, Serialize};

use crate::{conv::{ConvTranspose2dModel, ConvTranspose2dModelConfig}, linear::{LinearModel, LinearModelConfig}, Init, SimpleInfer, SimpleTrain};

#[derive(Debug, Module)]
pub struct LinearConvDecoderModel<B: Backend> {
    linear: LinearModel<B>,
    conv_input_size: Ignored<[i32; 2]>,
    conv: ConvTranspose2dModel<B>,
    interpolate: Option<Interpolate2d>
}

impl<B: Backend> SimpleInfer<B, 2, 4> for LinearConvDecoderModel<B> {
    fn forward(&self, tensor: Tensor<B, 2>) -> Tensor<B, 4> {
        let batch_size = tensor.dims()[0] as i32;
        let tensor = self.conv.infer(self.linear.infer(tensor).reshape([batch_size, -1, self.conv_input_size[0], self.conv_input_size[1]]));
        if let Some(interpolate) = &self.interpolate {
            interpolate.forward(tensor)
        } else {
            tensor
        }
    }
}

impl<B: Backend> SimpleTrain<B,2, 4> for LinearConvDecoderModel<B> {
    fn forward(&self, tensor: Tensor<B, 2>) -> Tensor<B, 4> {
        let batch_size = tensor.dims()[0] as i32;
        let tensor = self.conv.train(self.linear.train(tensor).reshape([batch_size, -1, self.conv_input_size[0], self.conv_input_size[1]]));
        if let Some(interpolate) = &self.interpolate {
            interpolate.forward(tensor)
        } else {
            tensor
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LinearConvDecoderModelConfig {
    pub linear: LinearModelConfig,
    pub conv_input_size: [u32; 2],
    pub conv: ConvTranspose2dModelConfig,
    pub interpolate: Option<Interpolate2dConfig>
}

impl<B: Backend> Init<B> for LinearConvDecoderModelConfig {
    type Output = LinearConvDecoderModel<B>;

    fn init(self, device: &<B as Backend>::Device) -> Self::Output {
        LinearConvDecoderModel {
            linear: self.linear.init(device),
            conv_input_size: Ignored([self.conv_input_size[0] as i32, self.conv_input_size[1] as i32]),
            conv: self.conv.init(device),
            interpolate: self.interpolate.map(|x| x.init()),
        }
    }
}