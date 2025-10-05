use burn::{nn::interpolate::Interpolate2d, prelude::*};

use crate::{conv::ConvTranspose2dModel, linear::LinearModel, SimpleInfer, SimpleTrain};

#[derive(Debug, Module)]
pub struct LinearConvDecoderModel<B: Backend> {
    linear: LinearModel<B>,
    conv_input_size: [i32; 2],
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
