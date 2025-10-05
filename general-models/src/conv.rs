use burn::{module::Module, nn::{Dropout, activation::Activation, conv::{Conv2d, ConvTranspose2d}}, prelude::Backend};

use crate::{SimpleInfer, SimpleTrain, common::{Norm}};

#[derive(Debug, Module)]
pub struct Conv2dModel<B: Backend> {
    layers: Vec<(Conv2d<B>, Option<Norm<B>>, Activation<B>)>,
    dropout: Dropout
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

#[derive(Debug, Module)]
pub struct ConvTranspose2dModel<B: Backend> {
    layers: Vec<(ConvTranspose2d<B>, Option<Norm<B>>, Activation)>,
    dropout: Dropout
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
