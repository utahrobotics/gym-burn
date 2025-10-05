use burn::{module::Module, nn::{Dropout, Linear}, prelude::Backend};

use crate::{SimpleForwardable, common::{Activation, Norm}};

#[derive(Debug, Module)]
pub struct LinearModel<B: Backend> {
    layers: Vec<(Linear<B>, Option<Norm<B>>, Activation)>,
    dropout: Dropout
}

impl<B: Backend> SimpleForwardable<B, 2, 2> for LinearModel<B> {
    fn forward(&self, mut tensor: burn::Tensor<B, 2>) -> burn::Tensor<B, 2> {
        for (i, (linear, norm, activation)) in self.layers.iter().enumerate() {
            tensor = linear.forward(tensor);
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
