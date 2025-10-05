use burn::{
    module::Module,
    nn::{
        Dropout, DropoutConfig, Linear, LinearConfig,
        activation::Activation,
    },
    prelude::*,
    tensor::activation::softmax,
};
use serde::{Deserialize, Serialize};

use crate::{
    Init, SimpleInfer, SimpleTrain, common::{ActivationConfig, Either, Norm, NormConfig}, default_f
};

#[derive(Debug, Module)]
pub struct LinearModel<B: Backend> {
    layers: Vec<(Linear<B>, Option<Norm<B>>, Activation<B>)>,
    dropout: Dropout,
}

impl<B: Backend> SimpleInfer<B, 2, 2> for LinearModel<B> {
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

impl<B: Backend> SimpleTrain<B, 2, 2> for LinearModel<B> {
    fn forward(&self, mut tensor: burn::Tensor<B, 2>) -> burn::Tensor<B, 2> {
        for (linear, norm, activation) in self.layers.iter() {
            tensor = linear.forward(tensor);
            if let Some(norm) = norm {
                tensor = norm.forward(tensor);
            }
            tensor = activation.forward(tensor);
        }
        tensor
    }
}

#[derive(Serialize, Debug, Deserialize, Clone)]
pub struct LinearModelConfig {
    pub input_size: usize,
    pub default_activation: Option<ActivationConfig>,
    pub default_norm: Option<NormConfig>,
    pub layers: Vec<Either<usize, NormConfig, ActivationConfig>>,
    #[serde(default = "default_dropout")]
    pub dropout: f64,
}

impl<B: Backend> Init<B> for LinearModelConfig {
    type Output = LinearModel<B>;

    fn init(self, device: &B::Device) -> Self::Output {
        let default_activation = self.default_activation.unwrap_or(ActivationConfig::GELU);
        let mut input_size = self.input_size;
        let mut layers = vec![];
        for (output_size, norm, activation) in self.layers.into_iter().map(Either::into_tuple) {
            layers.push((
                LinearConfig::new(input_size, output_size)
                    .with_bias(norm.is_none())
                    .init(device),
                norm.or_else(|| self.default_norm.clone()).map(|norm| norm.init(device, output_size)),
                activation
                    .unwrap_or_else(|| default_activation.clone())
                    .init(device),
            ));
            input_size = output_size;
        }
        LinearModel {
            layers,
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

#[derive(Module, Debug)]
pub struct LinearClassifierModel<B: Backend> {
    linear: LinearModel<B>,
}

impl<B: Backend> SimpleInfer<B, 2, 2> for LinearClassifierModel<B> {
    fn forward(&self, tensor: burn::Tensor<B, 2>) -> burn::Tensor<B, 2> {
        softmax(self.linear.infer(tensor), 1)
    }
}

impl<B: Backend> SimpleTrain<B, 2, 2> for LinearClassifierModel<B> {
    fn forward(&self, tensor: burn::Tensor<B, 2>) -> burn::Tensor<B, 2> {
        self.linear.train(tensor)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LinearClassifierModelConfig(LinearModelConfig);

impl<B: Backend> Init<B> for LinearClassifierModelConfig {
    type Output = LinearClassifierModel<B>;

    fn init(self, device: &B::Device) -> Self::Output {
        LinearClassifierModel {
            linear: self.0.init(device),
        }
    }
}

default_f!(default_dropout, f64, 0.0);
