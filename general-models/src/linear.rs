use burn::{
    module::Module,
    nn::{Dropout, DropoutConfig, Linear, LinearConfig, activation::Activation},
    prelude::*,
    tensor::activation::softmax,
};
use serde::{Deserialize, Serialize};
use utils::default_f;

use crate::{
    Init, SimpleInfer, SimpleTrain,
    common::{ActivationConfig, Either, Norm, NormConfig, Optional, handle_norm_activation},
};

#[derive(Debug, Module)]
pub struct LinearModel<B: Backend> {
    layers: Vec<(Linear<B>, Option<Norm<B>>, Option<Activation<B>>)>,
    dropout: Dropout,
    dropout_last: bool,
}

impl<B: Backend> SimpleTrain<B, 2, 2> for LinearModel<B> {
    fn forward(&self, mut tensor: burn::Tensor<B, 2>) -> burn::Tensor<B, 2> {
        for (i, (linear, norm, activation)) in self.layers.iter().enumerate() {
            tensor = linear.forward(tensor);
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

impl<B: Backend> SimpleInfer<B, 2, 2> for LinearModel<B> {
    fn forward(&self, mut tensor: burn::Tensor<B, 2>) -> burn::Tensor<B, 2> {
        for (linear, norm, activation) in self.layers.iter() {
            tensor = linear.forward(tensor);
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
impl<B: Backend> LinearModel<B> {
    pub fn iter_layers(
        &mut self,
        mut map: impl FnMut(
            Linear<B>,
            Option<Norm<B>>,
            Option<Activation<B>>,
        ) -> (Linear<B>, Option<Norm<B>>, Option<Activation<B>>),
    ) {
        self.layers = std::mem::take(&mut self.layers)
            .into_iter()
            .map(|(linear, norm, activation)| map(linear, norm, activation))
            .collect();
    }

    pub fn get_input_size(&self) -> usize {
        self.layers[0].0.weight.dims()[1]
    }

    pub fn get_output_size(&self) -> usize {
        self.layers.last().unwrap().0.weight.dims()[1]
    }
}

#[derive(Serialize, Debug, Deserialize, Clone)]
pub struct LinearModelConfig {
    pub input_size: usize,
    pub default_activation: Option<ActivationConfig>,
    pub default_norm: Option<NormConfig>,
    pub layers: Vec<Either<usize, Optional<ActivationConfig>, Optional<NormConfig>, Option<f64>>>,
    #[serde(default = "default_dropout")]
    pub dropout: f64,
    #[serde(default = "default_dropout_last")]
    pub dropout_last: bool,
    pub weights_gain: Option<f64>
}

impl<B: Backend> Init<B, LinearModel<B>> for LinearModelConfig {
    fn init(self, device: &B::Device) -> LinearModel<B> {
        let mut input_size = self.input_size;
        let mut layers = vec![];
        for (output_size, activation, norm, weights_gain) in self.layers.into_iter().map(Either::into_tuple) {
            let (norm, activation, init) = handle_norm_activation(
                norm,
                activation,
                &self.default_norm,
                &self.default_activation,
                weights_gain,
                output_size,
                device
            );
            layers.push((
                LinearConfig::new(input_size, output_size)
                    .with_bias(norm.is_none())
                    .with_initializer(init)
                    .init(device),
                norm,
                activation
            ));
            input_size = output_size;
        }
        LinearModel {
            layers,
            dropout: DropoutConfig::new(self.dropout).init(),
            dropout_last: self.dropout_last,
        }
    }
}

#[derive(Module, Debug)]
pub struct LinearClassifierModel<B: Backend> {
    pub linear: LinearModel<B>,
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

impl<B: Backend> Init<B, LinearClassifierModel<B>> for LinearClassifierModelConfig {
    fn init(self, device: &B::Device) -> LinearClassifierModel<B> {
        LinearClassifierModel {
            linear: self.0.init(device),
        }
    }
}

default_f!(default_dropout, f64, 0.0);
default_f!(default_dropout_last, bool, true);
