use burn::{
    module::Module,
    nn::{Dropout, DropoutConfig, activation::Activation, gru::{Gru, GruConfig}},
    prelude::*,
};
use serde::{Deserialize, Serialize};
use utils::default_f;

use crate::{
    Init, SimpleInfer, SimpleTrain,
    common::{ActivationConfig, Either, Norm, NormConfig, Optional, handle_norm_activation},
};

#[derive(Debug, Module)]
pub struct GruModel<B: Backend> {
    layers: Vec<(Gru<B>, Option<Norm<B>>, Option<Activation<B>>)>,
    dropout: Dropout,
    dropout_last: bool,
}

impl<B: Backend> SimpleTrain<B, 3, 2> for GruModel<B> {
    fn forward(&self, mut tensor: burn::Tensor<B, 3>) -> burn::Tensor<B, 2> {
        for (i, (gru, norm, activation)) in self.layers.iter().enumerate() {
            tensor = gru.forward(tensor, None);
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
        let [batch_size, _seq_len, hidden_size] = tensor.dims();
        tensor.slice(s![.., -1, ..]).reshape([batch_size, hidden_size])
    }
}

impl<B: Backend> SimpleInfer<B, 3, 2> for GruModel<B> {
    fn forward(&self, mut tensor: burn::Tensor<B, 3>) -> burn::Tensor<B, 2> {
        for (gru, norm, activation) in self.layers.iter() {
            tensor = gru.forward(tensor, None);
            if let Some(norm) = norm {
                tensor = norm.forward(tensor);
            }
            if let Some(activation) = activation {
                tensor = activation.forward(tensor);
            }
        }
        let [batch_size, _seq_len, hidden_size] = tensor.dims();
        tensor.slice(s![.., -1, ..]).reshape([batch_size, hidden_size])
    }
}
impl<B: Backend> GruModel<B> {
    pub fn iter_layers(
        &mut self,
        mut map: impl FnMut(
            Gru<B>,
            Option<Norm<B>>,
            Option<Activation<B>>,
        ) -> (Gru<B>, Option<Norm<B>>, Option<Activation<B>>),
    ) {
        self.layers = std::mem::take(&mut self.layers)
            .into_iter()
            .map(|(gru, norm, activation)| map(gru, norm, activation))
            .collect();
    }

    pub fn get_input_size(&self) -> usize {
        self.layers[0].0.new_gate.input_transform.weight.dims()[1]
    }

    pub fn get_output_size(&self) -> usize {
        self.layers.last().unwrap().0.d_hidden
    }
}

#[derive(Serialize, Debug, Deserialize, Clone)]
pub struct GruModelConfig {
    pub input_size: usize,
    pub default_activation: Option<ActivationConfig>,
    pub default_norm: Option<NormConfig>,
    pub layers: Vec<Either<usize, Optional<ActivationConfig>, Optional<NormConfig>, Option<f64>>>,
    #[serde(default = "default_dropout")]
    pub dropout: f64,
    #[serde(default = "default_dropout_last")]
    pub dropout_last: bool,
}

impl<B: Backend> Init<B, GruModel<B>> for GruModelConfig {
    fn init(self, device: &B::Device) -> GruModel<B> {
        let mut input_size = self.input_size;
        let mut layers = vec![];
        for (output_size, activation, norm, weights_gain) in
            self.layers.into_iter().map(Either::into_tuple)
        {
            let (norm, activation, init) = handle_norm_activation(
                norm,
                activation,
                &self.default_norm,
                &self.default_activation,
                weights_gain,
                output_size,
                device,
            );
            layers.push((
                GruConfig::new(input_size, output_size, norm.is_none())
                    .with_initializer(init)
                    .init(device),
                norm,
                activation,
            ));
            input_size = output_size;
        }
        GruModel {
            layers,
            dropout: DropoutConfig::new(self.dropout).init(),
            dropout_last: self.dropout_last,
        }
    }
}

default_f!(default_dropout, f64, 0.0);
default_f!(default_dropout_last, bool, true);
