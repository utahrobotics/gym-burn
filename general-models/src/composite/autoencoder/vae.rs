use burn::{module::ModuleDisplay, prelude::*, tensor::Distribution};
use serde::{Deserialize, Serialize};
use utils::default_f;

use crate::{
    Init, SimpleInfer, SimpleTrain,
    linear::{LinearModel, LinearModelConfig},
};

#[derive(Module, Debug)]
pub struct VariationalEncoderModel<B: Backend, M> {
    pub model: M,
    pub mean: LinearModel<B>,
    pub logvar: LinearModel<B>,
}

impl<B, M, const D: usize> SimpleInfer<B, D, 2> for VariationalEncoderModel<B, M>
where
    B: Backend,
    M: SimpleInfer<B, D, 2> + ModuleDisplay,
{
    fn forward(&self, tensor: Tensor<B, D>) -> Tensor<B, 2> {
        self.mean.infer(self.model.infer(tensor))
    }
}

impl<B: Backend, M> VariationalEncoderModel<B, M>
where
    B: Backend,
{
    pub fn train<const D: usize>(&self, tensor: Tensor<B, D>) -> (Tensor<B, 2>, Tensor<B, 2>)
    where
        M: SimpleTrain<B, D, 2>,
    {
        let latent = self.model.train(tensor);
        (self.mean.train(latent.clone()), self.logvar.train(latent))
    }

    pub fn get_model(&self) -> &M {
        &self.model
    }

    pub fn reparameterize(&self, mu: Tensor<B, 2>, logvar: Tensor<B, 2>) -> Tensor<B, 2> {
        let std = logvar.mul_scalar(0.5).exp();

        // Sample epsilon from standard Gaussian (0, 1)
        let epsilon = Tensor::random(mu.shape(), Distribution::Normal(0.0, 1.0), &mu.device());

        // z = mu + std * epsilon
        mu + std.mul(epsilon)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct VariationalEncoderModelConfig<M> {
    pub model: M,
    #[serde(default = "default_kld_weight")]
    pub kld_weight: f64,
    pub mean: LinearModelConfig,
    pub logvar: LinearModelConfig,
}

impl<B, M, T> Init<B, VariationalEncoderModel<B, T>> for VariationalEncoderModelConfig<M>
where
    B: Backend,
    M: Init<B, T>,
{
    fn init(self, device: &<B as Backend>::Device) -> VariationalEncoderModel<B, T> {
        VariationalEncoderModel {
            model: self.model.init(device),
            mean: self.mean.init(device),
            logvar: self.logvar.init(device),
        }
    }
}

default_f!(default_kld_weight, f64, 0.1);
