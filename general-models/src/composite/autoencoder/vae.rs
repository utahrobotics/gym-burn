use burn::{module::ModuleDisplay, prelude::*, tensor::Distribution};
use serde::{Deserialize, Serialize};

use crate::{
    Init, SimpleInfer, SimpleTrain, default_f, linear::{LinearModel, LinearModelConfig}
};

#[derive(Module, Debug)]
pub struct VariationalEncoder<B: Backend, M> {
    model: M,
    kld_weight: f64,
    mean: LinearModel<B>,
    logvar: LinearModel<B>,
}

impl<B, M, const D: usize> SimpleInfer<B, D, 2> for VariationalEncoder<B, M>
where
    B: Backend,
    M: SimpleInfer<B, D, 2> + ModuleDisplay,
{
    fn forward(&self, tensor: Tensor<B, D>) -> Tensor<B, 2> {
        self.mean.infer(self.model.infer(tensor))
    }
}

impl<B: Backend, M> VariationalEncoder<B, M>
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

    pub fn reparameterize(&self, mu: Tensor<B, 2>, logvar: Tensor<B, 2>) -> Tensor<B, 2> {
        let std = logvar.mul_scalar(0.5).exp();

        // Sample epsilon from standard Gaussian (0, 1)
        let epsilon = Tensor::random(mu.shape(), Distribution::Normal(0.0, 1.0), &mu.device());

        // z = mu + std * epsilon
        mu + std.mul(epsilon)
    }
    
    pub fn get_kld_weight(&self) -> f64 {
        self.kld_weight
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct VariationalEncoderConfig<M> {
    pub model: M,
    #[serde(default = "default_kld_weight")]
    pub kld_weight: f64,
    pub mean: LinearModelConfig,
    pub logvar: LinearModelConfig,
}

impl<B, M> Init<B> for VariationalEncoderConfig<M>
where
    B: Backend,
    M: Init<B>,
{
    type Output = VariationalEncoder<B, M::Output>;

    fn init(self, device: &<B as Backend>::Device) -> Self::Output {
        VariationalEncoder {
            model: self.model.init(device),
            mean: self.mean.init(device),
            kld_weight: self.kld_weight,
            logvar: self.logvar.init(device),
        }
    }
}

default_f!(default_kld_weight, f64, 0.1);
