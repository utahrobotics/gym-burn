use burn::{module::ModuleDisplay, prelude::*, tensor::Distribution};
use serde::{Deserialize, Serialize};

use crate::{linear::{LinearModel, LinearModelConfig}, Init, SimpleInfer, SimpleTrain};

#[derive(Module, Debug)]
pub struct VariationalEncoder<B: Backend, M> {
    model: M,
    mean: LinearModel<B>,
    logvar: LinearModel<B>,
}

impl<B, M, const D: usize> SimpleInfer<B, D, 2> for VariationalEncoder<B, M>
where
    B: Backend,
    M: SimpleInfer<B, D, 2> + ModuleDisplay
{
    fn forward(&self, tensor: Tensor<B, D>) -> Tensor<B, 2> {
        self.mean.infer(self.model.infer(tensor))
    }
}

impl<B: Backend, M> VariationalEncoder<B, M>
where
    B: Backend
{
    pub fn train<const D: usize>(&self, tensor: Tensor<B, D>) -> (Tensor<B, 2>, Tensor<B, 2>)
    where
        M: SimpleInfer<B, D, 2> + ModuleDisplay
    {
        let latent = self.model.infer(tensor);
        (
            self.mean.train(latent.clone()),
            self.logvar.train(latent),
        )
    }
    
    pub fn reparameterize(&self, mu: Tensor<B, 2>, logvar: Tensor<B, 2>) -> Tensor<B, 2> {
        let std = logvar.mul_scalar(0.5).exp();
        
        // Sample epsilon from standard Gaussian (0, 1)
        let epsilon = Tensor::random(
            mu.shape(), 
            Distribution::Normal(0.0, 1.0),
            &mu.device(),
        );

        // z = mu + std * epsilon
        mu + std.mul(epsilon)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct VariationalEncoderConfig<M> {
    pub model: M,
    pub mean: LinearModelConfig,
    pub logvar: LinearModelConfig,
}

impl<B, M> Init<B> for VariationalEncoderConfig<M>
where
    B: Backend,
    M: Init<B>
{
    type Output = VariationalEncoder<B, M::Output>;

    fn init(self, device: &<B as Backend>::Device) -> Self::Output {
        VariationalEncoder {
            model: self.model.init(device),
            mean: self.mean.init(device),
            logvar: self.logvar.init(device),
        }
    }
}
