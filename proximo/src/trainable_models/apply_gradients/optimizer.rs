use burn::{grad_clipping::GradientClippingConfig, module::AutodiffModule, optim::{Adam, AdamConfig, GradientsParams, Optimizer as BurnOptimizer, adaptor::OptimizerAdaptor, decay::WeightDecayConfig}, tensor::backend::AutodiffBackend};
use serde::{Deserialize, Serialize};
use utils::default_f;

pub enum Optimizer<B: AutodiffBackend, M: AutodiffModule<B>> {
    Adam(OptimizerAdaptor<Adam, M, B>)
}

impl<B: AutodiffBackend, M: AutodiffModule<B>> Optimizer<B, M> {
    pub fn step(&mut self, lr: f64, module: M, grads: GradientsParams) -> M {
        match self {
            Optimizer::Adam(optimizer_adaptor) => optimizer_adaptor.step(lr, module, grads),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum OptimizerConfig {
    Adam {
        #[serde(default = "default_beta_1")]
        beta_1: f32,
        #[serde(default = "default_beta_2")]
        beta_2: f32,
        #[serde(default = "default_eps")]
        eps: f32,
        weight_decay: Option<f32>,
        grad_clipping: Option<GradientClippingConfig>,
    }
}

impl OptimizerConfig {
    pub fn init<B: AutodiffBackend, M: AutodiffModule<B>>(self) -> Optimizer<B, M> {
        match self {
            OptimizerConfig::Adam { beta_1, beta_2, eps, weight_decay, grad_clipping } => Optimizer::Adam(
                AdamConfig::new()
                    .with_beta_1(beta_1)
                    .with_beta_2(beta_2)
                    .with_epsilon(eps)
                    .with_weight_decay(weight_decay.map(|penalty| WeightDecayConfig { penalty }))
                    .with_grad_clipping(grad_clipping)
                    .init()
            )
        }
    }
}

default_f!(default_beta_1, f32, 0.9);
default_f!(default_beta_2, f32, 0.999);
default_f!(default_eps, f32, 1e-5);