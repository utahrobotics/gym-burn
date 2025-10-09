use burn::{
    tensor::backend::AutodiffBackend,
};
use serde::de::DeserializeOwned;
use utils::default_f;


pub mod lr_scheduler;
pub mod optimizer;

pub mod conv;
pub mod image;
pub mod linear;
pub mod autoencoder;

pub trait ApplyGradients<B: AutodiffBackend> {
    type Plan;
    type PlanConfig: DeserializeOwned;

    fn config_to_plan(config: Self::PlanConfig) -> Self::Plan;
    fn apply_gradients(
        &mut self,
        lr: f64,
        grads: &mut <B as AutodiffBackend>::Gradients,
        plan: &mut Self::Plan,
    );
}

default_f!(default_lr_multiplier, f64, 1.0);
