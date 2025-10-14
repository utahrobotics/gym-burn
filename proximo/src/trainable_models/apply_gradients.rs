use burn::tensor::backend::AutodiffBackend;
use serde::de::DeserializeOwned;
use utils::default_f;

use crate::trainable_models::AdHocLossModel;

pub mod lr_scheduler;
pub mod optimizer;

pub mod autoencoder;
pub mod conv;
pub mod image;
pub mod linear;

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

impl<B: AutodiffBackend, M: ApplyGradients<B>> ApplyGradients<B> for &mut M {
    type Plan = M::Plan;
    type PlanConfig = M::PlanConfig;

    fn config_to_plan(config: Self::PlanConfig) -> Self::Plan {
        M::config_to_plan(config)
    }

    fn apply_gradients(
        &mut self,
        lr: f64,
        grads: &mut <B as AutodiffBackend>::Gradients,
        plan: &mut Self::Plan,
    ) {
        <M as ApplyGradients<B>>::apply_gradients(self, lr, grads, plan);
    }
}

impl<B: AutodiffBackend, F, M: ApplyGradients<B>> ApplyGradients<B> for AdHocLossModel<M, F> {
    type Plan = M::Plan;
    type PlanConfig = M::PlanConfig;

    fn config_to_plan(config: Self::PlanConfig) -> Self::Plan {
        M::config_to_plan(config)
    }

    fn apply_gradients(
        &mut self,
        lr: f64,
        grads: &mut <B as AutodiffBackend>::Gradients,
        plan: &mut Self::Plan,
    ) {
        self.model.apply_gradients(lr, grads, plan);
    }
}

default_f!(default_lr_multiplier, f64, 1.0);
