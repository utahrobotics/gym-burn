use burn::tensor::backend::AutodiffBackend;
use general_models::{
    composite::autoencoder::{AutoEncoderModel, vae::VariationalEncoderModel},
    linear::LinearModel,
};
use serde::{Deserialize, Serialize};

use crate::trainable_models::apply_gradients::{
    ApplyGradients,
    linear::{LinearModelPlan, LinearModelPlanConfig},
};

pub struct AutoEncoderModelPlan<E, D> {
    encoder: E,
    decoder: D,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AutoEncoderModelPlanConfig<E, D> {
    pub encoder: E,
    pub decoder: D,
}

impl<B: AutodiffBackend, E: ApplyGradients<B>, D: ApplyGradients<B>> ApplyGradients<B>
    for AutoEncoderModel<B, E, D>
{
    type Plan = AutoEncoderModelPlan<E::Plan, D::Plan>;
    type PlanConfig = AutoEncoderModelPlanConfig<E::PlanConfig, D::PlanConfig>;

    fn apply_gradients(
        &mut self,
        lr: f64,
        grads: &mut <B as AutodiffBackend>::Gradients,
        plan: &mut Self::Plan,
    ) {
        self.encoder.apply_gradients(lr, grads, &mut plan.encoder);
        self.decoder.apply_gradients(lr, grads, &mut plan.decoder);
    }

    fn config_to_plan(config: Self::PlanConfig) -> Self::Plan {
        AutoEncoderModelPlan {
            encoder: E::config_to_plan(config.encoder),
            decoder: D::config_to_plan(config.decoder),
        }
    }
}

pub struct VariationalEncoderModelPlan<B: AutodiffBackend, T> {
    model_plan: T,
    mean_plan: LinearModelPlan<B>,
    logvar_plan: LinearModelPlan<B>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VariationalEncoderModelPlanConfig<T> {
    pub model_plan: T,
    pub mean_plan: LinearModelPlanConfig,
    pub logvar_plan: LinearModelPlanConfig,
}

impl<B: AutodiffBackend, T: ApplyGradients<B>> ApplyGradients<B> for VariationalEncoderModel<B, T> {
    type Plan = VariationalEncoderModelPlan<B, T::Plan>;
    type PlanConfig = VariationalEncoderModelPlanConfig<T::PlanConfig>;

    fn apply_gradients(
        &mut self,
        lr: f64,
        grads: &mut <B as AutodiffBackend>::Gradients,
        plan: &mut Self::Plan,
    ) {
        self.model.apply_gradients(lr, grads, &mut plan.model_plan);
        self.mean.apply_gradients(lr, grads, &mut plan.mean_plan);
        self.logvar
            .apply_gradients(lr, grads, &mut plan.logvar_plan);
    }

    fn config_to_plan(config: Self::PlanConfig) -> Self::Plan {
        VariationalEncoderModelPlan {
            model_plan: T::config_to_plan(config.model_plan),
            mean_plan: LinearModel::config_to_plan(config.mean_plan),
            logvar_plan: LinearModel::config_to_plan(config.logvar_plan),
        }
    }
}
