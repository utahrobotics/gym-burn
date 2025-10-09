use burn::tensor::backend::AutodiffBackend;
use general_models::{composite::image::{ConvLinearModel, LinearConvTranspose2dModel}, conv::{Conv2dModel, ConvTranspose2dModel}, linear::LinearModel};
use serde::{Deserialize, Serialize};

use crate::trainable_models::apply_gradients::{ApplyGradients, conv::{Conv2dModelPlan, Conv2dModelPlanConfig, ConvTranspose2dModelPlan, ConvTranspose2dModelPlanConfig}, linear::{LinearModelPlan, LinearModelPlanConfig}};


pub struct ConvLinearModelPlan<B: AutodiffBackend> {
    conv: Conv2dModelPlan<B>,
    linear: LinearModelPlan<B>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ConvLinearModelPlanConfig {
    pub conv: Conv2dModelPlanConfig,
    pub linear: LinearModelPlanConfig,
}

impl<B: AutodiffBackend> ApplyGradients<B> for ConvLinearModel<B> {
    type Plan = ConvLinearModelPlan<B>;
    type PlanConfig = ConvLinearModelPlanConfig;

    fn config_to_plan(config: Self::PlanConfig) -> Self::Plan {
        ConvLinearModelPlan {
            conv: Conv2dModel::config_to_plan(config.conv),
            linear: LinearModel::config_to_plan(config.linear),
        }
    }

    fn apply_gradients(
        &mut self,
        lr: f64,
        grads: &mut <B as AutodiffBackend>::Gradients,
        plan: &mut Self::Plan,
    ) {
        self.conv.apply_gradients(lr, grads, &mut plan.conv);
        self.linear.apply_gradients(lr, grads, &mut plan.linear);
    }
}

pub struct LinearConvTranspose2dModelPlan<B: AutodiffBackend> {
    linear: LinearModelPlan<B>,
    conv: ConvTranspose2dModelPlan<B>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LinearConvTranspose2dModelPlanConfig {
    pub linear: LinearModelPlanConfig,
    pub conv: ConvTranspose2dModelPlanConfig,
}

impl<B: AutodiffBackend> ApplyGradients<B> for LinearConvTranspose2dModel<B> {
    type Plan = LinearConvTranspose2dModelPlan<B>;
    type PlanConfig = LinearConvTranspose2dModelPlanConfig;

    fn config_to_plan(config: Self::PlanConfig) -> Self::Plan {
        LinearConvTranspose2dModelPlan {
            linear: LinearModel::config_to_plan(config.linear),
            conv: ConvTranspose2dModel::config_to_plan(config.conv),
        }
    }

    fn apply_gradients(
        &mut self,
        lr: f64,
        grads: &mut <B as AutodiffBackend>::Gradients,
        plan: &mut Self::Plan,
    ) {
        self.conv.apply_gradients(lr, grads, &mut plan.conv);
        self.linear.apply_gradients(lr, grads, &mut plan.linear);
    }
}