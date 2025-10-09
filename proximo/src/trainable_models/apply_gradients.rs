use burn::{
    nn::{Linear, activation::Activation},
    optim::GradientsParams,
    tensor::backend::AutodiffBackend,
};
use general_models::{
    common::Norm, composite::autoencoder::vae::VariationalEncoderModel, linear::LinearModel,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use utils::default_f;

use crate::trainable_models::apply_gradients::optimizer::{Optimizer, OptimizerConfig};

pub mod lr_scheduler;
pub mod optimizer;

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

pub struct LinearModelPlan<B: AutodiffBackend> {
    weights_optim: Optimizer<B, Linear<B>>,
    bias_optim: Option<Optimizer<B, Linear<B>>>,
    norm_optim: Optimizer<B, Norm<B>>,
    activation_optim: Optimizer<B, Activation<B>>,
    weights_lr_multiplier: f64,
    bias_lr_multiplier: f64,
    norm_lr_multiplier: f64,
    activation_lr_multiplier: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LinearModelPlanConfig {
    pub weights_optim: OptimizerConfig,
    pub bias_optim: Option<OptimizerConfig>,
    pub norm_optim: Option<OptimizerConfig>,
    pub activation_optim: Option<OptimizerConfig>,
    #[serde(default = "default_lr_multiplier")]
    pub weights_lr_multiplier: f64,
    #[serde(default = "default_lr_multiplier")]
    pub bias_lr_multiplier: f64,
    #[serde(default = "default_lr_multiplier")]
    pub norm_lr_multiplier: f64,
    #[serde(default = "default_lr_multiplier")]
    pub activation_lr_multiplier: f64,
}

impl<B: AutodiffBackend> ApplyGradients<B> for LinearModel<B> {
    type Plan = LinearModelPlan<B>;
    type PlanConfig = LinearModelPlanConfig;

    fn apply_gradients(
        &mut self,
        lr: f64,
        grads: &mut <B as AutodiffBackend>::Gradients,
        plan: &mut Self::Plan,
    ) {
        self.iter_layers(|mut linear, mut norm, mut activation| {
            if let Some(bias_optim) = &mut plan.bias_optim {
                if let Some(bias) = &linear.bias {
                    let grad_params = GradientsParams::from_module(grads, bias);
                    linear = bias_optim.step(lr * plan.bias_lr_multiplier, linear, grad_params);
                }
                let grad_params = GradientsParams::from_module(grads, &linear.weight);
                linear =
                    plan.weights_optim
                        .step(lr * plan.weights_lr_multiplier, linear, grad_params);
            } else {
                let grad_params = GradientsParams::from_module(grads, &linear);
                linear =
                    plan.weights_optim
                        .step(lr * plan.weights_lr_multiplier, linear, grad_params);
            }

            norm = if let Some(norm) = norm {
                let grad_params = GradientsParams::from_module(grads, &norm);
                Some(
                    plan.norm_optim
                        .step(lr * plan.norm_lr_multiplier, norm, grad_params),
                )
            } else {
                None
            };

            activation = if let Some(activation) = activation {
                let grad_params = GradientsParams::from_module(grads, &activation);
                Some(plan.activation_optim.step(
                    lr * plan.activation_lr_multiplier,
                    activation,
                    grad_params,
                ))
            } else {
                None
            };

            (linear, norm, activation)
        });
    }

    fn config_to_plan(config: Self::PlanConfig) -> Self::Plan {
        LinearModelPlan {
            bias_optim: config.bias_optim.map(|x| x.init()),
            norm_optim: config
                .norm_optim
                .unwrap_or_else(|| config.weights_optim.clone())
                .init(),
            activation_optim: config
                .activation_optim
                .unwrap_or_else(|| config.weights_optim.clone())
                .init(),
            weights_optim: config.weights_optim.init(),
            weights_lr_multiplier: config.weights_lr_multiplier,
            bias_lr_multiplier: config.bias_lr_multiplier,
            norm_lr_multiplier: config.norm_lr_multiplier,
            activation_lr_multiplier: config.activation_lr_multiplier,
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

default_f!(default_lr_multiplier, f64, 1.0);
