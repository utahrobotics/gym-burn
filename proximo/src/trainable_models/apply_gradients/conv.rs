use burn::{
    nn::{
        activation::Activation,
        conv::{Conv2d, ConvTranspose2d},
    },
    optim::GradientsParams,
};

use serde::{Deserialize, Serialize};

use super::ApplyGradients;

use crate::trainable_models::apply_gradients::optimizer::OptimizerConfig;

use general_models::{
    common::Norm,
    conv::{Conv2dModel, ConvTranspose2dModel},
};

use crate::trainable_models::apply_gradients::optimizer::Optimizer;

use burn::tensor::backend::AutodiffBackend;

use super::default_lr_multiplier;

pub struct Conv2dModelPlan<B: AutodiffBackend> {
    weights_optim: Optimizer<B, Conv2d<B>>,
    bias_optim: Option<Optimizer<B, Conv2d<B>>>,
    norm_optim: Optimizer<B, Norm<B>>,
    activation_optim: Optimizer<B, Activation<B>>,
    weights_lr_multiplier: f64,
    bias_lr_multiplier: f64,
    norm_lr_multiplier: f64,
    activation_lr_multiplier: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Conv2dModelPlanConfig {
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

impl<B: AutodiffBackend> ApplyGradients<B> for Conv2dModel<B> {
    type Plan = Conv2dModelPlan<B>;
    type PlanConfig = Conv2dModelPlanConfig;

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
        Conv2dModelPlan {
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

pub struct ConvTranspose2dModelPlan<B: AutodiffBackend> {
    weights_optim: Optimizer<B, ConvTranspose2d<B>>,
    bias_optim: Option<Optimizer<B, ConvTranspose2d<B>>>,
    norm_optim: Optimizer<B, Norm<B>>,
    activation_optim: Optimizer<B, Activation<B>>,
    weights_lr_multiplier: f64,
    bias_lr_multiplier: f64,
    norm_lr_multiplier: f64,
    activation_lr_multiplier: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(transparent)]
pub struct ConvTranspose2dModelPlanConfig(pub Conv2dModelPlanConfig);

impl<B: AutodiffBackend> ApplyGradients<B> for ConvTranspose2dModel<B> {
    type Plan = ConvTranspose2dModelPlan<B>;
    type PlanConfig = ConvTranspose2dModelPlanConfig;

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
        let config = config.0;
        ConvTranspose2dModelPlan {
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
