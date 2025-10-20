use burn::{module::AutodiffModule, optim::GradientsParams, tensor::backend::AutodiffBackend};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use utils::default_f;

use crate::trainable_models::{
    AdHocLossModel,
    apply_gradients::optimizer::{Optimizer, OptimizerConfig},
};

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

// pub trait ApplyAllGradients<B: AutodiffBackend> {
//     type Plan;
//     type PlanConfig: DeserializeOwned;

//     fn config_to_plan(config: Self::PlanConfig) -> Self::Plan;
//     fn apply_all_gradients(
//         &mut self,
//         lr: f64,
//         grads: <B as AutodiffBackend>::Gradients,
//         plan: &mut Self::Plan,
//     );
// }

// impl<B: AutodiffBackend, M: ApplyGradients<B>> ApplyAllGradients<B> for M {
//     fn apply_gradients(
//         &mut self,
//         lr: f64,
//         grads: <B as AutodiffBackend>::Gradients,
//         plan: &mut Self::Plan,
//     ) {
//         <M as ApplyAllGradients<B>>::apply_gradients(self, lr, grads, plan);
//     }

//     type Plan = M::Plan;

//     type PlanConfig = M::PlanConfig;

//     fn config_to_plan(config: Self::PlanConfig) -> Self::Plan {
//         M::config_to_plan(config)
//     }
// }

// impl<B: AutodiffBackend, M: ApplyAllGradients<B>> ApplyAllGradients<B> for &mut M {
//     type Plan = M::Plan;
//     type PlanConfig = M::PlanConfig;

//     fn config_to_plan(config: Self::PlanConfig) -> Self::Plan {
//         M::config_to_plan(config)
//     }

//     fn apply_gradients(
//         &mut self,
//         lr: f64,
//         grads: <B as AutodiffBackend>::Gradients,
//         plan: &mut Self::Plan,
//     ) {
//         <M as ApplyAllGradients<B>>::apply_gradients(self, lr, grads, plan);
//     }
// }

// impl<B: AutodiffBackend, M: ApplyGradients<B>> ApplyGradients<B> for &mut M {
//     fn apply_gradients(
//         &mut self,
//         lr: f64,
//         grads: &mut <B as AutodiffBackend>::Gradients,
//         plan: &mut Self::Plan,
//     ) {
//         <M as ApplyGradients<B>>::apply_gradients(self, lr, grads, plan);
//     }
// }

pub struct AdHocTrainingPlan<B: AutodiffBackend, M: ApplyGradients<B> + AutodiffModule<B>> {
    default_optimizer: Optimizer<B, M>,
    plan: Option<M::Plan>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AdHocTrainingPlanConfig<P> {
    pub default_optimizer: OptimizerConfig,
    pub plan: Option<P>,
}

impl<B: AutodiffBackend, F, M: ApplyGradients<B> + AutodiffModule<B>> ApplyGradients<B>
    for AdHocLossModel<M, F>
{
    type Plan = AdHocTrainingPlan<B, M>;
    type PlanConfig = AdHocTrainingPlanConfig<M::PlanConfig>;

    fn config_to_plan(config: Self::PlanConfig) -> Self::Plan {
        AdHocTrainingPlan {
            default_optimizer: config.default_optimizer.init(),
            plan: config.plan.map(|plan| M::config_to_plan(plan)),
        }
    }

    fn apply_gradients(
        &mut self,
        lr: f64,
        mut grads: &mut <B as AutodiffBackend>::Gradients,
        plan: &mut Self::Plan,
    ) {
        if let Some(plan) = &mut plan.plan {
            self.model
                .as_mut()
                .unwrap()
                .apply_gradients(lr, &mut grads, plan);
        }
        let grads = GradientsParams::from_module(grads, self.model.as_ref().unwrap());
        self.model = Some(
            plan.default_optimizer
                .step(lr, self.model.take().unwrap(), grads),
        );
    }
}

default_f!(default_lr_multiplier, f64, 1.0);
