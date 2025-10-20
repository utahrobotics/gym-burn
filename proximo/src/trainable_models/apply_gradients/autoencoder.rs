use burn::{Tensor, prelude::Backend, tensor::backend::AutodiffBackend};
use general_models::{
    SimpleTrain, composite::autoencoder::{AutoEncoderModel, vae::VariationalEncoderModel}, linear::LinearModel
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

impl<E, D> AutoEncoderModelPlan<E, D> {
    pub fn encoder(&self) -> &E {
        &self.encoder
    }
    
    pub fn decoder(&self) -> &D {
        &self.decoder
    }
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
    model: T,
    mean: LinearModelPlan<B>,
    logvar: LinearModelPlan<B>,
    kld_weight: f64,
}

impl<B: AutodiffBackend, T> VariationalEncoderModelPlan<B, T> {
    pub fn get_kld_weight(&self) -> f64 {
        self.kld_weight
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VariationalEncoderModelPlanConfig<T> {
    pub model: T,
    pub mean: LinearModelPlanConfig,
    pub logvar: LinearModelPlanConfig,
    pub kld_weight: f64,
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
        self.model.apply_gradients(lr, grads, &mut plan.model);
        self.mean.apply_gradients(lr, grads, &mut plan.mean);
        self.logvar
            .apply_gradients(lr, grads, &mut plan.logvar);
    }

    fn config_to_plan(config: Self::PlanConfig) -> Self::Plan {
        VariationalEncoderModelPlan {
            model: T::config_to_plan(config.model),
            mean: LinearModel::config_to_plan(config.mean),
            logvar: LinearModel::config_to_plan(config.logvar),
            kld_weight: config.kld_weight
        }
    }
}

// pub struct VariationalEncoderModelTrainingConfig {
//     pub kld_weight: f64,
// }

// impl<B: Backend, E, D> ValidatableModel<B, AutoEncoderImageBatch<B>, MseLoss, Specialized>
//     for AutoEncoderModel<B, VariationalEncoderModel<B, E>, D>
// where
//     Self: SimpleInfer<B, 4, 4>,
// {
//     fn batch_valid(&self, batch: AutoEncoderImageBatch<B>, loss: &MseLoss) -> Tensor<B, 1> {
//         loss.forward(self.infer(batch.input), batch.expected, Reduction::Auto)
//     }
// }

// impl<B: AutodiffBackend, E, D> TrainableModel<B, AutoEncoderImageBatch<B>, MseLoss, Specialized>
//     for AutoEncoderModel<B, VariationalEncoderModel<B, E>, D>
// where
//     E: SimpleTrain<B, 4, 2>,
//     D: SimpleTrain<B, 2, 4>,
// {
//     type TrainingConfig = VariationalEncoderModelTrainingConfig;

//     fn batch_train(
//         &self,
//         batch: AutoEncoderImageBatch<B>,
//         loss: &MseLoss,
//         config: &Self::TrainingConfig,
//     ) -> Tensor<B, 1> {
//         let (actual_mean, actual_logvar) = self.encoder.train(batch.input.clone());
//         let sampled_latent = self
//             .encoder
//             .reparameterize(actual_mean.clone(), actual_logvar.clone());
//         let actual_reconstructed = self.decoder.train(sampled_latent);
//         let mse_loss = loss.forward(
//             actual_reconstructed.clone(),
//             batch.expected.clone(),
//             Reduction::Mean,
//         );

//         let kld_element = actual_mean
//             .powf_scalar(2.0)
//             .add(actual_logvar.clone().exp())
//             .sub_scalar(1.0)
//             .sub(actual_logvar);
//         let kld = kld_element.sum_dim(1).mean().mul_scalar(0.5);

//         mse_loss + (kld * config.kld_weight)
//     }
// }

pub fn sample_vae<B, E, D, const N_I: usize>(model: &AutoEncoderModel<B, VariationalEncoderModel<B, E>, D>, input: Tensor<B, N_I>) -> (Tensor<B, N_I>, Tensor<B, 1>)
where
    B: Backend,
    E: SimpleTrain<B, N_I, 2>,
    D: SimpleTrain<B, 2, N_I>
{
        let (actual_mean, actual_logvar) = model.encoder.train(input);
        let sampled_latent = model
            .encoder
            .reparameterize(actual_mean.clone(), actual_logvar.clone());
        let actual_reconstructed = model.decoder.train(sampled_latent);

        let kld_element = actual_mean
            .powf_scalar(2.0)
            .add(actual_logvar.clone().exp())
            .sub_scalar(1.0)
            .sub(actual_logvar);
        let kld = kld_element.sum_dim(1).mean().mul_scalar(0.5);

        (actual_reconstructed, kld)
}