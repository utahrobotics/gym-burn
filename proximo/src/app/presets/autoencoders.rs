use burn::{module::Module, prelude::Backend, tensor::backend::AutodiffBackend};
use general_models::{Init, SimpleInfer, composite::{autoencoder::{AutoEncoderModel, AutoEncoderModelConfig, vae::{VariationalEncoderModel, VariationalEncoderModelConfig}}, image::{Conv2dLinearModel, Conv2dLinearModelConfig, LinearConvTranspose2dModel, LinearConvTranspose2dModelConfig}}};
use serde::{Deserialize, Serialize};

use crate::trainable_models::apply_gradients::{ApplyGradients, autoencoder::{AutoEncoderModelPlan, AutoEncoderModelPlanConfig, VariationalEncoderModelPlan, VariationalEncoderModelPlanConfig}, image::{Conv2dLinearModelPlan, Conv2dLinearModelPlanConfig, LinearConvTranspose2dModelPlan, LinearConvTranspose2dModelPlanConfig}};

#[derive(Module, Debug)]
pub enum ImageAutoEncoder<B: Backend> {
    Normal(AutoEncoderModel<
        B,
        Conv2dLinearModel<B>,
        LinearConvTranspose2dModel<B>,
    >),
    Vae(AutoEncoderModel<
        B,
        VariationalEncoderModel<B, Conv2dLinearModel<B>>,
        LinearConvTranspose2dModel<B>,
    >)
}

impl<B: Backend> ImageAutoEncoder<B> {
    pub fn get_input_channels(&self) -> usize {
        match self {
            ImageAutoEncoder::Normal(x) => x.encoder.get_input_channels(),
            ImageAutoEncoder::Vae(x) => x.encoder.model.get_input_channels(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(untagged)]
pub enum ImageAutoEncoderConfig {
    Normal(AutoEncoderModelConfig<
        Conv2dLinearModelConfig,
        LinearConvTranspose2dModelConfig
    >),
    Vae(AutoEncoderModelConfig<
        VariationalEncoderModelConfig<Conv2dLinearModelConfig>,
        LinearConvTranspose2dModelConfig
    >),
}


impl<B: Backend> Init<B, ImageAutoEncoder<B>> for ImageAutoEncoderConfig {
    fn init(self, device: &<B as Backend>::Device) -> ImageAutoEncoder<B> {
        match self {
            ImageAutoEncoderConfig::Normal(x) => ImageAutoEncoder::Normal(x.init(device)),
            ImageAutoEncoderConfig::Vae(x) => ImageAutoEncoder::Vae(x.init(device)),
        }
    }
}

pub enum ImageAutoEncoderPlan<B: AutodiffBackend> {
    Normal(AutoEncoderModelPlan<Conv2dLinearModelPlan<B>, LinearConvTranspose2dModelPlan<B>>),
    Vae(AutoEncoderModelPlan<VariationalEncoderModelPlan<B, Conv2dLinearModelPlan<B>>, LinearConvTranspose2dModelPlan<B>>)
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(untagged)]
pub enum ImageAutoEncoderPlanConfig {
    Normal(AutoEncoderModelPlanConfig<Conv2dLinearModelPlanConfig, LinearConvTranspose2dModelPlanConfig>),
    Vae(AutoEncoderModelPlanConfig<VariationalEncoderModelPlanConfig<Conv2dLinearModelPlanConfig>, LinearConvTranspose2dModelPlanConfig>)
}

impl<B: AutodiffBackend> ApplyGradients<B> for ImageAutoEncoder<B> {
    type Plan = ImageAutoEncoderPlan<B>;

    type PlanConfig = ImageAutoEncoderPlanConfig;

    fn config_to_plan(config: Self::PlanConfig) -> Self::Plan {
        match config {
            ImageAutoEncoderPlanConfig::Normal(x) => ImageAutoEncoderPlan::Normal(AutoEncoderModel::<B, Conv2dLinearModel<B>, LinearConvTranspose2dModel<B>>::config_to_plan(x)),
            ImageAutoEncoderPlanConfig::Vae(x) => ImageAutoEncoderPlan::Vae(AutoEncoderModel::<B, VariationalEncoderModel<B, Conv2dLinearModel<B>>, LinearConvTranspose2dModel<B>>::config_to_plan(x)),
        }
    }

    fn apply_gradients(
        &mut self,
        lr: f64,
        grads: &mut <B as burn::tensor::backend::AutodiffBackend>::Gradients,
        plan: &mut Self::Plan,
    ) {
        match self {
            ImageAutoEncoder::Normal(x) => x.apply_gradients(lr, grads, {
                let ImageAutoEncoderPlan::Normal(plan) = plan else { panic!("Incorrect model plan"); };
                plan
            }),
            ImageAutoEncoder::Vae(x) => x.apply_gradients(lr, grads, {
                let ImageAutoEncoderPlan::Vae(plan) = plan else { panic!("Incorrect model plan"); };
                plan
            }),
        }
    }
}

// impl<B: Backend> SimpleTrain<B, 4, 4> for ImageAutoEncoder<B> {
//     fn forward(&self, tensor: burn::Tensor<B, 4>) -> burn::Tensor<B, 4> {
//         match self {
//             ImageAutoEncoder::Normal(x) => x.train(tensor),
//             ImageAutoEncoder::Vae(x) => x.train(tensor)
//         }
//     }
// }

impl<B: Backend> SimpleInfer<B, 4, 4> for ImageAutoEncoder<B> {
    fn forward(&self, tensor: burn::Tensor<B, 4>) -> burn::Tensor<B, 4> {
        match self {
            ImageAutoEncoder::Normal(x) => x.infer(tensor),
            ImageAutoEncoder::Vae(x) => x.infer(tensor)
        }
    }
}