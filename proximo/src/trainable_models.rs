use burn::{Tensor, prelude::Backend, tensor::backend::AutodiffBackend};

pub mod apply_gradients;

pub trait ValidatableModel<B: Backend, I> {
    fn batch_valid(&mut self, batch: I) -> Tensor<B, 1>;
}

pub trait TrainableModel<B: AutodiffBackend, I> {
    fn batch_train(&mut self, batch: I) -> Tensor<B, 1>;
}

pub struct AdHocLossModel<M, F> {
    pub model: M,
    pub f: F,
}

impl<F, B, I, M> ValidatableModel<B, I> for AdHocLossModel<M, F>
where
    F: FnMut(&M, I) -> Tensor<B, 1>,
    B: Backend,
{
    fn batch_valid(&mut self, batch: I) -> Tensor<B, 1> {
        (self.f)(&self.model, batch)
    }
}

impl<F, B, I, M> TrainableModel<B, I> for AdHocLossModel<M, F>
where
    F: FnMut(&M, I) -> Tensor<B, 1>,
    B: AutodiffBackend,
{
    fn batch_train(&mut self, batch: I) -> Tensor<B, 1> {
        (self.f)(&self.model, batch)
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
