use burn::{Tensor, prelude::Backend};
use general_models::{
    SimpleTrain,
    composite::autoencoder::{AutoEncoderModel, vae::VariationalEncoderModel},
};

pub fn sample_vae<B, E, D, const N_I: usize>(
    model: &AutoEncoderModel<B, VariationalEncoderModel<B, E>, D>,
    input: Tensor<B, N_I>,
) -> (Tensor<B, N_I>, Tensor<B, 1>)
where
    B: Backend,
    E: SimpleTrain<B, N_I, 2>,
    D: SimpleTrain<B, 2, N_I>,
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
