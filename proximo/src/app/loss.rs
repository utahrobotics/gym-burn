use burn::{Tensor, prelude::Backend};

const EPSILON: f64 = 1e-7;

pub fn bce_float_loss<B: Backend, const D: usize>(
    mut expected: Tensor<B, D>,
    mut actual: Tensor<B, D>,
) -> Tensor<B, 1> {
    expected = expected.clamp(EPSILON, 1.0 - EPSILON).detach();
    actual = actual.clamp(EPSILON, 1.0 - EPSILON);
    let loss = expected.clone() * actual.clone().log() + (-expected + 1.0) * (-actual + 1.0).log();
    -loss.mean()
}

// pub fn bce_mse_loss<B: Backend, const D: usize>(
//     expected: Tensor<B, D>,
//     actual: Tensor<B, D>,
//     bce_weight: f64,
//     mse_weight: f64,
// ) -> Tensor<B, 1> {
//     bce_float_loss(expected.clone(), actual.clone()) * bce_weight
//         + MseLoss::new().forward(actual, expected, Reduction::Mean) * mse_weight
// }

// pub fn bce_with_energy_loss<B: Backend, const D: usize>(
//     expected: Tensor<B, D>,
//     actual: Tensor<B, D>,
//     bce_weight: f64,
//     energy_weight: f64,
// ) -> Tensor<B, 1> {
//     bce_float_loss(expected.clone(), actual.clone()) * bce_weight
//         + energy_loss(expected, actual.clamp(0.0, 1.0)) * energy_weight
// }

// pub fn energy_loss<B: Backend, const D: usize>(
//     expected: Tensor<B, D>,
//     actual: Tensor<B, D>,
// ) -> Tensor<B, 1> {
//     let actual = actual.flatten::<2>(1, D - 1).sum_dim(1);
//     let expected = expected.flatten::<2>(1, D - 1).sum_dim(1);
//     (expected - actual).abs().mean()
// }
