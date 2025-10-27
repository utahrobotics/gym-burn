use burn::{Tensor, nn::loss::{MseLoss, Reduction}, prelude::Backend};

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

pub fn mse<B: Backend, const D: usize>(
    expected: Tensor<B, D>,
    actual: Tensor<B, D>,
) -> Tensor<B, 1> {
    MseLoss::new().forward(actual, expected, Reduction::Mean)
}