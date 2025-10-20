use burn::{Tensor, module::AutodiffModule, prelude::Backend, tensor::backend::AutodiffBackend};

use crate::trainable_models::apply_gradients::ApplyGradients;

pub mod apply_gradients;

pub trait ValidatableModel<B: Backend, I> {
    fn batch_valid(&mut self, batch: I) -> Tensor<B, 1>;
}

pub trait TrainableModel<B: AutodiffBackend, I>: ApplyGradients<B> {
    fn batch_train(&mut self, batch: I, plan: &Self::Plan) -> Tensor<B, 1>;
}

pub struct AdHocLossModel<M, F = ()> {
    model: Option<M>,
    f: F,
}

impl<M, F> AdHocLossModel<M, F> {
    pub fn new(model: M, f: F) -> Self {
        Self {
            model: Some(model),
            f,
        }
    }

    pub fn unwrap(self) -> M {
        self.model.unwrap()
    }
}

impl<F, B, I, M> ValidatableModel<B, I> for AdHocLossModel<M, F>
where
    F: FnMut(&M, I) -> Tensor<B, 1>,
    B: Backend,
{
    fn batch_valid(&mut self, batch: I) -> Tensor<B, 1> {
        (self.f)(self.model.as_ref().unwrap(), batch)
    }
}

impl<F, B, I, M> TrainableModel<B, I> for AdHocLossModel<M, F>
where
    F: FnMut(&M, I, &Self::Plan) -> Tensor<B, 1>,
    M: ApplyGradients<B> + AutodiffModule<B>,
    B: AutodiffBackend,
{
    fn batch_train(&mut self, batch: I, plan: &Self::Plan) -> Tensor<B, 1> {
        (self.f)(self.model.as_ref().unwrap(), batch, plan)
    }
}
