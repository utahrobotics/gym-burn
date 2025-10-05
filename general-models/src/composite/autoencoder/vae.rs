use burn::{module::ModuleDisplay, prelude::*};

use crate::{linear::LinearModel, SimpleInfer, SimpleTrain};

#[derive(Module, Debug)]
pub struct VariationalEncoder<B: Backend, M> {
    model: M,
    mean: LinearModel<B>,
    logvar: LinearModel<B>,
}

impl<B, M, const D: usize> SimpleInfer<B, D, 2> for VariationalEncoder<B, M>
where
    B: Backend,
    M: SimpleInfer<B, D, 2> + ModuleDisplay
{
    fn forward(&self, tensor: Tensor<B, D>) -> Tensor<B, 2> {
        self.mean.infer(self.model.infer(tensor))
    }
}


impl<B: Backend, M> VariationalEncoder<B, M>
where
    B: Backend
{
    pub fn train<const D: usize>(&self, tensor: Tensor<B, D>) -> (Tensor<B, 2>, Tensor<B, 2>)
    where
        M: SimpleInfer<B, D, 2> + ModuleDisplay
    {
        let latent = self.model.infer(tensor);
        (
            self.mean.train(latent.clone()),
            self.logvar.train(latent),
        )
    }
}
