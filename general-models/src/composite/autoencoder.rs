use burn::{module::{Module, ModuleDisplay}, prelude::*};

use crate::{common::PhantomModule, SimpleInfer, SimpleTrain};

pub mod decoder;
pub mod vae;


#[derive(Debug, Module)]
pub struct AutoEncoder<B: Backend, E, D> {
    encoder: E,
    decoder: D,
    _phantom: PhantomModule<B>
}

macro_rules! impls {
    ($input: literal) => {
        impl<B, E, D> SimpleInfer<B, $input, $input> for AutoEncoder<B, E, D>
        where
            B: Backend,
            E: SimpleInfer<B, $input, 2> + ModuleDisplay,
            D: SimpleInfer<B, 2, $input> + ModuleDisplay
        {
            fn forward(&self, tensor: Tensor<B, $input>) -> Tensor<B, $input> {
                self.decoder.infer(self.encoder.forward(tensor))
            }
        }
        
        impl<B, E, D> SimpleTrain<B, $input, $input> for AutoEncoder<B, E, D>
        where
            B: Backend,
            E: SimpleTrain<B, $input, 2> + ModuleDisplay,
            D: SimpleTrain<B, 2, $input> + ModuleDisplay
        {
            fn forward(&self, tensor: Tensor<B, $input>) -> Tensor<B, $input> {
                self.decoder.train(self.encoder.train(tensor))
            }
        }
    }
}

impls!(2);
impls!(3);
impls!(4);
