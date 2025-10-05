use burn::{
    module::{Module, ModuleDisplay},
    prelude::*,
};
use serde::{Deserialize, Serialize};

use crate::{Init, SimpleInfer, SimpleTrain, common::PhantomModule};

pub mod decoder;
pub mod vae;

#[derive(Debug, Module)]
pub struct AutoEncoderModel<B: Backend, E, D> {
    pub encoder: E,
    pub decoder: D,
    _phantom: PhantomModule<B>,
}

impl<B: Backend, E, D> AutoEncoderModel<B, E, D> {
    pub fn new(encoder: E, decoder: D) -> Self {
        Self {
            encoder,
            decoder,
            _phantom: Default::default(),
        }
    }
}

macro_rules! impls {
    ($input: literal) => {
        impl<B, E, D> SimpleInfer<B, $input, $input> for AutoEncoderModel<B, E, D>
        where
            B: Backend,
            E: SimpleInfer<B, $input, 2> + ModuleDisplay,
            D: SimpleInfer<B, 2, $input> + ModuleDisplay,
        {
            fn forward(&self, tensor: Tensor<B, $input>) -> Tensor<B, $input> {
                self.decoder.infer(self.encoder.forward(tensor))
            }
        }

        impl<B, E, D> SimpleTrain<B, $input, $input> for AutoEncoderModel<B, E, D>
        where
            B: Backend,
            E: SimpleTrain<B, $input, 2> + ModuleDisplay,
            D: SimpleTrain<B, 2, $input> + ModuleDisplay,
        {
            fn forward(&self, tensor: Tensor<B, $input>) -> Tensor<B, $input> {
                self.decoder.train(self.encoder.train(tensor))
            }
        }
    };
}

impls!(2);
impls!(3);
impls!(4);

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct AutoEncoderModelConfig<E, D> {
    pub encoder: E,
    pub decoder: D,
}

impl<B, E, D> Init<B> for AutoEncoderModelConfig<E, D>
where
    B: Backend,
    E: Init<B>,
    D: Init<B>,
{
    type Output = AutoEncoderModel<B, E::Output, D::Output>;

    fn init(self, device: &<B as Backend>::Device) -> Self::Output {
        AutoEncoderModel {
            encoder: self.encoder.init(device),
            decoder: self.decoder.init(device),
            _phantom: Default::default(),
        }
    }
}
