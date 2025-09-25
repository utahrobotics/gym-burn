use burn::{Tensor, prelude::Backend};

pub struct AutoEncoderImageBatch<B: Backend> {
    pub input: Tensor<B, 3>,
    pub expected: Tensor<B, 3>,
}
