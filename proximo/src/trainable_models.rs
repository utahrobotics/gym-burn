use burn::{Tensor, nn::loss::{MseLoss, Reduction}, prelude::Backend};
use general_dataset::presets::autoencoder::AutoEncoderImageBatch;
use general_models::{SimpleTrain, composite::autoencoder::AutoEncoderModel};
use serde::de::DeserializeOwned;

pub mod apply_gradients;


pub trait TrainableModel<B: Backend, I> {
    type Loss;
    type LossConfig: DeserializeOwned;

    fn batch_train(&self, batch: I, loss: &Self::Loss) -> Tensor<B, 1>;
}

impl<B: Backend, E, D> TrainableModel<B, AutoEncoderImageBatch<B>> for AutoEncoderModel<B, E, D>
where 
    Self: SimpleTrain<B, 4, 4>
{
    type Loss = MseLoss;
    type LossConfig = ();

    fn batch_train(&self, batch: AutoEncoderImageBatch<B>, loss: &Self::Loss) -> Tensor<B, 1> {
        loss.forward(
            self.train(batch.input),
            batch.expected,
            Reduction::Auto
        )
    }
}
