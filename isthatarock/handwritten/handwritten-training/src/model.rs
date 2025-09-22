use burn::{
    module::Module,
    prelude::Backend,
    tensor::backend::AutodiffBackend,
    train::{RegressionOutput, TrainOutput, TrainStep, ValidStep},
};
use handwritten_model::HandwrittenAutoEncoder;

use crate::data::HandwrittenAutoEncoderBatch;

#[derive(Module, Debug)]
pub struct TrainableHandwrittenAutoEncoder<B: Backend> {
    pub model: HandwrittenAutoEncoder<B>,
}

impl<B: AutodiffBackend> TrainStep<HandwrittenAutoEncoderBatch<B>, RegressionOutput<B>>
    for TrainableHandwrittenAutoEncoder<B>
{
    fn step(&self, batch: HandwrittenAutoEncoderBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.model.forward_regression(batch.images);

        TrainOutput::new(&self.model, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<HandwrittenAutoEncoderBatch<B>, RegressionOutput<B>>
    for TrainableHandwrittenAutoEncoder<B>
{
    fn step(&self, batch: HandwrittenAutoEncoderBatch<B>) -> RegressionOutput<B> {
        self.model.forward_regression(batch.images)
    }
}
