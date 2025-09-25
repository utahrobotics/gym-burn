use burn::{
    data::dataloader::batcher::Batcher,
    prelude::Backend,
    tensor::{Tensor, TensorData},
};
use image::ImageFormat;
use serde::{Deserialize, Serialize};

#[derive(Clone, Default)]
pub struct HandwrittenAutoEncoderBatcher {}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct HandwrittenAutoEncoderItem {
    pub image_blob: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct HandwrittenAutoEncoderBatch<B: Backend> {
    pub images: Tensor<B, 3>,
}

impl<B: Backend> Batcher<B, HandwrittenAutoEncoderItem, HandwrittenAutoEncoderBatch<B>>
    for HandwrittenAutoEncoderBatcher
{
    fn batch(
        &self,
        items: Vec<HandwrittenAutoEncoderItem>,
        device: &B::Device,
    ) -> HandwrittenAutoEncoderBatch<B> {
        let images = items
            .iter()
            .map(|item| {
                image::load_from_memory_with_format(&item.image_blob, ImageFormat::WebP).unwrap()
            })
            .map(|image| image.to_luma32f())
            .map(|image| TensorData::from(&*image))
            .map(|data| Tensor::<B, 1>::from_data(data, device))
            .map(|tensor| tensor.reshape([1, 28, 28]))
            // // Normalize: scale between [0,1] and make the mean=0 and std=1
            // // values mean=0.1307,std=0.3081 are from the PyTorch MNIST example
            // // https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/mnist/main.py#L122
            // .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)
            .collect();

        let images = Tensor::cat(images, 0);

        HandwrittenAutoEncoderBatch { images }
    }
}
