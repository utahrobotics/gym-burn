use std::fmt::Debug;

use burn::{Tensor, data::dataloader::batcher::Batcher, prelude::Backend, tensor::TensorData};
use image::{ImageFormat, load_from_memory_with_format};
use serde::{Deserialize, Serialize};

pub struct AutoEncoderImageBatch<B: Backend> {
    pub input: Tensor<B, 4>,
    pub expected: Tensor<B, 4>,
}

impl<B: Backend> Clone for AutoEncoderImageBatch<B> {
    fn clone(&self) -> Self {
        Self {
            input: self.input.clone(),
            expected: self.expected.clone(),
        }
    }
}

impl<B: Backend> Debug for AutoEncoderImageBatch<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AutoEncoderImageBatch")
            .field("input", &self.input)
            .field("expected", &self.expected)
            .finish()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AutoEncoderImageItem {
    pub luma_input: Option<Vec<f32>>,
    pub webp_input: Option<Vec<u8>>,
    pub input_width: usize,
    pub input_height: usize,
    pub luma_expected: Option<Vec<f32>>,
    pub webp_expected: Option<Vec<u8>>,
    pub expected_width: usize,
    pub expected_height: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct AutoEncoderImageBatcher {
    pub channels: usize
}

impl<B: Backend, I: AsRef<AutoEncoderImageItem>> Batcher<B, I, AutoEncoderImageBatch<B>>
    for AutoEncoderImageBatcher
{
    fn batch(&self, items: Vec<I>, device: &<B as Backend>::Device) -> AutoEncoderImageBatch<B> {
        let slices_to_data = |webp_data: Option<&[u8]>, luma_data: Option<&[f32]>| {
            if let Some(luma_data) = luma_data {
                TensorData::from(luma_data)
            } else {
                let webp_data = webp_data.expect("Either webp or luma data should exist");
                let luma_data = load_from_memory_with_format(webp_data, ImageFormat::WebP)
                    .unwrap()
                    .to_luma32f()
                    .into_vec();
                TensorData::from(&*luma_data)
            }
        };
        let input = items
            .iter()
            .map(AsRef::as_ref)
            .map(|item| {
                (
                    slices_to_data(item.webp_input.as_deref(), item.luma_input.as_deref()),
                    item,
                )
            })
            .map(|(data, item)| (Tensor::<B, 1>::from_data(data, device), item))
            .map(|(tensor, item)| tensor.reshape([1, self.channels, item.input_width, item.input_height]))
            .collect();

        let input = Tensor::cat(input, 0);

        let expected = items
            .iter()
            .map(AsRef::as_ref)
            .map(|item| {
                (
                    slices_to_data(item.webp_expected.as_deref(), item.luma_expected.as_deref()),
                    item,
                )
            })
            .map(|(data, item)| (Tensor::<B, 1>::from_data(data, device), item))
            .map(|(tensor, item)| tensor.reshape([1, self.channels, item.expected_width, item.expected_height]))
            .collect();

        let expected = Tensor::cat(expected, 0);

        AutoEncoderImageBatch { input, expected }
    }
}
