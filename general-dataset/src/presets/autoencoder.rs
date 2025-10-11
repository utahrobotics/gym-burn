use burn::prelude::*;
use image::{ImageFormat, load_from_memory_with_format};
use rand::seq::SliceRandom;
use rayon::join;

use crate::{StatefulBatcher, sql_object};

sql_object!(
    pub struct AutoEncoderImageItem {
        pub webp_input: Vec<u8>,
        pub input_width: usize,
        pub input_height: usize,
        pub webp_expected: Vec<u8>,
        pub expected_width: usize,
        pub expected_height: usize,
    }
);

pub struct AutoEncoderImageBatch<B: Backend> {
    pub input: Tensor<B, 4>,
    pub expected: Tensor<B, 4>,
}

// impl<B: Backend> Clone for AutoEncoderImageBatch<B> {
//     fn clone(&self) -> Self {
//         Self {
//             input: self.input.clone(),
//             expected: self.expected.clone(),
//         }
//     }
// }

#[derive(Debug)]
pub struct AutoEncoderImageBatcher<B: Backend> {
    channels: usize,
    input_tensors: Vec<Tensor<B, 4>>,
    expected_tensors: Vec<Tensor<B, 4>>,
    device: B::Device,
}

impl<B: Backend> AutoEncoderImageBatcher<B> {
    pub fn new(channels: usize, device: B::Device) -> Self {
        Self {
            channels,
            input_tensors: vec![],
            expected_tensors: vec![],
            device,
        }
    }
}

impl<B: Backend> StatefulBatcher<AutoEncoderImageItem, AutoEncoderImageBatch<B>>
    for AutoEncoderImageBatcher<B>
{
    fn reset(&mut self) {
        self.input_tensors.clear();
        self.expected_tensors.clear();
    }

    fn ingest(&mut self, item: AutoEncoderImageItem) {
        macro_rules! process {
            ($webp: ident) => {{
                let img = load_from_memory_with_format(&item.$webp, ImageFormat::WebP).unwrap();
                let data = match self.channels {
                    1 => img.to_luma32f().into_vec(),
                    2 => img.to_luma_alpha32f().into_vec(),
                    3 => img.into_rgb32f().into_vec(),
                    4 => img.into_rgba32f().into_vec(),
                    _ => unreachable!(),
                };
                // assert!(data.iter().all(|x| *x <= 1.0), "{:?}", data);
                // assert!(data.iter().all(|x| *x >= 0.0), "{:?}", data);
                Tensor::<B, 1>::from_data(data.as_slice(), &self.device).reshape([
                    1,
                    self.channels,
                    item.input_width,
                    item.input_height,
                ])
            }};
        }
        join(
            || {
                let tensor = process!(webp_input);
                self.input_tensors.push(tensor);
            },
            || {
                let tensor = process!(webp_expected);
                self.expected_tensors.push(tensor);
            },
        );
    }

    fn shuffle(&mut self, rng: &mut impl rand::Rng) {
        self.input_tensors.shuffle(rng);
        self.expected_tensors.shuffle(rng);
    }

    fn finish(&mut self) -> AutoEncoderImageBatch<B> {
        let input_replace = Vec::with_capacity(self.input_tensors.len());
        let expected_replace = Vec::with_capacity(self.expected_tensors.len());

        AutoEncoderImageBatch {
            input: Tensor::cat(std::mem::replace(&mut self.input_tensors, input_replace), 0),
            expected: Tensor::cat(
                std::mem::replace(&mut self.expected_tensors, expected_replace),
                0,
            ),
        }
    }
    // fn batch(&self, items: Vec<I>, device: &<B as Backend>::Device) -> AutoEncoderImageBatch<B> {
    //     let slices_to_data = |webp_data: Option<&[u8]>, luma_data: Option<&[f32]>| {
    //         if let Some(luma_data) = luma_data {
    //             TensorData::from(luma_data)
    //         } else {
    //             let webp_data = webp_data.expect("Either webp or luma data should exist");
    //             let luma_data = load_from_memory_with_format(webp_data, ImageFormat::WebP)
    //                 .unwrap()
    //                 .to_luma32f()
    //                 .into_vec();
    //             TensorData::from(&*luma_data)
    //         }
    //     };
    //     let input = items
    //         .iter()
    //         .map(AsRef::as_ref)
    //         .map(|item| {
    //             (
    //                 slices_to_data(item.webp_input.as_deref(), item.luma_input.as_deref()),
    //                 item,
    //             )
    //         })
    //         .map(|(data, item)| (Tensor::<B, 1>::from_data(data, device), item))
    //         .map(|(tensor, item)| {
    //             tensor.reshape([1, self.channels, item.input_width, item.input_height])
    //         })
    //         .collect();

    //     let input = Tensor::cat(input, 0);

    //     let expected = items
    //         .iter()
    //         .map(AsRef::as_ref)
    //         .map(|item| {
    //             (
    //                 slices_to_data(item.webp_expected.as_deref(), item.luma_expected.as_deref()),
    //                 item,
    //             )
    //         })
    //         .map(|(data, item)| (Tensor::<B, 1>::from_data(data, device), item))
    //         .map(|(tensor, item)| {
    //             tensor.reshape([1, self.channels, item.expected_width, item.expected_height])
    //         })
    //         .collect();

    //     let expected = Tensor::cat(expected, 0);

    //     AutoEncoderImageBatch { input, expected }
    // }
}
