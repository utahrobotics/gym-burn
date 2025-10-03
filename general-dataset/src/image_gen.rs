use std::{f32::consts::TAU, io::Cursor};

use image::{ImageBuffer, ImageFormat, Pixel, Rgb};
use imageproc::{geometric_transformations::{Interpolation, Projection, rotate_about_center, warp}, noise::gaussian_noise};
use rand::{Rng, random, rng};
use rand_distr::Normal;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub type Image = ImageBuffer<Rgb<u8>, Vec<u8>>;


pub fn flips<P: Pixel<Subpixel = u8> + Send + Sync + 'static>(iter: impl ParallelIterator<Item = ImageBuffer<P, Vec<u8>>>) -> impl ParallelIterator<Item = ImageBuffer<P, Vec<u8>>> {
    iter
        .flat_map(|img| {
            [
                image::imageops::flip_horizontal(&img),
                image::imageops::flip_vertical(&img),
                img
            ]
        })
}


pub fn rotations(iter: impl ParallelIterator<Item = Image>, count: usize, default: Rgb<u8>) -> impl ParallelIterator<Item = Image> {
    iter
        .flat_map(move |src_img| {
            let src_img2= src_img.clone();
            (0..count)
                .into_par_iter()
                .map(move |_| {
                    rotate_about_center(
                        &src_img,
                        rng().random_range(0.0..TAU),
                        Interpolation::Bilinear,
                        default
                    )
                })
                .chain(Some(src_img2))
        })
}


pub fn translations(iter: impl ParallelIterator<Item = Image>, count: usize, std_dev_multiplier: f32, default: Rgb<u8>) -> impl ParallelIterator<Item = Image> {
    iter
        .flat_map(move |src_img| {
            let src_img2= src_img.clone();
            let width = src_img.width();
            let height = src_img.height();
            (0..count)
                .into_par_iter()
                .map(move |_| {
                    let width_distr = Normal::new(0.0, std_dev_multiplier * width as f32).unwrap();
                    let height_distr = Normal::new(0.0, std_dev_multiplier * height as f32).unwrap();
                    let mut rng = rng();
                    warp(
                        &src_img,
                        &Projection::translate(
                            rng.sample(width_distr).clamp(-(width as f32), width as f32),
                            rng.sample(height_distr).clamp(-(height as f32), height as f32)
                        ),
                        Interpolation::Bilinear,
                        default
                    )
                })
                .chain(Some(src_img2))
        })
}


pub fn noises(iter: impl ParallelIterator<Item = Image>, count: usize, std_devs: impl IntoParallelIterator<Item = f64> + Send + Sync + Clone) -> impl ParallelIterator<Item = Image> {
    iter
        .flat_map(move |src_img| {
            let src_img2= src_img.clone();
            let std_devs = std_devs.clone();
            (0..count)
                .into_par_iter()
                .flat_map(move |_| {
                    let src_img = src_img.clone();
                    std_devs
                        .clone()
                        .into_par_iter()
                        .map(move |std_dev| {
                            gaussian_noise(&src_img, 0.0, std_dev, random())
                        })
                })
                .chain(Some(src_img2))
        })
}


pub fn to_webp(iter: impl ParallelIterator<Item = Image>) -> impl ParallelIterator<Item = Vec<u8>> {
    iter
        .map(|img| {
            let mut webp_buf = Cursor::new(vec![]);
            img.write_to(&mut webp_buf, ImageFormat::WebP).unwrap();
            webp_buf.into_inner()
        })
}
