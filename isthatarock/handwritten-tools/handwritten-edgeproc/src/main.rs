use std::path::PathBuf;

use clap::Parser;
use image::DynamicImage;
use imageproc::distance_transform::Norm;

const EXPANSION: f32 = 1.4 / 28.0;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    image: PathBuf,
    #[arg(short, long)]
    feature_size: usize,
    #[arg(long)]
    low_threshold: f32,
    #[arg(long)]
    high_threshold: f32,
}

fn main() {
    let args = Args::parse();

    let image = image::open(args.image).expect("Expected image to be readable");
    let image = image.into_luma8();

    let mut edges = imageproc::edges::canny(&image, args.low_threshold, args.high_threshold);
    let radius = EXPANSION * args.feature_size as f32;
    imageproc::morphology::dilate_mut(
        &mut edges,
        Norm::L2,
        (radius.round() as u64)
            .try_into()
            .expect("Dilation is too large"),
    );
    edges = imageproc::filter::gaussian_blur_f32(&edges, radius);

    DynamicImage::from(edges).save("output.png").unwrap();
}
