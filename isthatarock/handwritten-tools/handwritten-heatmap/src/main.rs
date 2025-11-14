use std::path::PathBuf;

use clap::Parser;
use handwritten::{
    Detector,
    burn::{Tensor, tensor::TensorData},
    wgpu::WgpuBackend,
};
use image::{DynamicImage, ImageBuffer, Rgb};
use ndarray::Axis;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    image: PathBuf,
    #[arg(short, long)]
    weights_path: PathBuf,
    #[arg(long, default_value = "model.json")]
    config_path: PathBuf,
    #[arg(long, default_value = "pca.json")]
    pca_path: PathBuf,
    #[arg(long, default_value = "targets.json")]
    targets_path: PathBuf,
    #[arg(short, long)]
    feature_size: usize,
}

fn map(from: f32, to: f32, weight: f32) -> f32 {
    (to - from) * weight + from
}

fn main() {
    let args = Args::parse();

    let device = handwritten::wgpu::get_device();

    let mut detector: Detector<WgpuBackend> =
        Detector::load(args.config_path, args.weights_path, device).unwrap();
    detector
        .load_pca(args.pca_path)
        .expect("Expected pca to be readable");
    detector
        .load_target_encodings(args.targets_path)
        .expect("Expected targets to be readable");

    let dyn_image = image::open(args.image).expect("Expected image to be readable");
    let image = dyn_image.to_luma32f();
    let img_width = image.width() as usize;
    let img_height = image.height() as usize;
    let mut image_tensor = Tensor::<WgpuBackend, 3>::from_data(
        TensorData::new(image.into_vec(), [img_height, img_width, 1]),
        device,
    );
    image_tensor = image_tensor.permute([2, 0, 1]);

    // let conn = Connection::open("handwritten.sqlite").unwrap();
    // conn.execute("DROP TABLE IF EXISTS distances", ()).unwrap();
    // conn.execute("CREATE TABLE distances (row_id INTEGER PRIMARY KEY, px INTEGER NOT NULL, py INTEGER NOT NULL, distance REAL NOT NULL)", ()).unwrap();

    // let mut dist_stmt = conn
    //     .prepare("INSERT OR IGNORE INTO distances (px, py, distance) VALUES (?1, ?2, ?3)")
    //     .unwrap();

    println!("Encoding");
    let mut encodings = detector.encode_tensor(image_tensor.clone(), [args.feature_size]);
    let feature = encodings.features.pop().unwrap();

    let smaller_img_width = img_width - args.feature_size + 1;
    let smaller_img_height = img_height - args.feature_size + 1;
    let mut scores = Vec::with_capacity(smaller_img_width * smaller_img_height);
    let mut output = vec![];

    println!("Producing Heatmap");
    for point in feature
        .latents_pca
        .iter()
        .flat_map(|arr| arr.axis_iter(Axis(0)))
    {
        detector.score_encoding_into(point.into_owned(), &mut output);
        // let max_score = *output.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
        // scores.push(output[0] / max_score);
        scores.push(output[0]);
        output.clear();
    }

    let image = dyn_image.to_rgb32f();

    let pixels: Vec<_> = (0..img_height)
        .into_par_iter()
        .flat_map(|y| (0..img_width).into_par_iter().map(move |x| (x, y)))
        .flat_map(|(x, y)| {
            let mut pixel = image.get_pixel(x as u32, y as u32).0;

            if x < args.feature_size / 2 || x >= img_width - args.feature_size / 2 {
                return pixel;
            } else if y < args.feature_size / 2 || y >= img_height - args.feature_size / 2 {
                return pixel;
            }

            let brightness =
                scores[x - args.feature_size / 2 + (y - args.feature_size / 2) * smaller_img_width];

            pixel[0] = map(pixel[0], 1.0, brightness as f32);
            pixel[1] = map(pixel[1], 0.0, brightness as f32);
            pixel[2] = map(pixel[2], 0.0, brightness as f32);

            pixel
        })
        .collect();

    DynamicImage::from(
        ImageBuffer::<Rgb<f32>, Vec<_>>::from_vec(img_width as u32, img_height as u32, pixels)
            .unwrap(),
    )
    .into_rgba8()
    .save("output.png")
    .unwrap();
}
