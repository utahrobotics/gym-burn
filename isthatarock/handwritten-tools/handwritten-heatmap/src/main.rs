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
use rusqlite::{Connection, params};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    x: f64,
    y: f64,
    z: f64,
    #[arg(short, long)]
    image: PathBuf,
    #[arg(short, long)]
    weights_path: PathBuf,
    #[arg(long, default_value = "model.json")]
    config_path: PathBuf,
    #[arg(long, default_value = "pca.json")]
    pca_path: PathBuf,
    #[arg(short, long)]
    feature_size: usize,
    #[arg(long)]
    elbow_dist: Option<f64>,
    #[arg(long)]
    elbow_brightness: Option<f64>,
}

fn map(from: f32, to: f32, weight: f32) -> f32 {
    (to - from) * weight + from
}

fn main() {
    let args = Args::parse();

    if args.elbow_dist.is_some() != args.elbow_brightness.is_some() {
        println!("elbow_dist and elbow_brightness must both be provided")
    }

    let device = handwritten::wgpu::get_device();

    let mut detector: Detector<WgpuBackend> =
        Detector::load(args.config_path, args.weights_path, device).unwrap();
    detector
        .load_pca(args.pca_path)
        .expect("Expected pca to be readable");

    let dyn_image = image::open(args.image).expect("Expected image to be readable");
    let image = dyn_image.to_luma32f();
    let img_width = image.width() as usize;
    let img_height = image.height() as usize;
    let mut image_tensor = Tensor::<WgpuBackend, 3>::from_data(
        TensorData::new(image.into_vec(), [img_height, img_width, 1]),
        device,
    );
    image_tensor = image_tensor.permute([2, 0, 1]);

    let conn = Connection::open("handwritten.sqlite").unwrap();
    conn.execute("DROP TABLE IF EXISTS distances", ()).unwrap();
    conn.execute("CREATE TABLE distances (row_id INTEGER PRIMARY KEY, px INTEGER NOT NULL, py INTEGER NOT NULL, distance REAL NOT NULL)", ()).unwrap();

    let mut dist_stmt = conn
        .prepare("INSERT OR IGNORE INTO distances (px, py, distance) VALUES (?1, ?2, ?3)")
        .unwrap();

    println!("Encoding");
    let mut encodings = detector.encode_tensor(image_tensor.clone(), [args.feature_size]);
    let feature = encodings.features.pop().unwrap();

    let smaller_img_width = img_width - args.feature_size + 1;
    let smaller_img_height = img_height - args.feature_size + 1;
    let mut distances = Vec::with_capacity(smaller_img_width * smaller_img_height);
    let origin = ndarray::arr1(&[args.x, args.y, args.z]);

    println!("Producing Heatmap");
    for (i, point) in feature
        .latents_pca
        .iter()
        .flat_map(|arr| arr.axis_iter(Axis(0)))
        .enumerate()
    {
        let x = i % smaller_img_width;
        let y = i / smaller_img_width;
        // println!("{:?} {:?}", point.shape(), origin.shape());
        let diff = point.into_owned() - origin.clone();
        let sqr = diff.pow2();
        let dist = sqr.sum().sqrt();
        distances.push(dist);
        dist_stmt.execute(params![x, y, dist]).unwrap();
    }

    let max_distance = *distances.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    let elbow = args.elbow_dist.map(|d| (d, args.elbow_brightness.unwrap()));
    let image = dyn_image.to_rgb32f();

    // let pixels: Vec<_> = distances.into_par_iter()
    //     .enumerate()
    //     .flat_map(|(i, dist)| {
    //         let x = (i % smaller_img_width + args.feature_size / 2) as u32;
    //         let y = (i / smaller_img_width + args.feature_size / 2) as u32;
    //         let mut pixel = image.get_pixel(x, y).0;
    //         let brightness;

    //         if let Some((elbow_dist, elbow_brightness)) = elbow {
    //             if dist <= elbow_dist {
    //                 brightness = dist / elbow_dist * elbow_brightness;
    //             } else {
    //                 brightness = (dist - elbow_dist) / (max_distance - elbow_dist) * elbow_brightness;
    //             }
    //         } else {
    //             brightness = dist / max_distance;
    //         }

    //         pixel[0] = map(pixel[0], 1.0, brightness as f32);
    //         pixel[1] = map(pixel[1], 0.0, brightness as f32);
    //         pixel[2] = map(pixel[2], 0.0, brightness as f32);

    //         pixel
    //     })
    //     .collect();

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

            let dist = distances
                [x - args.feature_size / 2 + (y - args.feature_size / 2) * smaller_img_width];
            let brightness;

            if let Some((_elbow_dist, _elbow_brightness)) = elbow {
                // if dist <= elbow_dist {
                //     brightness = dist / elbow_dist * elbow_brightness;
                // } else {
                //     brightness = (dist - elbow_dist) / (max_distance - elbow_dist) * elbow_brightness;
                // }
                todo!();
            } else {
                brightness = 1.0 - dist / max_distance;
            }

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
    .save("output.webp")
    .unwrap();
}
