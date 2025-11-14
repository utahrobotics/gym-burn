use std::{collections::hash_map::Entry, path::PathBuf};

use clap::Parser;
use handwritten::{
    Detector,
    burn::{
        Tensor,
        nn::interpolate::{Interpolate2dConfig, InterpolateMode},
        tensor::{TensorData, s},
    },
    psnr,
    wgpu::WgpuBackend,
};
use image::{DynamicImage, ImageBuffer, Luma, imageops::FilterType};
use ndarray::Axis;
use rusqlite::{Connection, params};
use rustc_hash::FxHashMap;

use crate::cluster::cluster;

mod cluster;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    image: PathBuf,
    #[arg(short, long)]
    mask_image: Option<PathBuf>,
    #[arg(short, long)]
    weights_path: PathBuf,
    #[arg(long, default_value = "model.json")]
    config_path: PathBuf,
    #[arg(long, default_value = "pca.json")]
    pca_path: PathBuf,
    #[arg(short, long)]
    feature_size: usize,
    #[arg(long, default_value = "5")]
    min_points: usize,
}

fn main() {
    let args = Args::parse();

    let device = handwritten::wgpu::get_device();

    let mut detector: Detector<WgpuBackend> =
        Detector::load(args.config_path, args.weights_path, device).unwrap();
    detector
        .load_pca(args.pca_path)
        .expect("Expected pca to be readable");

    let image = image::open(args.image).expect("Expected image to be readable");
    let image = image.to_luma32f();
    let img_width = image.width() as usize;
    let img_height = image.height() as usize;
    let mut original_image_tensor = Tensor::<WgpuBackend, 3>::from_data(
        TensorData::new(image.into_vec(), [img_height, img_width, 1]),
        device,
    );
    original_image_tensor = original_image_tensor.permute([2, 0, 1]);

    let mask_image = args.mask_image.map(|img| {
        let mut img = image::open(img).expect("Expected mask image to be readable");
        img = img.resize_exact(
            (img_width - args.feature_size + 1) as u32,
            (img_height - args.feature_size + 1) as u32,
            FilterType::Nearest,
        );
        img.into_rgb8()
    });

    let conn = Connection::open("handwritten.sqlite").unwrap();
    conn.execute("DROP TABLE IF EXISTS pca", ()).unwrap();
    conn.execute("CREATE TABLE pca (row_id INTEGER PRIMARY KEY, brightness REAL NOT NULL, p0 REAL NOT NULL, p1 REAL NOT NULL, p2 REAL NOT NULL)", ()).unwrap();

    conn.execute("DROP TABLE IF EXISTS colored", ()).unwrap();
    conn.execute("CREATE TABLE colored (row_id INTEGER PRIMARY KEY, r INTEGER NOT NULL, g INTEGER NOT NULL, b INTEGER NOT NULL, x REAL NOT NULL, y REAL NOT NULL, z REAL NOT NULL)", ()).unwrap();

    conn.execute("DROP TABLE IF EXISTS clusters", ()).unwrap();
    conn.execute("CREATE TABLE clusters (row_id INTEGER PRIMARY KEY, r INTEGER NOT NULL, g INTEGER NOT NULL, b INTEGER NOT NULL, x REAL NOT NULL, y REAL NOT NULL, z REAL NOT NULL, size INTEGER NOT NULL)", ()).unwrap();

    let mut pca_stmt = conn
        .prepare("INSERT OR IGNORE INTO pca (brightness, p0, p1, p2) VALUES (?1, ?2, ?3, ?4)")
        .unwrap();
    let mut colored_stmt = conn
        .prepare("INSERT OR IGNORE INTO colored (r, g, b, x, y, z) VALUES (?1, ?2, ?3, ?4, ?5, ?6)")
        .unwrap();
    let mut clusters_stmt = conn
        .prepare("INSERT OR IGNORE INTO clusters (r, g, b, x, y, z, size) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)")
        .unwrap();

    let mut clustering_map: FxHashMap<[u8; 3], Vec<[f64; 3]>> = FxHashMap::default();

    println!("Encoding");
    let mut encodings = detector.encode_tensor(original_image_tensor.clone(), [args.feature_size]);
    // let mut psnr_sum = 0.0;
    // let mut psnr_count = 0.0;
    let mut i = 0usize;

    let mut sum_image = Tensor::<WgpuBackend, 3>::zeros([1, img_height, img_width], device);
    let mut count_image = Tensor::<WgpuBackend, 3>::zeros([1, img_height, img_width], device);
    let feature_ones =
        Tensor::<WgpuBackend, 3>::ones([1, args.feature_size, args.feature_size], device);

    let feature = encodings.features.pop().unwrap();
    let interp = Interpolate2dConfig::new()
        .with_mode(InterpolateMode::Linear)
        .with_output_size(Some([args.feature_size, args.feature_size]))
        .init();

    // let mut j = 0usize;
    // let _ = std::fs::create_dir_all("slices");

    println!("Decoding");
    let smaller_img_width = img_width - args.feature_size + 1;
    // for ((latents, batched), latents_pca) in feature
    //     .latents
    //     .into_iter()
    //     .zip(feature.batched)
    //     .zip(feature.latents_pca)
    for (latents, latents_pca) in feature.latents.into_iter().zip(feature.latents_pca) {
        let mut decoded = detector.decode_latents(latents);

        // for decoded in decoded.clone().iter_dim(0) {
        //     let decoded = decoded.reshape([1, 28, 28]);
        //     let image = ImageBuffer::<Luma<f32>, _>::from_vec(
        //         28,
        //         28,
        //         decoded.permute([1, 2, 0]).into_data().into_vec().unwrap(),
        //     )
        //     .unwrap();
        //     DynamicImage::from(image).save(format!("slices/{j}.png")).unwrap();
        //     j += 1;
        // }

        // let psnr_val = psnr_batched(batched.clone(), decoded.clone())
        //     .mean()
        //     .into_scalar();
        // psnr_sum += psnr_val;
        // psnr_count += 1.0;
        let brightnesses = decoded
            .clone()
            .mean_dims(&[1, 2, 3])
            .into_data()
            .into_vec::<f32>()
            .unwrap();
        decoded = interp.forward(decoded);

        for ((point, decoded), brightness) in latents_pca
            .axis_iter(Axis(0))
            .zip(decoded.iter_dim(0))
            .zip(brightnesses)
        {
            let x = i % smaller_img_width;
            let y = i / smaller_img_width;

            if let Some(mask_image) = &mask_image {
                let pixel = mask_image.get_pixel(x as u32, y as u32).0;

                colored_stmt
                    .execute(params![
                        pixel[0], pixel[1], pixel[2], point[0], point[1], point[2]
                    ])
                    .unwrap();

                let point = [point[0], point[1], point[2]];
                match clustering_map.entry(pixel) {
                    Entry::Occupied(mut entry) => {
                        entry.get_mut().push(point);
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(vec![point]);
                    }
                }
            }

            let slice = s![.., y..y + args.feature_size, x..x + args.feature_size];
            let initial = sum_image.clone().slice(slice);
            sum_image = sum_image.slice_assign(
                slice,
                initial + decoded.reshape([1, args.feature_size, args.feature_size]),
            );

            let initial = count_image.clone().slice(slice);
            count_image = count_image.slice_assign(slice, initial + feature_ones.clone());

            // let brightness = x as f64 / img_width as f64;
            pca_stmt
                .execute(params![brightness, point[0], point[1], point[2]])
                .unwrap();
            i += 1;
        }
    }

    if mask_image.is_some() {
        println!("Clustering");

        let mut total_unknown = 0usize;
        clustering_map
            .into_iter()
            .filter(|(color, points)| {
                if points.len() == 1 {
                    // ignore. Probably just noise
                    false
                } else if points.len() < args.min_points + 1 {
                    println!("Color {color:?} has only {} points", points.len());
                    false
                } else {
                    true
                }
            })
            .map(|(color, points)| (color, cluster(points, args.min_points)))
            .for_each(|(color, result)| {
                println!("{color:?} Unknowns: {}", result.unknowns);

                for ([x, y, z], size) in result.centers {
                    let [r, g, b] = color;
                    clusters_stmt
                        .execute(params![r, g, b, x, y, z, size])
                        .unwrap();
                }

                total_unknown += result.unknowns;
            });

        println!("Total Unknowns: {total_unknown}");
        println!("Total Points: {i}")
    }

    let decoded_image_tensor = sum_image / count_image.clone();
    let psnr = psnr(decoded_image_tensor.clone(), original_image_tensor);
    println!("PSNR: {:.2}", psnr);
    let image = ImageBuffer::<Luma<f32>, _>::from_vec(
        img_width as u32,
        img_height as u32,
        decoded_image_tensor
            .permute([1, 2, 0])
            .into_data()
            .into_vec()
            .unwrap(),
    )
    .unwrap();
    DynamicImage::from(image).into_rgba8().save("output.png").unwrap();

    // println!("Mean element-wise PSNR: {:.2}", psnr_sum / psnr_count);
    assert!(count_image.greater_elem(0.0).all().into_scalar() != 0);
}
