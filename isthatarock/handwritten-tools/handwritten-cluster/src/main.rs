use std::path::PathBuf;

use clap::Parser;
use handwritten::{
    Detector,
    burn::{
        Tensor,
        nn::interpolate::{Interpolate2dConfig, InterpolateMode},
        tensor::{TensorData, s},
    },
    psnr, psnr_batched,
    wgpu::WgpuBackend,
};
use image::{DynamicImage, ImageBuffer, Luma};
use ndarray::Axis;
use rusqlite::{Connection, params};

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
    #[arg(short, long)]
    feature_size: usize,
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

    let conn = Connection::open("handwritten.sqlite").unwrap();
    conn.execute("DROP TABLE IF EXISTS pca", ()).unwrap();
    conn.execute("CREATE TABLE pca (row_id INTEGER PRIMARY KEY, brightness REAL NOT NULL, p0 REAL NOT NULL, p1 REAL NOT NULL, p2 REAL NOT NULL)", ()).unwrap();

    let mut stmt = conn
        .prepare("INSERT OR IGNORE INTO pca (brightness, p0, p1, p2) VALUES (?1, ?2, ?3, ?4)")
        .unwrap();

    let mut encodings = detector.encode_tensor(original_image_tensor.clone(), [args.feature_size]);
    let mut psnr_sum = 0.0;
    let mut psnr_count = 0.0;
    let mut i = 0usize;

    let mut sum_image = Tensor::<WgpuBackend, 3>::zeros([1, img_height, img_width], device);
    let mut count_image = Tensor::<WgpuBackend, 3>::zeros([1, img_height, img_width], device);
    let feature_ones =
        Tensor::<WgpuBackend, 3>::ones([1, args.feature_size, args.feature_size], device);

    let feature = encodings.features.pop().unwrap();
    let interp = Interpolate2dConfig::new()
        .with_mode(InterpolateMode::Cubic)
        .with_output_size(Some([args.feature_size, args.feature_size]))
        .init();
    let mut j = 0usize;
    let _ = std::fs::create_dir_all("slices");

    for ((latents, batched), latents_pca) in feature
        .latents
        .into_iter()
        .zip(feature.batched)
        .zip(feature.latents_pca)
    {
        let img_width = img_width - args.feature_size + 1;
        let mut decoded = detector.decode_latents(latents);

        for decoded in decoded.clone().iter_dim(0) {
            let decoded = decoded.reshape([1, 28, 28]);
            let image = ImageBuffer::<Luma<f32>, _>::from_vec(
                28,
                28,
                decoded.permute([1, 2, 0]).into_data().into_vec().unwrap(),
            )
            .unwrap();
            DynamicImage::from(image).save(format!("slices/{j}.webp")).unwrap();
            j += 1;
        }

        let psnr_val = psnr_batched(batched.clone(), decoded.clone())
            .mean()
            .into_scalar();
        psnr_sum += psnr_val;
        psnr_count += 1.0;
        decoded = interp.forward(decoded);

        for (point, decoded) in latents_pca.axis_iter(Axis(0)).zip(decoded.iter_dim(0)) {
            let x = i % img_width;
            let y = i / img_width;

            let slice = s![.., y..y + args.feature_size, x..x + args.feature_size];
            let initial = sum_image.clone().slice(slice);
            sum_image = sum_image.slice_assign(
                slice,
                initial + decoded.reshape([1, args.feature_size, args.feature_size]),
            );

            let initial = count_image.clone().slice(slice);
            count_image = count_image.slice_assign(slice, initial + feature_ones.clone());

            let brightness = x as f64 / img_width as f64;
            stmt.execute(params![brightness, point[0], point[1], point[2]])
                .unwrap();
            i += 1;
        }
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
    DynamicImage::from(image).save("output.webp").unwrap();

    println!("Mean element-wise PSNR: {:.2}", psnr_sum / psnr_count);
    assert!(count_image.greater_elem(0.0).all().into_scalar() != 0);
}

// println!("Running Dbscan");

// let clusters = Dbscan::params::<f64>(min_points)
//     .tolerance(tolerance)
//     .transform(latent_points)
//     .unwrap();

// println!("Interpreting Dbscan");

// let mut unknown_count = 0usize;
// let mut sums = FxHashMap::<usize, (usize, Array1<f64>)>::default();

// for (cluster, point) in clusters
//     .axis_iter(Axis(0))
//     .zip(latent_points.axis_iter(Axis(0)))
// {
//     if let Some(cluster_id) = cluster.as_slice().unwrap()[0] {
//         match sums.entry(cluster_id) {
//             Entry::Occupied(mut occupied_entry) => {
//                 occupied_entry.get_mut().0 += 1;
//                 let sum = occupied_entry.get().1.clone() + point;
//                 occupied_entry.get_mut().1 = sum;
//             }
//             Entry::Vacant(vacant_entry) => {
//                 vacant_entry.insert((1, point.to_owned()));
//             }
//         }
//     } else {
//         unknown_count += 1;
//     }
// }

// println!("Writing results");

// {
//     let mut cluster_size_file =
//         BufWriter::new(File::create("cluster_size.csv").unwrap());
//     writeln!(cluster_size_file, "cluster id, count").unwrap();
//     for (cluster_id, (count, _)) in &sums {
//         writeln!(cluster_size_file, "{cluster_id},{count}").unwrap();
//     }
//     writeln!(cluster_size_file, "unknown,{}", unknown_count).unwrap();
// }

// {
//     let mut centroids_file = BufWriter::new(File::create("centroids.csv").unwrap());
//     writeln!(centroids_file, "cluster id,").unwrap();
//     for (cluster_id, (count, sum)) in &sums {
//         let mean = sum / *count as f64;
//         write!(centroids_file, "{cluster_id}").unwrap();
//         for c in mean {
//             write!(centroids_file, ",{c:>8.4}").unwrap();
//         }
//         writeln!(centroids_file).unwrap();
//     }
// }

// {
//     let mut ids_file = BufWriter::new(File::create("ids.csv").unwrap());
//     writeln!(ids_file, "i,cluster id").unwrap();
//     for (i, (cluster, point)) in clusters
//         .axis_iter(Axis(0))
//         .zip(latent_points.axis_iter(Axis(0)))
//         .enumerate()
//     {
//         if let Some(cluster_id) = cluster.as_slice().unwrap()[0] {
//             write!(ids_file, "{i},{cluster_id}").unwrap();
//         } else {
//             write!(ids_file, "{i},-1").unwrap();
//         }
//         for c in point {
//             write!(ids_file, ",{c:>8.4}").unwrap();
//         }
//         writeln!(ids_file).unwrap();
//     }
// }
