use std::path::PathBuf;

use clap::Parser;
use handwritten::{Detector, burn::{Tensor, tensor::TensorData}, psnr, psnr_batched, wgpu::WgpuBackend};
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
    feature_size: Vec<usize>,
}

fn main() {
    let args = Args::parse();

    let device = handwritten::wgpu::get_device();

    let mut detector: Detector<WgpuBackend> = Detector::load(args.config_path, args.weights_path, device).unwrap();
    detector.load_pca(args.pca_path).expect("Expected pca to be readable");

    let image = image::open(args.image).expect("Expected image to be readable");
    let image = image.to_luma32f();
    let img_width = image.width() as usize;
    let img_height = image.height() as usize;
    let mut image_tensor = Tensor::<WgpuBackend, 3>::from_data(TensorData::new(image.into_vec(), [img_width, img_height, 1]), device);
    image_tensor = image_tensor.permute([2, 0, 1]);

    let encodings = detector.encode_tensor(image_tensor, args.feature_size);
    let decoded = detector.decode_latents(encodings.latents);

    let batch_psnr = psnr_batched(encodings.batched.clone(), decoded.clone()).into_data().to_vec::<f32>().unwrap();

    for (i, original) in encodings.batched.iter_dim(0).enumerate() {
        let decoded = decoded.clone().slice_dim(0, i..=i);
        let psnr = psnr(original, decoded);
        println!("{psnr:.2} {:.2}", batch_psnr[i]);
    }

    let conn = Connection::open("handwritten.sqlite").unwrap();
    conn.execute("DROP TABLE IF EXISTS pca", ()).unwrap();
    conn.execute("CREATE TABLE pca (row_id INTEGER PRIMARY KEY, brightness REAL NOT NULL, p0 REAL NOT NULL, p1 REAL NOT NULL, p2 REAL NOT NULL)", ()).unwrap();
    let mut stmt = conn
        .prepare_cached(
            "INSERT OR IGNORE INTO pca (brightness, p0, p1, p2) VALUES (?1, ?2, ?3, ?4)",
        )
        .unwrap();

    for point in encodings.latents_pca.unwrap().axis_iter(Axis(0)) {
        stmt.execute(params![
            1.0,
            point[0],
            point[1],
            point[2]
        ])
        .unwrap();
    }
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
