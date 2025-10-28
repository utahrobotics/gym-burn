#![recursion_limit = "256"]
use std::{
    collections::hash_map::Entry,
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
};

use clap::{Parser, Subcommand};
use general_dataset::{
    SqliteDataset, SqliteDatasetConfig, presets::autoencoder::AutoEncoderImageBatcher,
};
use general_models::{SimpleInfer, wgpu::WgpuBackend};
use linfa::{
    DatasetBase,
    traits::{Fit, Predict, Transformer},
};
use linfa_clustering::Dbscan;
use linfa_reduction::Pca;
use ndarray::{Array1, Array2, Axis};
use rerun::Color;
use rusqlite::{Connection, ToSql};
use rustc_hash::FxHashMap;
use serde_json::json;
use utils::parse_json_file;

use handwritten::ImageEncoder;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    // Cluster {
    //     #[arg(short, long)]
    //     weights_path: PathBuf,
    //     #[arg(long, default_value = "model.json")]
    //     config_path: PathBuf,
    //     #[arg(short, long)]
    //     dataset_configs: Vec<PathBuf>,
    //     #[arg(long, default_value_t = 256)]
    //     batch_size: usize,
    //     #[arg(long, default_value_t = 3)]
    //     min_points: usize,
    //     #[arg(long, default_value_t = 1e-2)]
    //     tolerance: f64,
    // },
    Pca {
        latents_size: usize,
        #[arg(long, default_value_t = 256)]
        batch_size: usize,
        max_len: Option<usize>
    },
    Infer {
        #[arg(short, long)]
        weights_path: PathBuf,
        #[arg(long, default_value = "model.json")]
        config_path: PathBuf,
        #[arg(short, long)]
        dataset_configs: Vec<PathBuf>,
        #[arg(long, default_value_t = 256)]
        batch_size: usize,
    },
}

fn main() {
    let args = Args::parse();
    let device = general_models::wgpu::get_device();

    match args.command {
        Command::Pca { latents_size, batch_size, max_len } => {
            let conn = Connection::open("latents.sqlite").unwrap();
            let mut stmt = if let Some(max_len) = max_len {
                conn.prepare(&format!("SELECT * FROM latents ORDER BY RANDOM() LIMIT {max_len}")).unwrap()
            } else {
                conn.prepare("SELECT * FROM latents ORDER BY RANDOM()").unwrap()
            };
            let mut query = stmt.query(()).unwrap();
            let mut latents: Vec<f64> = vec![];
            let mut brightnesses: Vec<f64> = vec![];

            while let Some(row) = query.next().unwrap() {
                for i in 0..latents_size {
                    latents.push(row.get(&*format!("p{i}")).unwrap());
                }
                brightnesses.push(row.get("brightness").unwrap());
            }

            let max_brightness = *brightnesses.iter().max_by(|a, b| a.total_cmp(b)).unwrap();

            let latent_points = Array2::from_shape_vec(
                (latents.len() / latents_size, latents_size),
                latents,
            )
            .unwrap();

            let pca_dataset = DatasetBase::new(
                latent_points.clone(),
                // targets is ignored
                Array2::<f64>::zeros((latent_points.shape()[0], 1)),
            );

            println!("Running PCA 3D");
            let embedding3 = Pca::params(3).fit(&pca_dataset).unwrap();
            
            let explained_ratio = embedding3.explained_variance_ratio();
            println!("Explained Variance Ratio: {explained_ratio}");
            println!("Total: {:.2}%", explained_ratio.sum() * 100.0);

            serde_json::to_writer_pretty(
                std::fs::File::create("pca.json").unwrap(),
                &json!({
                    "mean": embedding3.mean(),
                    "components": embedding3.components()
                })
            ).unwrap();

            let output3 = embedding3.predict(&pca_dataset);

            println!("Saving PCA 3D");
            let points: Vec<_> = output3.axis_iter(Axis(0)).collect();

            let _ = std::fs::remove_file("clustering.rrd");
            let rerun_save = rerun::RecordingStreamBuilder::new("handwritten_rerun")
                .save("clustering.rrd")
                .unwrap();
            
            for (points, brightnesses) in points.chunks(batch_size).zip(brightnesses.chunks(batch_size)) {
                rerun_save
                    .log(
                        "pca3d",
                        &rerun::Points3D::new(
                            points.iter().map(|point| (point[0] as f32, point[1] as f32, point[2] as f32)),
                        )
                        .with_colors(brightnesses.iter().map(|b| (*b / max_brightness * 255.0) as u8).map(|b| Color::from_rgb(b, b, b)))
                        .with_radii((0..points.len()).map(|_| 0.01)),
                    )
                    .unwrap();
            }
        },
        Command::Infer {
            weights_path,
            config_path,
            dataset_configs,
            batch_size,
        } => {
            let model = ImageEncoder::<WgpuBackend, handwritten::Model<WgpuBackend>>::load(
                config_path,
                weights_path,
                device,
            )
            .expect("Configuration should be valid");

            let latents_size = model.get_encoder().linear.get_output_size();
            let mut batcher = AutoEncoderImageBatcher::new(1, device.clone());

            let conn = Connection::open("latents.sqlite").unwrap();
            let mut fields = String::new();

            for i in 0..latents_size {
                fields.push_str(", p");
                fields.push_str(&i.to_string());
                fields.push_str(" REAL NOT NULL");
            }

            conn.execute(&format!("CREATE TABLE IF NOT EXISTS latents (row_id INTEGER PRIMARY KEY, brightness REAL NOT NULL{fields})"), ()).unwrap();
            conn.execute("DELETE FROM latents", ()).unwrap();
            fields.clear();
            let mut inputs = String::new();

            for i in 0..latents_size {
                fields.push_str(", p");
                fields.push_str(&i.to_string());
                inputs.push_str(", ?");
                inputs.push_str(&(i + 2).to_string());
            }

            let mut stmt = conn.prepare_cached(&format!("INSERT INTO latents (brightness{fields}) VALUES (?1{inputs})")).unwrap();

            for dataset_config in dataset_configs {
                println!("Reading from {:?}", dataset_config);
                let dataset_config: SqliteDatasetConfig = parse_json_file(dataset_config).unwrap();
                let dataset: SqliteDataset = dataset_config.try_into().unwrap();

                for i in 0..dataset.get_batch_count(batch_size) {
                    let batch = dataset.query(i, batch_size, &mut batcher);
                    let tensor = model.get_encoder().forward(batch.input);
                    // if tensor.clone().contains_nan().into_scalar() != 0 {
                    //     println!("Found NaN");
                    //     continue;
                    // }
                    // latents.extend(tensor.into_data().iter::<f32>());
                    // expected_brightnesses.extend(batch.expected.mean_dims(&[1, 2, 3]).into_data().iter::<f32>());
                    let expected_data = batch.expected.mean_dims(&[1, 2, 3]).into_data().into_vec::<f32>().unwrap();
                    let components = tensor.into_data().into_vec::<f32>().unwrap();

                    let iter = expected_data.iter().zip(components.chunks(latents_size));
                    let mut params = Vec::<&dyn ToSql>::with_capacity(1 + latents_size);

                    for (brightness, point) in iter {
                        params.clear();
                        params.push(brightness);
                        params.extend(point.iter().map(|p| p as &dyn ToSql));
                        stmt.execute(params.as_slice()).unwrap();
                    }
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
        }
    }
}
