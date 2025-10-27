use std::{
    collections::hash_map::Entry,
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
    sync::Arc,
};

use burn::data::{dataloader::batcher::Batcher, dataset::Dataset};
use clap::{Parser, Subcommand};
use general_models::{SimpleInfer, wgpu::WgpuBackend};
use general_training::{
    batches::{AutoEncoderImageBatcher, AutoEncoderImageItem},
    dataset::{SqliteDataset, SqliteDatasetConfig},
};
use linfa::traits::Transformer;
use linfa_clustering::Dbscan;
use ndarray::{Array1, Array2, Axis};
use rustc_hash::FxHashMap;
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
    Cluster {
        #[arg(short, long)]
        weights_path: PathBuf,
        #[arg(long, default_value = "model.json")]
        config_path: PathBuf,
        #[arg(short, long)]
        dataset_configs: Vec<PathBuf>,
        #[arg(long, default_value_t = 64)]
        batch_size: usize,
        #[arg(long, default_value_t = 3)]
        min_points: usize,
        #[arg(long, default_value_t = 1e-2)]
        tolerance: f64,
    },
}

fn main() {
    let args = Args::parse();
    let device = general_models::wgpu::get_device();

    match args.command {
        Command::Cluster {
            weights_path,
            config_path,
            dataset_configs,
            batch_size,
            min_points,
            tolerance,
        } => {
            let model = ImageEncoder::<WgpuBackend>::load(config_path, weights_path, device)
                .expect("Configuration should be valid");
            let mut latents: Vec<f32> = vec![];
            let mut latent_size = None;
            let batcher = AutoEncoderImageBatcher { channels: 1 };

            for dataset_config in dataset_configs {
                println!("Reading from {:?}", dataset_config);
                let dataset_config: SqliteDatasetConfig = parse_json_file(dataset_config).unwrap();
                let dataset: SqliteDataset<Arc<AutoEncoderImageItem>> =
                    dataset_config.try_into().unwrap();

                let mut batch_items = Vec::with_capacity(batch_size);
                for i in 0..dataset.len() {
                    batch_items.push(dataset.get(i).unwrap());
                    if batch_items.len() >= batch_size {
                        let batch = batcher.batch(
                            std::mem::replace(&mut batch_items, Vec::with_capacity(batch_size)),
                            device,
                        );
                        let tensor = model.encoder.forward(batch.input);
                        let [_, tmp] = tensor.dims();
                        latent_size = Some(tmp);
                        latents.extend(tensor.into_data().iter::<f32>());
                    }
                }

                if !batch_items.is_empty() {
                    let batch = batcher.batch(batch_items, device);
                    let tensor = model.encoder.forward(batch.input);
                    let [_, tmp] = tensor.dims();
                    latent_size = Some(tmp);
                    latents.extend(tensor.into_data().iter::<f32>());
                    // latents.extend(tensor.iter_dim(0).map(|tensor| tensor.into_data()));
                }

                println!("Latents size: {}", latents.len());
            }

            let Some(latent_size) = latent_size else {
                eprintln!("No latents created");
                return;
            };
            println!("Preparing for Dbscan");

            let latent_points = Array2::from_shape_vec(
                (latents.len() / latent_size, latent_size),
                latents.into_iter().map(|x| x as f64).collect(),
            )
            .unwrap();

            println!("Running Dbscan");

            let clusters = Dbscan::params::<f64>(min_points)
                .tolerance(tolerance)
                .transform(&latent_points)
                .unwrap();

            println!("Interpreting Dbscan");

            let mut unknown_count = 0usize;
            let mut sums = FxHashMap::<usize, (usize, Array1<f64>)>::default();

            for (cluster, point) in clusters
                .axis_iter(Axis(0))
                .zip(latent_points.axis_iter(Axis(0)))
            {
                if let Some(cluster_id) = cluster.as_slice().unwrap()[0] {
                    match sums.entry(cluster_id) {
                        Entry::Occupied(mut occupied_entry) => {
                            occupied_entry.get_mut().0 += 1;
                            let sum = occupied_entry.get().1.clone() + point;
                            occupied_entry.get_mut().1 = sum;
                        }
                        Entry::Vacant(vacant_entry) => {
                            vacant_entry.insert((1, point.to_owned()));
                        }
                    }
                } else {
                    unknown_count += 1;
                }
            }

            println!("Writing results");

            {
                let mut cluster_size_file =
                    BufWriter::new(File::create("cluster_size.csv").unwrap());
                writeln!(cluster_size_file, "cluster id, count").unwrap();
                for (cluster_id, (count, _)) in &sums {
                    writeln!(cluster_size_file, "{cluster_id},{count}").unwrap();
                }
                writeln!(cluster_size_file, "unknown,{}", unknown_count).unwrap();
            }

            {
                let mut centroids_file = BufWriter::new(File::create("centroids.csv").unwrap());
                writeln!(centroids_file, "cluster id,").unwrap();
                for (cluster_id, (count, sum)) in &sums {
                    let mean = sum / *count as f64;
                    write!(centroids_file, "{cluster_id}").unwrap();
                    for c in mean {
                        write!(centroids_file, ",{c:>8.4}").unwrap();
                    }
                    writeln!(centroids_file).unwrap();
                }
            }

            {
                let mut ids_file = BufWriter::new(File::create("ids.csv").unwrap());
                writeln!(ids_file, "i,cluster id").unwrap();
                for (i, (cluster, point)) in clusters
                    .axis_iter(Axis(0))
                    .zip(latent_points.axis_iter(Axis(0)))
                    .enumerate()
                {
                    if let Some(cluster_id) = cluster.as_slice().unwrap()[0] {
                        write!(ids_file, "{i},{cluster_id}").unwrap();
                    } else {
                        write!(ids_file, "{i},-1").unwrap();
                    }
                    for c in point {
                        write!(ids_file, ",{c:>8.4}").unwrap();
                    }
                    writeln!(ids_file).unwrap();
                }
            }
        }
    }
}
