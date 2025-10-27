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
            let model = ImageEncoder::<WgpuBackend, handwritten::Model<WgpuBackend>>::load(
                config_path,
                weights_path,
                device,
            )
            .expect("Configuration should be valid");

            let mut latents: Vec<f32> = vec![];
            let mut latent_size = None;
            let mut batcher = AutoEncoderImageBatcher::new(1, device.clone());

            for dataset_config in dataset_configs {
                println!("Reading from {:?}", dataset_config);
                let dataset_config: SqliteDatasetConfig = parse_json_file(dataset_config).unwrap();
                let dataset: SqliteDataset = dataset_config.try_into().unwrap();

                for i in 0..dataset.get_batch_count(batch_size) {
                    let batch = dataset.query(i, batch_size, &mut batcher);
                    let tensor = model.get_encoder().forward(batch.input);
                    let [_, tmp] = tensor.dims();
                    latent_size = Some(tmp);
                    latents.extend(tensor.into_data().iter::<f32>());
                }

                println!("Latents size: {}", latents.len());
            }

            let Some(latent_size) = latent_size else {
                eprintln!("No latents created");
                return;
            };

            let latent_points = Array2::from_shape_vec(
                (latents.len() / latent_size, latent_size),
                latents.into_iter().map(|x| x as f64).collect(),
            )
            .unwrap();


            let pca_dataset = DatasetBase::new(
                latent_points.clone(),
                // targets is ignored
                Array2::<f64>::zeros((latent_points.shape()[0], 1)),
            );

            {
                let rerun_save = rerun::RecordingStreamBuilder::new("handwritten_rerun")
                    .save("clustering.rrd")
                    .unwrap();

                println!("Running PCA 2D");
                let embedding2 = Pca::params(2).fit(&pca_dataset).unwrap();
                let output2 = embedding2.predict(&pca_dataset);

                // rerun_save
                //     .log(
                //         "pca2d",
                //         &rerun::Points2D::new(
                //             output2.axis_iter(Axis(0)).map(|p| (p[0] as f32, p[1] as f32)),
                //         )
                //         .with_radii((0..output2.shape()[0]).map(|_| 0.1)),
                //     )
                //     .unwrap();
                println!("Saving PCA 2D");
                for point in output2.axis_iter(Axis(0)) {
                    rerun_save
                        .log(
                            "pca2d",
                            &rerun::Points2D::new(
                                [(point[0] as f32, point[1] as f32)],
                            )
                            .with_radii(Some(0.1)),
                        )
                        .unwrap();
                }

                println!("Running PCA 3D");
                let embedding3 = Pca::params(3).fit(&pca_dataset).unwrap();
                let output3 = embedding3.predict(&pca_dataset);

                println!("Saving PCA 3D");
                for point in output3.axis_iter(Axis(0)) {
                    rerun_save
                        .log(
                            "pca3d",
                            &rerun::Points3D::new(
                                [(point[0] as f32, point[1] as f32, point[2] as f32)],
                            )
                            .with_radii(Some(0.1)),
                        )
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
        }
    }
}
