fn main() {
    println!("Hello, world!");
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