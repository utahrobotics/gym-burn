use clap::{Parser, Subcommand};
use rerun::{Color, FillMode};
use rusqlite::Connection;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[clap(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    PCA {
        #[clap(default_value = "32")]
        batch_size: usize,
        #[clap(default_value = "20")]
        std_dev_count: usize,
    },
}

fn main() {
    let args = Args::parse();

    match args.command {
        Command::PCA {
            batch_size,
            std_dev_count,
        } => {
            let conn = Connection::open("handwritten.sqlite")
                .expect("Expected handwritten.sqlite to be readable");
            let mut stmt = conn
                .prepare("SELECT * FROM pca ORDER BY brightness DESC")
                .expect("Expected pca table to exist");
            let max_brightness: f32 = conn
                .prepare("SELECT MAX(brightness) AS max_brightness FROM pca")
                .unwrap()
                .query_one((), |row| row.get("max_brightness"))
                .unwrap();

            let _ = std::fs::remove_file("clustering.rrd");
            let rerun_save = rerun::RecordingStreamBuilder::new("handwritten_rerun")
                .save("clustering.rrd")
                .unwrap();

            let mut rows = stmt.query(()).unwrap();
            let mut batch: Vec<[f32; 4]> = vec![];
            let mut collections: Vec<Vec<[f32; 3]>> = vec![vec![]; std_dev_count];

            loop {
                while let Some(row) = rows.next().unwrap() {
                    let brightness = row.get::<_, f32>("brightness").unwrap() / max_brightness;
                    let mut brightness_index = (brightness * std_dev_count as f32).floor() as usize;
                    brightness_index = std_dev_count - brightness_index.min(std_dev_count - 1) - 1;
                    let p0 = row.get("p0").unwrap();
                    let p1 = row.get("p1").unwrap();
                    let p2 = row.get("p2").unwrap();
                    collections[brightness_index].push([p0, p1, p2]);

                    batch.push([p0, p1, p2, brightness]);
                    if batch.len() >= batch_size {
                        break;
                    }
                }

                rerun_save
                    .log(
                        "pca3d",
                        &rerun::Points3D::new(
                            batch.iter().map(|point| (point[0], point[1], point[2])),
                        )
                        .with_colors(
                            batch
                                .iter()
                                .map(|[_, _, _, b]| (*b * 255.0) as u8)
                                .map(|b| Color::from_rgb(b, b, b)),
                        ),
                    )
                    .unwrap();

                if batch.len() < batch_size {
                    break;
                }
                batch.clear();
            }

            for i in 0..std_dev_count {
                let sum = collections
                    .iter()
                    .take(i + 1)
                    .flatten()
                    .copied()
                    .map(|[x, y, z]| [x as f64, y as f64, z as f64])
                    .reduce(|a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2]])
                    .unwrap();
                let count: usize = collections.iter().take(i + 1).map(|c| c.len()).sum();
                let mean = [
                    sum[0] / count as f64,
                    sum[1] / count as f64,
                    sum[2] / count as f64,
                ];
                let mut std_dev: f64 = collections
                    .iter()
                    .take(i + 1)
                    .flatten()
                    .copied()
                    .map(|[x, y, z]| [x as f64, y as f64, z as f64])
                    .map(|[x, y, z]| {
                        let distance_squared =
                            (x - mean[0]).powi(2) + (y - mean[1]).powi(2) + (z - mean[2]).powi(2);
                        distance_squared
                    })
                    .sum();
                std_dev /= count as f64;
                std_dev = std_dev.sqrt();

                let brightness = (std_dev_count - i - 1) as f64 / std_dev_count as f64 * 255.0;
                let brightness = brightness as u8;

                rerun_save
                    .log(
                        "pca3d_std_dev",
                        &rerun::Capsules3D::from_lengths_and_radii([0.0], [std_dev as f32])
                            .with_colors([Color::from_rgb(brightness, brightness, brightness)])
                            .with_translations([
                                mean
                            ])
                            .with_fill_mode(FillMode::Solid)
                    )
                    .unwrap();
            }
        }
    }
}
