use clap::Parser;
use efficient_pca::PCA;
use ndarray::{Array2, Axis};
use rusqlite::{Connection, params};
use serde_json::json;


#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    max_len: Option<usize>
}

fn main() {
    let Args { max_len } = Args::parse();

    let conn = Connection::open("handwritten.sqlite").unwrap();
    let mut stmt = if let Some(max_len) = max_len {
        conn.prepare(&format!("SELECT * FROM latents ORDER BY RANDOM() LIMIT {max_len}")).unwrap()
    } else {
        conn.prepare("SELECT * FROM latents ORDER BY RANDOM()").unwrap()
    };
    let mut query = stmt.query(()).unwrap();
    let mut latents: Vec<f64> = vec![];
    let mut brightnesses: Vec<f64> = vec![];
    let mut latents_size = 0usize;
    let mut first_row = true;

    while let Some(row) = query.next().unwrap() {
        if first_row {
            first_row = false;
            for i in 0.. {
                latents_size = i;
                let Ok(val) = row.get(&*format!("p{i}")) else {
                    break;
                };
                latents.push(val);
            }
        } else {
            for i in 0..latents_size {
                latents.push(row.get(&*format!("p{i}")).unwrap());
            }
        }
        brightnesses.push(row.get("brightness").unwrap());
    }

    let max_brightness = *brightnesses.iter().max_by(|a, b| a.total_cmp(b)).unwrap();

    let latent_points = Array2::from_shape_vec(
        (latents.len() / latents_size, latents_size),
        latents,
    )
    .unwrap();

    println!("Running PCA 3D");
    let mut pca = PCA::new();
    pca.fit(latent_points.clone(), None).expect("Expected PCA to finish succesfully");
    
    let explained_ratio = pca.explained_variance().unwrap().clone() / latents_size as f64;
    println!("Explained Variance Ratio: {explained_ratio}");
    println!("Total: {:.2}%", explained_ratio.sum() * 100.0);

    serde_json::to_writer_pretty(
        std::fs::File::create("pca.json").unwrap(),
        &json!({
            "mean": pca.mean().unwrap(),
            "components": pca.rotation().unwrap()
        })
    ).unwrap();

    let output3 = pca.transform(latent_points).unwrap();

    println!("Saving PCA 3D");
    conn.execute("DROP TABLE IF EXISTS pca", ()).unwrap();
    conn.execute("CREATE TABLE pca (row_id INTEGER PRIMARY KEY, brightness REAL NOT NULL, p0 REAL NOT NULL, p1 REAL NOT NULL, p2 REAL NOT NULL)", ()).unwrap();
    let mut stmt = conn.prepare_cached("INSERT OR IGNORE INTO pca (brightness, p0, p1, p2) VALUES (?1, ?2, ?3, ?4)").unwrap();

    for (point, brightness) in output3.axis_iter(Axis(0)).zip(brightnesses) {
        stmt.execute(params![brightness / max_brightness, point[0], point[1], point[2]]).unwrap();
    }

    // let _ = std::fs::remove_file("clustering.rrd");
    // let rerun_save = rerun::RecordingStreamBuilder::new("handwritten_rerun")
    //     .save("clustering.rrd")
    //     .unwrap();
    
    // for (points, brightnesses) in points.chunks(batch_size).zip(brightnesses.chunks(batch_size)) {
    //     rerun_save
    //         .log(
    //             "pca3d",
    //             &rerun::Points3D::new(
    //                 points.iter().map(|point| (point[0] as f32, point[1] as f32, point[2] as f32)),
    //             )
    //             .with_colors(brightnesses.iter().map(|b| (*b / max_brightness * 255.0) as u8).map(|b| Color::from_rgb(b, b, b)))
    //             .with_radii((0..points.len()).map(|_| 0.01)),
    //         )
    //         .unwrap();
    // }
}
