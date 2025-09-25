use std::{
    io::{Cursor, Write},
    path::Path,
};

use crossbeam::{queue::SegQueue, utils::Backoff};
use image::{DynamicImage, ImageFormat, imageops::FilterType};
use rand::{SeedableRng, rngs::SmallRng};
use rand_distr::{Distribution, Normal};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, ParallelBridge, ParallelIterator,
};
use rusqlite::{Connection, ErrorCode, params_from_iter};
use sha2::{Digest, Sha256};

fn recursive_iter(
    path: &Path,
    f: &(impl Fn(&Path, Box<dyn FnOnce() -> DynamicImage + 'static>) + Sync),
) {
    if path.is_dir() {
        std::fs::read_dir(&path)
            .unwrap()
            .par_bridge()
            .for_each(|result| {
                let entry = result.unwrap();
                recursive_iter(&entry.path(), f);
            });
    } else {
        let Some(extension) = path.extension().map(|s| s.to_str().unwrap()) else {
            return;
        };
        if !matches!(extension, "png" | "webp" | "jpg") {
            return;
        }
        let next = path.to_path_buf();
        let blob_fn = Box::new(move || {
            let mut img = image::open(next).unwrap();
            img = img.resize_exact(28, 28, FilterType::CatmullRom);
            validate_color(&mut img);
            img
        });
        f(&path, blob_fn);
    }
}

fn image_to_webp(img: &DynamicImage) -> Vec<u8> {
    let mut blob = vec![];
    img.write_to(&mut Cursor::new(&mut blob), ImageFormat::WebP)
        .unwrap();
    blob
}

fn validate_color(img: &mut DynamicImage) {
    let mut luma_bytes = img.to_luma8().into_vec();
    luma_bytes.sort_unstable();
    if luma_bytes[luma_bytes.len() / 2] > 127 {
        img.invert();
    }
}

pub const SQLITE_DATABASE: &str = "isthatarock-handwritten-dataset.sqlite";

// fn hash_file(file: &Path) -> [u8; 32] {
//     let mut sha2 = Sha256::new();
//     std::io::copy(&mut std::fs::File::open(file).unwrap(), &mut sha2).unwrap();
//     sha2.finalize().as_slice().try_into().unwrap()
// }

fn main() {
    let Some(command) = std::env::args().nth(1) else {
        eprintln!("Provide a command");
        return;
    };

    let conn = Connection::open(SQLITE_DATABASE).unwrap();
    conn.execute(
        r#"
        CREATE TABLE IF NOT EXISTS `train` (
            `row_id` INTEGER PRIMARY KEY,
            `hash` TEXT NOT NULL,
            `image_blob` BLOB NOT NULL,
            `noisy` INTEGER NOT NULL DEFAULT 0,
            `scaled` INTEGER NOT NULL DEFAULT 0,
            `damaged` INTEGER NOT NULL DEFAULT 0,
            `rotated` INTEGER NOT NULL DEFAULT 0,
            `translated` INTEGER NOT NULL DEFAULT 0
        ) STRICT"#,
        (),
    )
    .unwrap();
    conn.execute(
        r#"
        CREATE TABLE IF NOT EXISTS `test` (
            `row_id` INTEGER PRIMARY KEY,
            `hash` TEXT NOT NULL,
            `image_blob` BLOB NOT NULL,
            `noisy` INTEGER NOT NULL DEFAULT 0,
            `scaled` INTEGER NOT NULL DEFAULT 0,
            `damaged` INTEGER NOT NULL DEFAULT 0,
            `rotated` INTEGER NOT NULL DEFAULT 0,
            `translated` INTEGER NOT NULL DEFAULT 0
        ) STRICT"#,
        (),
    )
    .unwrap();

    let second_arg = std::env::args().nth(2);
    let third_arg = std::env::args().nth(3);
    match command.as_str() {
        "read" => {
            let Some(table) = second_arg else {
                eprintln!("No table provided");
                return;
            };
            let Some(extras) = third_arg else {
                eprintln!("No additional constraints provided");
                return;
            };
            let mut stmt = conn
                .prepare(&format!(
                    "SELECT row_id, image_blob FROM {table} WHERE {extras}"
                ))
                .unwrap();
            let mut rows = stmt.query(()).unwrap();
            while let Some(row) = rows.next().unwrap() {
                let row_id: u32 = row.get("row_id").unwrap();
                let image_blob: Vec<u8> = row.get("image_blob").unwrap();
                std::fs::write(format!("{row_id}.webp"), image_blob).unwrap();
            }
        }
        "select" => {
            let Some(stmt) = second_arg else {
                eprintln!("No statement provided");
                return;
            };
            let stmt = stmt.trim();
            if !stmt.starts_with("select") && !stmt.starts_with("SELECT") {
                eprintln!("Not a select statement");
                return;
            }
            let mut stmt = conn.prepare(stmt).unwrap();
            let mut rows = stmt.query(()).unwrap();
            while let Some(row) = rows.next().unwrap() {
                println!("{row:?}");
                // let row_id: u32 = row.get("row_id").unwrap();
                // let image_blob: Vec<u8> = row.get("image_blob").unwrap();
                // std::fs::write(format!("{row_id}.webp"), image_blob).unwrap();
            }
        }
        "append" => {
            let Some(split) = second_arg else {
                eprintln!("No split provided");
                return;
            };
            let Some(path) = third_arg else {
                eprintln!("No path provided");
                return;
            };
            if !matches!(split.as_str(), "train" | "test") {
                eprintln!("Invalid split. Must be 'train' or 'test'");
                return;
            }
            let conn_queue = SegQueue::new();
            conn_queue.push(conn);
            recursive_iter(path.as_ref(), &|path, img_fn| {
                let conn = conn_queue
                    .pop()
                    .unwrap_or_else(|| Connection::open(SQLITE_DATABASE).unwrap());
                let mut sha2 = Sha256::new();
                std::io::copy(&mut std::fs::File::open(path).unwrap(), &mut sha2).unwrap();
                let mut hash_hex_bytes = [0u8; 64];
                hex::encode_to_slice(sha2.clone().finalize(), &mut hash_hex_bytes).unwrap();
                let hash_hex = std::str::from_utf8(&hash_hex_bytes).unwrap();
                let image = img_fn();
                let mut blobs = vec![image_to_webp(&image)];
                let image = image.to_rgb8();
                let mut tmp_blobs = vec![];

                const NOISE_STD_DEV: f32 = 20.0;
                const NOISE_COUNT: usize = 3;

                (0..NOISE_COUNT)
                    .into_par_iter()
                    .map(|i| {
                        let mut sha2 = sha2.clone();
                        sha2.write(&i.to_ne_bytes()).unwrap();
                        let mut image = image.clone();
                        let mut rng =
                            SmallRng::from_seed(sha2.finalize().as_slice().try_into().unwrap());
                        image.pixels_mut().for_each(|p| {
                            let new = Normal::new(p.0[0] as f32, NOISE_STD_DEV)
                                .unwrap()
                                .sample(&mut rng)
                                .clamp(0.0, 255.0)
                                .round() as u8;
                            p.0 = [new; 3];
                        });
                        image_to_webp(&image.into())
                    })
                    .collect_into_vec(&mut tmp_blobs);
                blobs.append(&mut tmp_blobs);

                let mut insert_cmd = format!(
                    "INSERT INTO `{split}` (`hash`, `image_blob`, `noisy`) VALUES ('{hash_hex}', ?1, 0)"
                );

                for i in 0..NOISE_COUNT {
                    insert_cmd.push_str(", ('");
                    insert_cmd.push_str(hash_hex);
                    insert_cmd.push_str("', ?");
                    insert_cmd.push_str(&(i + 2).to_string());
                    insert_cmd.push_str(", 1)");
                }

                let backoff = Backoff::new();

                loop {
                    if let Err(e) = conn.execute(&insert_cmd, params_from_iter(blobs.iter())) {
                        if let Some(ErrorCode::DatabaseBusy) = e.sqlite_error_code() {
                            backoff.spin();
                            continue;
                        }
                        panic!("{e}");
                    }
                    break;
                }

                conn_queue.push(conn);
            });
        }
        _ => {
            eprintln!("Unknown command");
            return;
        }
    }
}
