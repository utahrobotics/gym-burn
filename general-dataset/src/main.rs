use std::io::{Cursor, Read};
use clap::{Parser, Subcommand};
use image::{Luma, LumaA, Rgb, Rgba, ImageFormat, ImageBuffer};
use rand::{SeedableRng, rngs::SmallRng};
use rand_distr::{Distribution, Normal};
use rusqlite::{Connection, params};
use rayon::{join, prelude::*};
use sha2::Digest;


#[derive(Debug, Subcommand)]
enum ProcessStdinPreset {
    AutoEncoder
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    db_path: String,
    #[arg(short, long)]
    table_name: String,
    #[arg(short, long)]
    noise_levels: Vec<f32>,
    #[command(subcommand)]
    preset: ProcessStdinPreset,
}


fn process_stdin(db_path: String, table_name: String, noise_levels: Vec<f32>, preset: ProcessStdinPreset) {
    let mut stdin = std::io::stdin().lock();
    let conn = Connection::open(db_path).expect("SQLite database should be accessible");
    
    conn.execute(&format!("CREATE TABLE IF NOT EXISTS images (row_id INTEGER PRIMARY KEY, sha256hex TEXT NOT NULL UNIQUE, webp BLOB NOT NULL) STRICT"), ()).unwrap();
    conn.execute(&format!("CREATE TABLE IF NOT EXISTS {table_name} (
        row_id INTEGER PRIMARY KEY,
        input INTEGER NOT NULL,
        expected INTEGER NOT NULL,
        FOREIGN KEY (input) REFERENCES images(row_id),
        FOREIGN KEY (expected) REFERENCES images(row_id),
        UNIQUE(input, expected)
) STRICT"), ()).unwrap();
    
    match preset {
        ProcessStdinPreset::AutoEncoder => {
            let mut size_buf = [0u8; 4];
            let mut image_buf = vec![];
            loop {
                stdin.read_exact(&mut size_buf).unwrap();
                let width = u32::from_le_bytes(size_buf);
                stdin.read_exact(&mut size_buf).unwrap();
                let height = u32::from_le_bytes(size_buf);
                stdin.read_exact(&mut size_buf[0..1]).unwrap();
                let channels = size_buf[0];
                
                if !matches!(channels, 1 | 2 | 3 | 4) {
                    panic!("Unsupported number of channels: {channels}");
                }
                
                image_buf.resize(width as usize * height as usize * channels as usize, 0u8);
                stdin.read_exact(&mut image_buf).unwrap();
                let image_buf = std::mem::take(&mut image_buf);
                
                macro_rules! img {
                    ($px: ty) => {
                        let (original_webp_buf, noisy_webp_bufs) = join(
                            || {
                                let mut webp_buf = Cursor::new(vec![]);
                                let image = ImageBuffer::<$px, _>::from_raw(width, height, image_buf.as_slice()).unwrap();
                                image.write_to(&mut webp_buf, ImageFormat::WebP).unwrap();
                                webp_buf.into_inner()
                            },
                            || {
                                noise_levels.par_iter()
                                    .map(|&std_dev| {
                                        let mut rng = SmallRng::from_rng(&mut rand::rng());
                                        let new_image_buf: Vec<_> = image_buf.iter()
                                            .map(|&p| {
                                                Normal::new(p as f32, std_dev).unwrap().sample(&mut rng).round().clamp(0.0, 255.0) as u8
                                            })
                                            .collect();
                                        let image = ImageBuffer::<$px, _>::from_raw(width, height, new_image_buf).unwrap();
                                        let mut webp_buf = Cursor::new(vec![]);
                                        image.write_to(&mut webp_buf, ImageFormat::WebP).unwrap();
                                        webp_buf.into_inner()
                                    })
                                    .collect::<Vec<_>>()
                            }
                        );
                        let mut insert_image_stmt = conn.prepare("INSERT IGNORE INTO images (sha256hex, webp) VALUES (?, ?)").unwrap();
                        let mut insert_table_stmt = conn.prepare(&format!("INSERT IGNORE INTO {table_name} (input, expected) VALUES (?, ?)")).unwrap();
                        
                        let hash = sha2::Sha256::digest(&original_webp_buf);
                        let mut hex_out = vec![0u8; 64];
                        hex::encode_to_slice(hash, &mut hex_out).unwrap();
                        let original_sha256hex = String::from_utf8(hex_out).unwrap();

                        noisy_webp_bufs.iter()
                            .map(|webp_buf| {
                                let hash = sha2::Sha256::digest(&webp_buf);
                                let mut hex_out = vec![0u8; 64];
                                hex::encode_to_slice(hash, &mut hex_out).unwrap();
                                (String::from_utf8(hex_out).unwrap(), webp_buf)
                            })
                            .chain(Some((original_sha256hex.clone(), &original_webp_buf)))
                            .for_each(|(sha256hex, webp_buf)| {
                                insert_image_stmt.execute(params![sha256hex, webp_buf]).unwrap();
                                insert_table_stmt.execute(params![sha256hex, original_sha256hex]).unwrap();
                            });
                    }
                }
                
                match channels {
                    1 => {
                        img!(Luma<u8>);
                    }
                    2 => {
                        img!(LumaA<u8>);
                    }
                    3 => {
                        img!(Rgb<u8>);
                    }
                    4 => {
                        img!(Rgba<u8>);
                    }
                    _ => unreachable!()
                }
            }
        }
    }
}

fn main() {
    let args = Args::parse();
    process_stdin(args.db_path, args.table_name, args.noise_levels, args.preset);
}
