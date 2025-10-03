use image::{
    ImageBuffer, ImageFormat, Rgb,
    imageops::{FilterType, resize},
};
use rand::{SeedableRng, rngs::SmallRng};
use rand_distr::{Distribution, Normal};
use rayon::{join, prelude::*};
use rusqlite::{Connection, params};
use serde::Deserialize;
use sha2::Digest;
use std::{
    io::{Cursor, ErrorKind, Read},
    num::NonZeroU32,
    process::Stdio,
};
use utils::parse_json_file;

#[derive(Debug, Deserialize)]
enum ProcessStdinPreset {
    #[serde(rename = "auto-encoder")]
    AutoEncoder,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
enum ColorSetting {
    #[serde(alias = "c")]
    #[default]
    Color,
    #[serde(alias = "bw")]
    BlackAndWhite,
    #[serde(alias = "pb-w")]
    PrimarilyBlackThenWhite,
}

#[derive(Deserialize)]
pub struct Config {
    db_path: String,
    table_name: String,
    #[serde(default)]
    noise_levels: Vec<f32>,
    #[serde(default)]
    color: ColorSetting,
    resize_to: Option<[NonZeroU32; 2]>,
    #[serde(default)]
    source_command: Vec<String>,
    preset: ProcessStdinPreset,
}

fn main() {
    let Config {
        db_path,
        table_name,
        noise_levels,
        color,
        resize_to,
        preset,
        source_command,
    }: Config = parse_json_file("dataset-config").unwrap();
    let mut input_reader: Box<dyn Read>;
    let child;

    if source_command.is_empty() {
        input_reader = Box::new(std::io::stdin().lock());
    } else {
        let mut string_iter = source_command.into_iter();
        child = std::process::Command::new(string_iter.next().unwrap())
            .args(string_iter)
            .stdout(Stdio::piped())
            .spawn()
            .expect("Failed to spawn source command");
        input_reader = Box::new(child.stdout.unwrap());
    }
    let conn = Connection::open(db_path).expect("SQLite database should be accessible");

    conn.execute(&format!("CREATE TABLE IF NOT EXISTS images (row_id INTEGER PRIMARY KEY, sha256hex TEXT NOT NULL UNIQUE, webp BLOB NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL) STRICT"), ()).unwrap();
    conn.execute(
        &format!(
            "CREATE TABLE IF NOT EXISTS {table_name} (
        row_id INTEGER PRIMARY KEY,
        input INTEGER NOT NULL,
        expected INTEGER NOT NULL,
        FOREIGN KEY (input) REFERENCES images(row_id),
        FOREIGN KEY (expected) REFERENCES images(row_id),
        UNIQUE(input, expected)
) STRICT"
        ),
        (),
    )
    .unwrap();

    let mut width;
    let mut height;

    match preset {
        ProcessStdinPreset::AutoEncoder => {
            let mut size_buf = [0u8; 4];
            for image_count in 0.. {
                if let Err(e) = input_reader.read_exact(&mut size_buf[0..1]) {
                    match e.kind() {
                        ErrorKind::UnexpectedEof => {
                            println!("Processed {image_count} images");
                            break;
                        }
                        _ => panic!("{e}"),
                    }
                }
                let format_byte = size_buf[0];

                let mut image: ImageBuffer<Rgb<u8>, Vec<_>>;
                match format_byte {
                    0 => unimplemented!("Raw pixel data is currently unsupported"),
                    // 0 => {
                    //     stdin.read_exact(&mut size_buf).unwrap();
                    //     width = u32::from_le_bytes(size_buf);
                    //     stdin.read_exact(&mut size_buf).unwrap();
                    //     height = u32::from_le_bytes(size_buf);
                    //     stdin.read_exact(&mut size_buf[0..1]).unwrap();
                    //     channels = size_buf[0];

                    //     if !matches!(channels, 1 | 2 | 3 | 4) {
                    //         panic!("Unsupported number of channels: {channels}");
                    //     }

                    //     image_buf = vec![0u8; width as usize * height as usize * channels as usize];
                    //     stdin.read_exact(&mut image_buf).unwrap();
                    // }
                    1 => {
                        input_reader.read_exact(&mut size_buf).unwrap();
                        let size = u32::from_le_bytes(size_buf);
                        let mut input_buf = vec![0u8; size as usize];
                        input_reader.read_exact(&mut input_buf).unwrap();
                        image = image::load(Cursor::new(input_buf), ImageFormat::Jpeg)
                            .expect("Expected a valid JPEG")
                            .into_rgb8();
                        width = image.width();
                        height = image.height();
                        if let Some([nwidth, nheight]) = resize_to {
                            width = nwidth.get();
                            height = nheight.get();
                            image =
                                resize(&image, nwidth.get(), nheight.get(), FilterType::CatmullRom);
                        }

                        match color {
                            ColorSetting::Color => {}
                            ColorSetting::BlackAndWhite | ColorSetting::PrimarilyBlackThenWhite => {
                                image.pixels_mut().for_each(|p| {
                                    let avg =
                                        ((p.0[0] as u16 + p.0[1] as u16 + p.0[2] as u16) / 3) as u8;
                                    p.0 = [avg, avg, avg];
                                });
                                if color == ColorSetting::PrimarilyBlackThenWhite {
                                    let mut bytes: Vec<_> =
                                        image.pixels().map(|p| p.0[0]).collect();
                                    bytes.sort_unstable();
                                    let median = bytes[bytes.len() / 2];

                                    if median > 127 {
                                        image.par_pixels_mut().for_each(|p| {
                                            let inverted = 255 - p.0[0];
                                            p.0 = [inverted, inverted, inverted];
                                        });
                                    }
                                }
                            }
                        }
                    }
                    _ => panic!("Unsupported format byte: {format_byte}"),
                }

                let (original_webp_buf, noisy_webp_bufs) = join(
                    || {
                        let mut webp_buf = Cursor::new(vec![]);
                        image.write_to(&mut webp_buf, ImageFormat::WebP).unwrap();
                        webp_buf.into_inner()
                    },
                    || {
                        noise_levels
                            .par_iter()
                            .map(|&std_dev| {
                                let mut rng = SmallRng::from_rng(&mut rand::rng());
                                let new_image_buf: Vec<_>;
                                if color == ColorSetting::Color {
                                    new_image_buf = image
                                        .iter()
                                        .map(|&p| {
                                            Normal::new(p as f32, std_dev)
                                                .unwrap()
                                                .sample(&mut rng)
                                                .round()
                                                .clamp(0.0, 255.0)
                                                as u8
                                        })
                                        .collect();
                                } else {
                                    new_image_buf = image
                                        .pixels()
                                        .flat_map(|&p| {
                                            let new_p = Normal::new(p.0[0] as f32, std_dev)
                                                .unwrap()
                                                .sample(&mut rng)
                                                .round()
                                                .clamp(0.0, 255.0)
                                                as u8;
                                            [new_p, new_p, new_p]
                                        })
                                        .collect();
                                }
                                let image = ImageBuffer::<Rgb<u8>, _>::from_raw(
                                    width,
                                    height,
                                    new_image_buf,
                                )
                                .unwrap();
                                let mut webp_buf = Cursor::new(vec![]);
                                image.write_to(&mut webp_buf, ImageFormat::WebP).unwrap();
                                webp_buf.into_inner()
                            })
                            .collect::<Vec<_>>()
                    },
                );
                let mut insert_image_stmt = conn
                    .prepare("INSERT OR IGNORE INTO images (sha256hex, webp, width, height) VALUES (?, ?, ?, ?)")
                    .unwrap();
                let mut insert_table_stmt = conn
                    .prepare(&format!(
                        "INSERT OR IGNORE INTO {table_name} (input, expected) 
                            SELECT i1.row_id, i2.row_id 
                            FROM images i1, images i2 
                            WHERE i1.sha256hex = ? AND i2.sha256hex = ?"
                    ))
                    .unwrap();

                let hash = sha2::Sha256::digest(&original_webp_buf);
                let mut hex_out = vec![0u8; 64];
                hex::encode_to_slice(hash, &mut hex_out).unwrap();
                let original_sha256hex = String::from_utf8(hex_out).unwrap();

                noisy_webp_bufs
                    .iter()
                    .map(|webp_buf| {
                        let hash = sha2::Sha256::digest(&webp_buf);
                        let mut hex_out = vec![0u8; 64];
                        hex::encode_to_slice(hash, &mut hex_out).unwrap();
                        (String::from_utf8(hex_out).unwrap(), webp_buf)
                    })
                    .chain(Some((original_sha256hex.clone(), &original_webp_buf)))
                    .for_each(|(sha256hex, webp_buf)| {
                        insert_image_stmt
                            .execute(params![sha256hex, webp_buf, width, height])
                            .unwrap();
                        insert_table_stmt
                            .execute(params![sha256hex, original_sha256hex])
                            .unwrap();
                    });
            }
        }
    }
}
