use image::{
    ImageBuffer, ImageFormat, Rgb,
    imageops::{FilterType, resize},
};
use imageproc::definitions::HasBlack;
use rayon::prelude::*;
use rusqlite::{Connection, params};
use serde::Deserialize;
use sha2::Digest;
use std::{
    io::{Cursor, ErrorKind, Read},
    num::NonZeroU32,
    process::Stdio,
};
use utils::parse_json_file;

use crate::image_gen::{flips, noises, rotations, to_webp, translations};

mod image_gen;

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
    rotations_count: usize,
    translations_count: usize,
    translation_std_dev_multiplier: f32,
    noises_count: usize,
    #[serde(default)]
    noise_levels: Vec<f64>,
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
        rotations_count,
        translations_count,
        translation_std_dev_multiplier,
        noises_count,
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
                                    let white_count: usize = image.pixels().filter(|p| p.0[0] > 127).map(|_| 1).sum();
                                    if white_count > width as usize * height as usize / 2 {
                                        image.par_pixels_mut().for_each(|p| {
                                            let inverted = 255 - p.0[0];
                                            p.0 = [inverted, inverted, inverted];
                                        });
                                    }
                                    // let mut bytes: Vec<_> =
                                    //     image.pixels().map(|p| p.0[0]).collect();
                                    // bytes.sort_unstable();
                                    // let median = bytes[bytes.len() / 2];

                                    // if median > 127 {
                                    //     image.par_pixels_mut().for_each(|p| {
                                    //         let inverted = 255 - p.0[0];
                                    //         p.0 = [inverted, inverted, inverted];
                                    //     });
                                    // }
                                }
                            }
                        }
                    }
                    _ => panic!("Unsupported format byte: {format_byte}"),
                }

                let original_webp_buf = {
                    let mut webp_buf = Cursor::new(vec![]);
                    image.write_to(&mut webp_buf, ImageFormat::WebP).unwrap();
                    webp_buf.into_inner()
                };
                let hash = sha2::Sha256::digest(&original_webp_buf);
                let mut hex_out = vec![0u8; 64];
                hex::encode_to_slice(hash, &mut hex_out).unwrap();
                let original_sha256hex = String::from_utf8(hex_out).unwrap();

                let x = flips(Some(image).into_par_iter());
                let x = rotations(x, rotations_count, Rgb::black());
                let x = translations(x, translations_count, translation_std_dev_multiplier, Rgb::black());
                let x = noises(x, noises_count, noise_levels.par_iter().copied());
                let webp_images: Vec<_> = to_webp(x).collect();

                let mut insert_image_stmt = conn
                    .prepare_cached("INSERT OR IGNORE INTO images (sha256hex, webp, width, height) VALUES (?, ?, ?, ?)")
                    .unwrap();

                insert_image_stmt
                    .execute(params![original_sha256hex, original_webp_buf, width, height])
                    .unwrap();

                let mut multi_insert_image_stmt = conn
                    .prepare_cached("INSERT OR IGNORE INTO images (sha256hex, webp, width, height) VALUES (?3, ?4, ?1, ?2), (?5, ?6, ?1, ?2), (?7, ?8, ?1, ?2), (?9, ?10, ?1, ?2)")
                    .unwrap();

                let mut insert_table_stmt = conn
                    .prepare_cached(&format!(
                        "INSERT OR IGNORE INTO {table_name} (input, expected) 
                            SELECT i1.row_id, i2.row_id 
                            FROM images i1, images i2 
                            WHERE i1.sha256hex = ? AND i2.sha256hex = ?"
                    ))
                    .unwrap();

                let mut multi_insert_table_stmt = conn
                    .prepare_cached(&format!(
                        "INSERT OR IGNORE INTO {table_name} (input, expected) 
                            SELECT i1.row_id, i2.row_id 
                            FROM images i1, images i2 
                            WHERE
                                (i1.sha256hex = ?2 AND i2.sha256hex = ?1) OR
                                (i1.sha256hex = ?3 AND i2.sha256hex = ?1) OR
                                (i1.sha256hex = ?4 AND i2.sha256hex = ?1) OR
                                (i1.sha256hex = ?5 AND i2.sha256hex = ?1)"
                    ))
                    .unwrap();

                let hex_hash = |bytes: &[u8]| {
                    let hash = sha2::Sha256::digest(bytes);
                    let mut hex_out = vec![0u8; 64];
                    hex::encode_to_slice(hash, &mut hex_out).unwrap();
                    String::from_utf8(hex_out).unwrap()
                };
                let original_sha256hex = hex_hash(&original_webp_buf);

                webp_images
                    .chunks(4)
                    .for_each(|webp_bufs| {
                        if webp_bufs.len() == 4 {
                            let sha256hex0 = hex_hash(&webp_bufs[0]);
                            let sha256hex1 = hex_hash(&webp_bufs[1]);
                            let sha256hex2 = hex_hash(&webp_bufs[2]);
                            let sha256hex3 = hex_hash(&webp_bufs[3]);
                            multi_insert_image_stmt
                                .execute(params![
                                    width, height,
                                    sha256hex0, webp_bufs[0],
                                    sha256hex1, webp_bufs[1],
                                    sha256hex2, webp_bufs[2],
                                    sha256hex3, webp_bufs[3],
                                ])
                                .unwrap();
                            multi_insert_table_stmt
                                .execute(params![original_sha256hex, sha256hex0, sha256hex1, sha256hex2, sha256hex3])
                                .unwrap();
                        } else {
                            webp_bufs.iter()
                                .for_each(|webp_buf| {
                                    let sha256hex = hex_hash(webp_buf);
                                    insert_image_stmt
                                        .execute(params![sha256hex, webp_buf, width, height])
                                        .unwrap();
                                    insert_table_stmt
                                        .execute(params![sha256hex, original_sha256hex])
                                        .unwrap();
                                });
                        }
                    });
            }
        }
    }
}
