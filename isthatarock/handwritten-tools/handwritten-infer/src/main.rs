#![recursion_limit = "256"]
use std::{
    path::PathBuf,
};

use clap::Parser;
use general_dataset::{
    SqliteDataset, SqliteDatasetConfig, presets::autoencoder::AutoEncoderImageBatcher,
};
use general_models::{SimpleInfer, wgpu::WgpuBackend};

use rusqlite::{Connection, ToSql};
use utils::parse_json_file;

use handwritten::ImageEncoder;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    weights_path: PathBuf,
    #[arg(long, default_value = "model.json")]
    config_path: PathBuf,
    #[arg(short, long)]
    dataset_configs: Vec<PathBuf>,
    #[arg(long, default_value_t = 256)]
    batch_size: usize,
}


fn main() {
    let Args { weights_path, config_path, dataset_configs, batch_size } = Args::parse();
    let device = general_models::wgpu::get_device();

    let model = ImageEncoder::<WgpuBackend, handwritten::Model<WgpuBackend>>::load(
        config_path,
        weights_path,
        device,
    )
    .expect("Configuration should be valid");

    let latents_size = model.get_encoder().linear.get_output_size();
    let mut batcher = AutoEncoderImageBatcher::new(1, device.clone());

    let conn = Connection::open("handwritten.sqlite").unwrap();
    let mut field_defs = String::new();
    let mut fields = String::new();

    for i in 0..latents_size {
        field_defs.push_str(", p");
        field_defs.push_str(&i.to_string());
        field_defs.push_str(" REAL NOT NULL");

        fields.push_str(", p");
        fields.push_str(&i.to_string());
    }

    conn.execute("DROP TABLE IF EXISTS latents", ()).unwrap();
    conn.execute(&format!("CREATE TABLE latents (row_id INTEGER PRIMARY KEY, brightness REAL NOT NULL{field_defs}, UNIQUE(brightness{fields}))"), ()).unwrap();
    field_defs.clear();
    let mut inputs = String::new();

    for i in 0..latents_size {
        field_defs.push_str(", p");
        field_defs.push_str(&i.to_string());
        inputs.push_str(", ?");
        inputs.push_str(&(i + 2).to_string());
    }

    let mut stmt = conn.prepare_cached(&format!("INSERT OR IGNORE INTO latents (brightness{field_defs}) VALUES (?1{inputs})")).unwrap();

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
}
