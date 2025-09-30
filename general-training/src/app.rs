use clap::{Parser, Subcommand};
use rusqlite::{Connection, params};
use std::io::Write;
use std::sync::Arc;

use crate::batches::AutoEncoderImageItem;
use crate::regression::RegressionTrainableModel;
use crate::training_loop::simple_regression_training_loop;
use crate::{
    batches::AutoEncoderImageBatcher,
    dataset::{SqliteDataset, SqliteDatasetConfig},
    training_loop::SimpleTrainingConfig,
};
use burn::backend::Autodiff;
use burn::config::Config;
use burn::module::Module;
use burn::record::CompactRecorder;
use general_models::autoencoder::LinearImageAutoEncoderConfig;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Train,
    ClearDb {
        #[arg(short, long)]
        db_path: String,
    },
}

#[derive(Debug, Config)]
struct ArtifactConfig {
    artifact_dir: String,
}

fn train() {
    #[cfg(feature = "wgpu")]
    type Backend = burn::backend::Wgpu;

    type AutodiffBackend = Autodiff<Backend>;

    #[cfg(feature = "wgpu")]
    let device = general_models::wgpu::get_device();

    let model_config = LinearImageAutoEncoderConfig::load("model.json").unwrap();
    let training_config = SimpleTrainingConfig::load("training.json").unwrap();
    let artifact_config = ArtifactConfig::load("training.json").unwrap();
    let train_dataset_config = SqliteDatasetConfig::load("training-data.json").unwrap();
    let test_dataset_config = SqliteDatasetConfig::load("test-data.json").unwrap();

    std::fs::create_dir_all(&artifact_config.artifact_dir).unwrap();

    let trained = simple_regression_training_loop::<
        AutodiffBackend,
        RegressionTrainableModel<_>,
        _,
        _,
        _,
        _,
        _,
        _,
    >(
        model_config.init::<_, 3, 2, _, _>(device).into(),
        training_config,
        AutoEncoderImageBatcher,
        SqliteDataset::<Arc<AutoEncoderImageItem>, _>::try_from(train_dataset_config).unwrap(),
        SqliteDataset::<Arc<AutoEncoderImageItem>, _>::try_from(test_dataset_config).unwrap(),
        &artifact_config.artifact_dir,
        &device,
    );

    trained
        .model
        .save_file("model.mpk", &CompactRecorder::new())
        .unwrap();
}

fn clear_db(db_path: String) {
    let conn = Connection::open(db_path).expect("SQLite database should be accessible");

    let mut stmt = conn
        .prepare("SELECT name FROM sqlite_master WHERE type='table'")
        .unwrap();
    let mut delete_stmt = conn.prepare("DROP TABLE ?1").unwrap();
    loop {
        let mut rows = stmt.query(()).unwrap();
        let mut had_something = false;
        while let Some(row) = rows.next().unwrap() {
            had_something = true;
            let name: String = row.get("name").unwrap();
            print!("Deleting {name}: ");
            std::io::stdout().flush().unwrap();
            let n = delete_stmt.execute(params![name]).unwrap();
            println!("{n}");
        }
        if !had_something {
            break;
        }
    }
}

pub fn main() {
    match Args::parse().command {
        Command::Train => train(),
        Command::ClearDb { db_path } => clear_db(db_path),
    }
}
