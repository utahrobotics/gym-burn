use std::sync::Arc;

use burn::{prelude::*, record::CompactRecorder, tensor::backend::AutodiffBackend};
use general_models::{
    Init,
    composite::{
        autoencoder::{
            AutoEncoderModel, AutoEncoderModelConfig,
            vae::{VariationalEncoder, VariationalEncoderConfig},
        },
        image::{
            ConvLinearModel, ConvLinearModelConfig, LinearConvTransposedModelConfig,
            LinearConvTransposedModel,
        },
    },
};
use utils::parse_json_file;

use crate::{
    app::ArtifactConfig,
    batches::{AutoEncoderImageBatcher, AutoEncoderImageItem},
    dataset::{SqliteDataset, SqliteDatasetConfig},
    regression::{RegressionTrainableModel, SPECIALIZED, STANDARD},
    training_loop::{simple_regression_training_loop, SimpleTrainingConfig},
};

macro_rules! epilogue {
    ($artifact_config: ident, $training_dataset: ident, $test_dataset: ident, $trained: ident) => {{
        println!(
            "Training Cache Performance: {} / {}",
            $training_dataset.get_cache_hits(),
            $training_dataset.get_reads()
        );
        println!(
            "Testing Cache Performance:  {} / {}",
            $test_dataset.get_cache_hits(),
            $test_dataset.get_reads()
        );
        $trained
            .model
            .save_file(
                $artifact_config.artifact_dir.join("model.mpk"),
                &CompactRecorder::new(),
            )
            .unwrap();
    }};
}

pub fn train_image_autoencoder<B: AutodiffBackend>(
    train_dataset_config: SqliteDatasetConfig,
    test_dataset_config: SqliteDatasetConfig,
    training_config: SimpleTrainingConfig,
    artifact_config: ArtifactConfig,
    device: &B::Device,
) {
    let model_config: AutoEncoderModelConfig<ConvLinearModelConfig, LinearConvTransposedModelConfig> =
        parse_json_file("model").unwrap();
    let channels = model_config.encoder.conv.input_channels;
    let training_dataset = Arc::new(
        SqliteDataset::<Arc<AutoEncoderImageItem>, _>::try_from(train_dataset_config).unwrap(),
    );
    let test_dataset = Arc::new(
        SqliteDataset::<Arc<AutoEncoderImageItem>, _>::try_from(test_dataset_config).unwrap(),
    );

    let trained = simple_regression_training_loop::<
        B,
        RegressionTrainableModel<
            _,
            AutoEncoderModel<_, ConvLinearModel<_>, LinearConvTransposedModel<_>>,
            STANDARD
        >,
        _,
        _,
        _,
    >(
        model_config.init(device).into(),
        training_config,
        AutoEncoderImageBatcher { channels },
        training_dataset.clone(),
        test_dataset.clone(),
        &artifact_config.artifact_dir,
        &device,
    );

    trained
        .model
        .encoder
        .clone()
        .save_file(
            artifact_config.artifact_dir.join("encoder.mpk"),
            &CompactRecorder::new(),
        )
        .unwrap();
    trained
        .model
        .decoder
        .clone()
        .save_file(
            artifact_config.artifact_dir.join("decoder.mpk"),
            &CompactRecorder::new(),
        )
        .unwrap();
    epilogue!(artifact_config, training_dataset, test_dataset, trained);
}

pub fn train_image_v_autoencoder<B: AutodiffBackend>(
    train_dataset_config: SqliteDatasetConfig,
    test_dataset_config: SqliteDatasetConfig,
    training_config: SimpleTrainingConfig,
    artifact_config: ArtifactConfig,
    device: &B::Device,
) {
    let model_config: AutoEncoderModelConfig<
        VariationalEncoderConfig<ConvLinearModelConfig>,
        LinearConvTransposedModelConfig,
    > = parse_json_file("model").unwrap();
    let channels = model_config.encoder.model.conv.input_channels;
    let training_dataset = Arc::new(
        SqliteDataset::<Arc<AutoEncoderImageItem>, _>::try_from(train_dataset_config).unwrap(),
    );
    let test_dataset = Arc::new(
        SqliteDataset::<Arc<AutoEncoderImageItem>, _>::try_from(test_dataset_config).unwrap(),
    );

    let trained = simple_regression_training_loop::<
        B,
        RegressionTrainableModel<
            _,
            AutoEncoderModel<
                _,
                VariationalEncoder<_, ConvLinearModel<_>>,
                LinearConvTransposedModel<_>,
            >,
            SPECIALIZED
        >,
        _,
        _,
        _,
    >(
        model_config.init(device).into(),
        training_config,
        AutoEncoderImageBatcher { channels },
        training_dataset.clone(),
        test_dataset.clone(),
        &artifact_config.artifact_dir,
        &device,
    );

    trained
        .model
        .encoder
        .clone()
        .save_file(
            artifact_config.artifact_dir.join("encoder.mpk"),
            &CompactRecorder::new(),
        )
        .unwrap();
    trained
        .model
        .decoder
        .clone()
        .save_file(
            artifact_config.artifact_dir.join("decoder.mpk"),
            &CompactRecorder::new(),
        )
        .unwrap();
    epilogue!(artifact_config, training_dataset, test_dataset, trained);
}
