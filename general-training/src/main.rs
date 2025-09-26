#[cfg(feature = "app")]
fn main() {
    use burn::module::Module;
    use burn::record::CompactRecorder;
    use burn::backend::Autodiff;
    use burn::config::Config;
    use general_models::autoencoder::LinearImageAutoEncoderConfig;
    use general_training::regression::RegressionTrainableModel;
    use general_training::training_loop::simple_regression_training_loop;
    use general_training::{batches::AutoEncoderImageBatcher, dataset::{SqliteDataset, SqliteDatasetConfig}, training_loop::SimpleTrainingConfig};

    #[cfg(feature = "wgpu")]
    type Backend = burn::backend::Wgpu;
    
    type AutodiffBackend = Autodiff<Backend>;
    
    #[cfg(feature = "wgpu")]
    let device = general_models::wgpu::get_device();
    
    let model_config = LinearImageAutoEncoderConfig::load("model.json").unwrap();
    let training_config = SimpleTrainingConfig::load("training.json").unwrap();
    let train_dataset_config = SqliteDatasetConfig::load("training-data.json").unwrap();
    let test_dataset_config = SqliteDatasetConfig::load("test-data.json").unwrap();
    
    let trained = simple_regression_training_loop::<
        AutodiffBackend,
        RegressionTrainableModel<_>,
        _,
        _,
        _,
        _,
        _,
        _
    >(
        model_config.init::<_, 3, 2, _, _>(device).into(),
        training_config,
        AutoEncoderImageBatcher,
        SqliteDataset::try_from(train_dataset_config).unwrap(),
        SqliteDataset::try_from(test_dataset_config).unwrap(),
        "artifacts",
        &device
    );
    
    trained.model.save_file("model.mpk", &CompactRecorder::new()).unwrap();
}

#[cfg(not(feature = "app"))]
fn main() {
    println!("The `app` feature needs to be enabled")
}
