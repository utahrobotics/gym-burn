#[cfg(feature = "app")]
fn main() {
    use burn::backend::Autodiff;
    use burn::config::Config;
    use general_models::autoencoder::LinearImageAutoEncoderConfig;
    use general_training::{batches::AutoEncoderImageBatcher, dataset::{SqliteDataset, SqliteDatasetConfig}, training_loop::{simple_training_loop, SimpleTrainingConfig}};

    let device;
    
    #[cfg(feature = "wgpu")]
    type Backend = Autodiff<burn::backend::Wgpu>;
    
    #[cfg(feature = "wgpu")]
    {
        use burn::backend::wgpu::WgpuDevice;
    
        device = WgpuDevice::default();
    }
    
    let model_config = LinearImageAutoEncoderConfig::load("model.json").unwrap();
    let training_config = SimpleTrainingConfig::load("training.json").unwrap();
    let train_dataset_config = SqliteDatasetConfig::load("training-data.json").unwrap();
    let test_dataset_config = SqliteDatasetConfig::load("test-data.json").unwrap();
    
    let trained = simple_training_loop::<Backend, _, _, _, _, _, _, _>(
        model_config.init(&device),
        training_config,
        AutoEncoderImageBatcher,
        SqliteDataset::try_from(train_dataset_config).unwrap(),
        SqliteDataset::try_from(test_dataset_config).unwrap(),
        "artifacts",
        &device
    );
}

#[cfg(not(feature = "app"))]
fn main() {
    println!("The `app` feature needs to be enabled")
}
