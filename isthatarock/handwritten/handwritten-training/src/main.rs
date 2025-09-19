mod data;

use burn::{
    backend::{Autodiff, NdArray},
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    prelude::*,
};
use data::{HandwrittenBatcher, HandwrittenDataset};

type Backend = NdArray<f32>;
type AutodiffBackend = Autodiff<Backend>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set up the device 
    let device = Default::default();
    
    // Path to the dataset (relative to the training crate)
    let dataset_path = "../../dataset";
    
    // Load the training dataset
    let train_dataset = HandwrittenDataset::train(dataset_path)?;
    println!("Training dataset size: {}", train_dataset.len());
    
    // Create a batcher with target image dimensions
    // Most handwritten character images should be relatively small
    let batcher = HandwrittenBatcher::new(28, 28); // Similar to MNIST dimensions
    
    // Create a data loader
    let dataloader = DataLoaderBuilder::new(batcher)
        .batch_size(32)
        .shuffle(42) // Seed for reproducibility
        .num_workers(4)
        .build(train_dataset);
    
    // Demonstrate loading a few batches
    let mut batch_count = 0;
    for batch in dataloader.iter(&device) {
        println!(
            "Batch {}: images shape {:?}, targets shape {:?}",
            batch_count,
            batch.images.shape(),
            batch.targets.shape()
        );
        
        // Print some sample target values (ASCII codes)
        let target_data = batch.targets.clone().into_data();
        let targets_vec: Vec<i32> = target_data.convert::<i32>().value;
        println!("Sample targets (ASCII values): {:?}", &targets_vec[..5.min(targets_vec.len())]);
        
        // Convert ASCII values to characters for display
        let chars: Vec<char> = targets_vec[..5.min(targets_vec.len())]
            .iter()
            .map(|&ascii| ascii as u8 as char)
            .collect();
        println!("Sample characters: {:?}", chars);
        
        batch_count += 1;
        
        // Only process first few batches for demonstration
        if batch_count >= 3 {
            break;
        }
    }
    
    println!("Handwritten character batcher implementation complete!");
    Ok(())
}
