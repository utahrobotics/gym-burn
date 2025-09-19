use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
    tensor::{ElementConversion, Tensor, TensorData},
};
use image::{ImageBuffer, Luma};
use std::fs;
use std::path::{Path, PathBuf};

/// Represents a single handwritten character item from the dataset
#[derive(Clone, Debug)]
pub struct HandwrittenItem {
    /// Path to the image file
    pub image_path: PathBuf,
    /// ASCII value of the character (e.g., 48 for '0')
    pub ascii_value: u8,
    /// The actual character (e.g., '0')
    pub character: char,
}

impl HandwrittenItem {
    pub fn new(image_path: PathBuf, ascii_value: u8) -> Self {
        let character = ascii_value as char;
        Self {
            image_path,
            ascii_value,
            character,
        }
    }
}

/// Represents a batch of handwritten character items
#[derive(Clone, Debug)]
pub struct HandwrittenBatch<B: Backend> {
    /// Batch of images with shape [batch_size, height, width]
    pub images: Tensor<B, 3>,
    /// Batch of ASCII values as targets with shape [batch_size]
    pub targets: Tensor<B, 1, Int>,
}

/// Load and preprocess an image from the given path
fn load_image(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let img = image::open(path)?;
    let img = img.to_luma8(); // Convert to grayscale
    let (width, height) = img.dimensions();
    
    // Convert image to normalized f32 values [0, 1]
    let pixels: Vec<f32> = img
        .pixels()
        .map(|p| p[0] as f32 / 255.0)
        .collect();
    
    Ok(pixels)
}

/// Batcher for handwritten character recognition
#[derive(Clone, Default)]
pub struct HandwrittenBatcher {
    /// Target image height
    pub image_height: usize,
    /// Target image width  
    pub image_width: usize,
}

impl HandwrittenBatcher {
    pub fn new(image_height: usize, image_width: usize) -> Self {
        Self {
            image_height,
            image_width,
        }
    }
}

impl<B: Backend> Batcher<B, HandwrittenItem, HandwrittenBatch<B>> for HandwrittenBatcher {
    fn batch(&self, items: Vec<HandwrittenItem>, device: &B::Device) -> HandwrittenBatch<B> {
        let images = items
            .iter()
            .map(|item| {
                // Load image and handle errors by using zeros if loading fails
                let pixels = load_image(&item.image_path)
                    .unwrap_or_else(|_| vec![0.0; self.image_height * self.image_width]);
                
                // Ensure the image has the correct dimensions
                let mut resized_pixels = vec![0.0f32; self.image_height * self.image_width];
                let len = pixels.len().min(resized_pixels.len());
                resized_pixels[..len].copy_from_slice(&pixels[..len]);
                
                TensorData::from(resized_pixels).convert::<B::FloatElem>()
            })
            .map(|data| Tensor::<B, 2>::from_data(data.reshape([self.image_height, self.image_width]), device))
            .map(|tensor| tensor.reshape([1, self.image_height, self.image_width]))
            // Normalize: make mean close to 0 and std close to 1
            // For handwritten characters, we'll use simple normalization
            .map(|tensor| (tensor - 0.5) * 2.0) // Scale from [0,1] to [-1,1]
            .collect();

        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data(
                    [item.ascii_value.elem::<B::IntElem>()], 
                    device
                )
            })
            .collect();

        let images = Tensor::cat(images, 0);
        let targets = Tensor::cat(targets, 0);

        HandwrittenBatch { images, targets }
    }
}

/// Dataset for handwritten character recognition
pub struct HandwrittenDataset {
    items: Vec<HandwrittenItem>,
}

impl HandwrittenDataset {
    /// Create a new dataset from the given directory path
    /// The directory should contain subdirectories named by ASCII values (e.g., "48", "49", etc.)
    /// Each subdirectory contains image files for that character
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Result<Self, Box<dyn std::error::Error>> {
        let mut items = Vec::new();
        let dataset_path = dataset_root.as_ref();
        
        // Read all subdirectories (should be ASCII values)
        for entry in fs::read_dir(dataset_path)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_dir() {
                // Try to parse directory name as ASCII value
                if let Some(dir_name) = path.file_name().and_then(|n| n.to_str()) {
                    if let Ok(ascii_value) = dir_name.parse::<u8>() {
                        // Read all image files in this directory
                        for image_entry in fs::read_dir(&path)? {
                            let image_entry = image_entry?;
                            let image_path = image_entry.path();
                            
                            // Check if it's an image file (by extension)
                            if image_path.is_file() {
                                if let Some(ext) = image_path.extension().and_then(|e| e.to_str()) {
                                    match ext.to_lowercase().as_str() {
                                        "jpg" | "jpeg" | "png" | "bmp" | "gif" => {
                                            items.push(HandwrittenItem::new(image_path, ascii_value));
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        println!("Loaded {} images from dataset", items.len());
        Ok(Self { items })
    }
    
    /// Create a dataset for training split
    pub fn train<P: AsRef<Path>>(dataset_root: P) -> Result<Self, Box<dyn std::error::Error>> {
        let train_path = dataset_root.as_ref().join("train");
        Self::new(train_path)
    }
    
    /// Create a dataset for validation split  
    pub fn validation<P: AsRef<Path>>(dataset_root: P) -> Result<Self, Box<dyn std::error::Error>> {
        let validation_path = dataset_root.as_ref().join("validation");
        Self::new(validation_path)
    }
    
    /// Create a dataset for test split
    pub fn test<P: AsRef<Path>>(dataset_root: P) -> Result<Self, Box<dyn std::error::Error>> {
        let test_path = dataset_root.as_ref().join("test");
        Self::new(test_path)
    }
}

impl Dataset<HandwrittenItem> for HandwrittenDataset {
    fn get(&self, index: usize) -> Option<HandwrittenItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}