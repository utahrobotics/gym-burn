use std::path::Path;

pub use efficient_pca::PCA;
use ndarray::{Array1, Array2};
use serde::Deserialize;
use serde_json::json;
use utils::parse_json_file;

#[derive(Deserialize)]
struct PcaJson {
    mean: Array1<f64>,
    components: Array2<f64>,
    scale: Array1<f64>,
}

pub fn save_pca(pca: &PCA, file: impl AsRef<Path>) -> serde_json::Result<()> {
    serde_json::to_writer_pretty(
        std::fs::File::create(file).map_err(|e| serde_json::Error::io(e))?,
        &json!({
            "mean": pca.mean().unwrap(),
            "components": pca.rotation().unwrap(),
            "scale": pca.scale().unwrap()
        }),
    )
}

pub fn load_pca(file: impl AsRef<Path>) -> serde_json::Result<PCA> {
    let pca_json: PcaJson =
        parse_json_file(file)?;

    let mut pca = PCA::new();
    pca.mean = Some(pca_json.mean);
    pca.rotation = Some(pca_json.components);
    pca.scale = Some(pca_json.scale);
    Ok(pca)
}
