use std::{io::{BufRead, stdin}, path::Path};

use base64::{Engine, prelude::BASE64_STANDARD};
use image::ImageFormat;
use rerun::RecordingStream;
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(untagged)]
enum Event {
    TrainingLoss {
        batch_i: usize,
        epoch: usize,
        loss: f64,
        lr: f64
    },
    ValidationLoss {
        batch_i: usize,
        epoch: usize,
        loss: f64
    },
    ChallengeImages {
        epoch: usize,
        challenge_images: Vec<(String, String)>
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let artifact_dir = std::env::var("ARTIFACT_DIR").unwrap();
    // let batch_count: usize = std::env::var("BATCH_COUNT").unwrap().parse().unwrap();
    let viz = rerun::RecordingStreamBuilder::new("proximo_rerun").spawn()?;
    let save = rerun::RecordingStreamBuilder::new("proximo_rerun").save(
        Path::new(&artifact_dir).join("training.rrd")
    )?;
    let mut stdin = stdin().lock();
    let mut buf = vec![];
    let mut iterations = 0i64;

    let mut record = |rec: &RecordingStream, event: &Event| {
        rec.set_time_sequence("iterations", iterations);
        match event {
            Event::TrainingLoss { batch_i, epoch, loss, lr } => {
                rec.set_time_sequence("batch_i", *batch_i as i64);
                rec.set_time_sequence("epoch", *epoch as i64);
                rec.log(
                    "training_loss",
                    &rerun::Scalars::single(*loss),
                ).unwrap();
                rec.log(
                    "learning_rate",
                    &rerun::Scalars::single(*lr),
                ).unwrap();
                rec.disable_timeline("batch_i");
            }
            Event::ValidationLoss { batch_i, epoch, loss } => {
                rec.set_time_sequence("batch_i", *batch_i as i64);
                rec.set_time_sequence("epoch", *epoch as i64);
                rec.log(
                    "validation_loss",
                    &rerun::Scalars::single(*loss),
                ).unwrap();
                rec.disable_timeline("batch_i");
            }
            Event::ChallengeImages { epoch, challenge_images } => {
                rec.set_time_sequence("epoch", *epoch as i64);
                for (i, (input, output)) in challenge_images.iter().enumerate() {
                    rec.set_time_sequence("challenge_index", i as i64);
                    let input = BASE64_STANDARD.decode(input).unwrap();
                    let output = BASE64_STANDARD.decode(output).unwrap();
                    rec.log(
                        "input",
                        &rerun::Image::from_image_bytes(ImageFormat::WebP, &input).unwrap(),
                    ).unwrap();
                    rec.log(
                        "output",
                        &rerun::Image::from_image_bytes(ImageFormat::WebP, &output).unwrap(),
                    ).unwrap();
                }
                rec.disable_timeline("challenge_index");
            }
        }
        iterations += 1;
    };

    while let Ok(_) = stdin.read_until(b'}', &mut buf) {
        if buf.last() != Some(&b'}') {
            break;
        }
        let event: Event = serde_json::from_slice(&buf).unwrap();
        record(&viz, &event);
        record(&save, &event);
        buf.clear();
    }

    Ok(())
}