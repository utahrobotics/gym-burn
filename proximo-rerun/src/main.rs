use std::{
    io::{BufRead, stdin},
    path::Path,
};

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
        loss: Option<f64>,
        lr: f64,
    },
    ValidationLoss {
        batch_i: usize,
        epoch: usize,
        loss: f64,
    },
    ChallengeImages {
        epoch: usize,
        challenge_images: Vec<(String, String)>,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let artifact_dir = std::env::var("ARTIFACT_DIR").unwrap();
    // let batch_count: usize = std::env::var("BATCH_COUNT").unwrap().parse().unwrap();
    let viz = rerun::RecordingStreamBuilder::new("proximo_rerun").spawn()?;
    let save = rerun::RecordingStreamBuilder::new("proximo_rerun")
        .save(Path::new(&artifact_dir).join("training.rrd"))?;
    let mut stdin = stdin().lock();
    let mut buf = vec![];
    let mut iterations = 0i64;
    let mut found_nan = false;

    let mut record = |rec: &RecordingStream, event: &Event| {
        rec.set_time_sequence("iterations", iterations);
        match event {
            Event::TrainingLoss {
                batch_i,
                epoch,
                loss,
                lr,
            } => {
                let Some(loss) = loss else {
                    if !found_nan {
                        found_nan = true;
                        eprintln!("Encountered NaN in training loss");
                    }
                    return;
                };
                rec.set_time_sequence("batch_i", *batch_i as i64);
                rec.set_time_sequence("epoch", *epoch as i64);
                rec.log("training_loss", &rerun::Scalars::single(*loss))
                    .unwrap();
                rec.log("learning_rate", &rerun::Scalars::single(*lr))
                    .unwrap();
                rec.disable_timeline("batch_i");
            }
            Event::ValidationLoss {
                batch_i,
                epoch,
                loss,
            } => {
                rec.set_time_sequence("batch_i", *batch_i as i64);
                rec.set_time_sequence("epoch", *epoch as i64);
                rec.log("validation_loss", &rerun::Scalars::single(*loss))
                    .unwrap();
                rec.disable_timeline("batch_i");
            }
            Event::ChallengeImages {
                epoch,
                challenge_images,
            } => {
                rec.set_time_sequence("epoch", *epoch as i64);
                for (input, output) in challenge_images.iter() {
                    rec.set_time_sequence("iterations", iterations);
                    let input = BASE64_STANDARD.decode(input).unwrap();
                    let output = BASE64_STANDARD.decode(output).unwrap();
                    rec.log(
                        "input",
                        &rerun::Image::from_image_bytes(ImageFormat::WebP, &input).unwrap(),
                    )
                    .unwrap();
                    rec.log(
                        "output",
                        &rerun::Image::from_image_bytes(ImageFormat::WebP, &output).unwrap(),
                    )
                    .unwrap();
                    iterations += 1;
                }
            }
        }
        iterations += 1;
    };

    ctrlc::set_handler(move || {
        std::thread::spawn(|| {
            std::thread::sleep(std::time::Duration::from_secs(2));
            std::process::exit(0);
        });
    })
    .expect("Error setting Ctrl-C handler");

    while let Ok(_) = stdin.read_until(b'}', &mut buf) {
        if buf.last() != Some(&b'}') {
            break;
        }
        // println!("{}", String::from_utf8_lossy(&buf));
        let event: Event = match serde_json::from_slice(&buf) {
            Ok(x) => x,
            Err(e) => {
                panic!("Payload: {:?}\n{e}", String::from_utf8_lossy(&buf));
            }
        };
        record(&viz, &event);
        record(&save, &event);
        buf.clear();
    }

    Ok(())
}
