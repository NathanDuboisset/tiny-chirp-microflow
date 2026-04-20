#![no_main]
#![no_std]

use ariel_os::debug::{exit, log::info, ExitCode};
use ariel_os::time::Instant;
use microflow::model;

mod audio_sample;
mod spectrogram;
mod audio_raw;

#[cfg(all(feature = "mel", feature = "cnn_tf"))]
#[model("models/cnn_mel_tf.tflite")]
struct BirdModel;

#[cfg(all(feature = "mel", feature = "cnn_torch"))]
#[model("models/cnn_mel_torch.tflite")]
struct BirdModel;

#[cfg(all(feature = "time", feature = "cnn_tf"))]
#[model("models/cnn_time_tf.tflite")]
struct BirdModel;

#[cfg(all(feature = "time", feature = "sincnet_tf"))]
#[model("models/sincnet_tf.tflite")]
struct BirdModel;

#[cfg(all(feature = "time", feature = "sincnet_real_tf"))]
#[model("models/sincnet_real_tf.tflite")]
struct BirdModel;

#[cfg(all(feature = "time", feature = "sincnet_real_multilayer_tf"))]
#[model("models/sincnet_real_multilayer_tf.tflite")]
struct BirdModel;

#[ariel_os::task(autostart)]
async fn main() {
    info!("tiny-chirp-microflow on {}.", ariel_os::buildinfo::BOARD);
    #[cfg(feature = "mel")]
    info!("Input: mel spectrogram");
    #[cfg(feature = "time")]
    info!("Input: time domain");
    #[cfg(feature = "cnn_tf")]
    info!("Model: cnn (tensorflow)");
    #[cfg(feature = "cnn_torch")]
    info!("Model: cnn (torch)");
    #[cfg(feature = "sincnet_tf")]
    info!("Model: sincnet (tensorflow)");
    #[cfg(feature = "sincnet_real_tf")]
    info!("Model: sincnet_real (tensorflow)");
    #[cfg(feature = "sincnet_real_multilayer_tf")]
    info!("Model: sincnet_real_multilayer (tensorflow)");
    info!(
        "Audio sample rate: {} Hz, clips={}",
        audio_sample::SAMPLE_RATE,
        audio_sample::TEST_CLIPS.len()
    );

    info!("Starting inference");
    info!("clip | expected   | predicted  | score0 | score1");

    for (i, clip) in audio_sample::TEST_CLIPS.iter().enumerate() {
        #[cfg(feature = "mel")]
        let input = spectrogram::compute(clip.audio);
        #[cfg(feature = "time")]
        let input = audio_raw::prepare(clip.audio);

        let start_time = Instant::now().as_micros();

        let prediction = BirdModel::predict(input);

        let mut predicted_class: usize = 0;
        let mut best_val = prediction[(0, 0)];
        let mut c = 1usize;
        while c < prediction.ncols() {
            let v = prediction[(0, c)];
            if v > best_val {
                best_val = v;
                predicted_class = c;
            }
            c += 1;
        }

        let predicted_label = if predicted_class == 1 {
            "target----"
        } else {
            "non_target"
        };
        let end_time = Instant::now().as_micros();
        let duration = end_time - start_time;
        info!(
            "{} | {} | {} | {} | {} in {} us",
            i,
            clip.expected_label,
            predicted_label,
            prediction[(0, 0)],
            prediction[(0, 1)],
            duration
        );
    }
    exit(ExitCode::SUCCESS);
}
