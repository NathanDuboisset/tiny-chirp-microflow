#![no_main]
#![no_std]

use ariel_os::debug::{ExitCode, exit, log::info};
use ariel_os::time::Instant;
use microflow::model;
#[cfg(feature = "time")]
use nalgebra::SMatrix;


#[cfg(all(feature = "mel"))]
#[path = "../audio_samples/no_quantized.rs"]
mod audio_sample;
#[cfg(all(feature = "time", feature = "cnn_tf"))]
#[path = "../audio_samples/cnn_time_tf.rs"]
mod audio_sample;
#[cfg(all(feature = "time", feature = "sincnet_tf"))]
#[path = "../audio_samples/sincnet_tf.rs"]
mod audio_sample;
#[cfg(all(feature = "time", feature = "sincnet_multi_tf"))]
#[path = "../audio_samples/sincnet_multi_tf.rs"]
mod audio_sample;
#[cfg(all(feature = "time", feature = "sincnet_mimic_mel_tf"))]
#[path = "../audio_samples/sincnet_mimic_mel_tf.rs"]
mod audio_sample;
mod spectrogram;

#[cfg(all(feature = "mel", feature = "cnn_tf"))]
#[model("models/cnn_mel_tf.tflite", enable_timing = true)]
struct BirdModel;

#[cfg(all(feature = "mel", feature = "cnn_torch"))]
#[model("models/cnn_mel_torch.tflite", enable_timing = true)]
struct BirdModel;

#[cfg(all(feature = "time", feature = "cnn_tf"))]
#[model("models/cnn_time_tf.tflite", enable_kernel_streaming = true, enable_timing = true)]
struct BirdModel;

#[cfg(all(feature = "time", feature = "sincnet_tf"))]
#[model("models/sincnet_tf.tflite", enable_timing = true)]
struct BirdModel;

#[cfg(all(feature = "time", feature = "sincnet_multi_tf"))]
#[model("models/sincnet_multi_tf.tflite", enable_timing = true)]
struct BirdModel;

#[cfg(all(feature = "time", feature = "sincnet_mimic_mel_tf"))]
#[model("models/sincnet_mimic_mel_tf.tflite", enable_timing = true)]
struct BirdModel;

pub const AUDIO_LEN: usize = (spectrogram::FRAMES - 1) * spectrogram::FRAME_STEP + spectrogram::FRAME_LENGTH;

#[cfg(feature = "time")]
fn prepare_rank2_input<const IN_L: usize>(clip_input: &[i8]) -> SMatrix<i8, 1, IN_L> {
    SMatrix::<i8, 1, IN_L>::from_row_slice(clip_input)
}

#[cfg(feature = "time")]
fn prepare_rank4_input<const IN_H: usize, const IN_W: usize, const IN_C: usize>(
    clip_input: &[i8],
) -> [SMatrix<[i8; IN_C], IN_H, IN_W>; 1] {
    // SAFETY: [i8; IN_C] has the same layout as IN_C contiguous i8 values.
    let raw: &[[i8; IN_C]] = unsafe {
        core::slice::from_raw_parts(clip_input.as_ptr() as *const [i8; IN_C], IN_H * IN_W)
    };
    [SMatrix::<[i8; IN_C], IN_H, IN_W>::from_column_slice(raw)]
}

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
    #[cfg(feature = "sincnet_multi_tf")]
    info!("Model: sincnet_multi (tensorflow)");
    #[cfg(feature = "sincnet_mimic_mel_tf")]
    info!("Model: sincnet_mimic_mel (tensorflow)");
    info!(
        "Audio sample rate: {} Hz, clips={}",
        audio_sample::SAMPLE_RATE,
        audio_sample::TEST_CLIPS.len()
    );

    info!("Starting inference");
    info!("clip | expected   | predicted  | score0 | score1");
    let mut total_inference_us: u64 = 0;
    let mut total_preparation_us: u64 = 0;

    for (i, clip) in audio_sample::TEST_CLIPS.iter().enumerate() {
        

        #[cfg(feature = "mel")]
        let (prediction, start_time,preparation_time) = {
            let preparation_start_time = Instant::now().as_micros();
            let input = spectrogram::compute(clip.audio);
            let start_time = Instant::now().as_micros();
            (BirdModel::predict(input), start_time, start_time - preparation_start_time)
        };
        #[cfg(all(feature = "time", feature = "cnn_tf"))]
        let (prediction, start_time,preparation_time) = {
            let preparation_start_time = Instant::now().as_micros();
            const SHAPE: [usize; 2] = BirdModel::expose_input().2;
            const IN_L: usize = SHAPE[1];
            let input = prepare_rank2_input::<IN_L>(clip.input);
            let start_time = Instant::now().as_micros();
            (BirdModel::predict_quantized(input), start_time, start_time - preparation_start_time)
        };
        #[cfg(all(
            feature = "time",
            any(
                feature = "sincnet_tf",
                feature = "sincnet_multi_tf",
                feature = "sincnet_mimic_mel_tf"
            )
        ))]
        let (prediction, start_time,preparation_time) = {
            let preparation_start_time = Instant::now().as_micros();
            const SHAPE: [usize; 4] = BirdModel::expose_input().2;
            const IN_H: usize = SHAPE[1];
            const IN_W: usize = SHAPE[2];
            const IN_C: usize = SHAPE[3];
            let input = prepare_rank4_input::<IN_H, IN_W, IN_C>(clip.input);
            let start_time = Instant::now().as_micros();
            (BirdModel::predict_quantized(input), start_time, start_time - preparation_start_time)
        };

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
            "target    "
        } else {
            "non_target"
        };
        let end_time = Instant::now().as_micros();
        let duration = end_time - start_time;
        total_inference_us += duration;
        total_preparation_us += preparation_time;
        let score0_deci = (prediction[(0, 0)] * 10.0 + 0.5) as i32;
        let score1_deci = (prediction[(0, 1)] * 10.0 + 0.5) as i32;
        let score0_sign = if score0_deci < 0 { "-" } else { "" };
        let score1_sign = if score1_deci < 0 { "-" } else { "" };
        let score0_abs = if score0_deci < 0 { -score0_deci } else { score0_deci };
        let score1_abs = if score1_deci < 0 { -score1_deci } else { score1_deci };
        info!(
            "{}    | {} | {} | {}{}.{} | {}{}.{}",
            i,
            clip.expected_label,
            predicted_label,
            score0_sign,
            score0_abs / 10,
            score0_abs % 10,
            score1_sign,
            score1_abs / 10,
            score1_abs % 10
        );
    }
    let clip_count = audio_sample::TEST_CLIPS.len() as u64;
    if clip_count > 0 {
        let avg_inference_us = (total_inference_us + clip_count / 2) / clip_count;
        let avg_preparation_us = (total_preparation_us + clip_count / 2) / clip_count;
        let total_inference_ms = (total_inference_us + 500) / 1000;
        let total_preparation_ms = (total_preparation_us + 500) / 1000;
        let avg_inference_ms = (avg_inference_us + 500) / 1000;
        let avg_preparation_ms = (avg_preparation_us + 500) / 1000;
        info!(
            "timing avg: infer={} ms, prep={} ms | totals: infer={} ms, prep={} ms",
            avg_inference_ms,
            avg_preparation_ms,
            total_inference_ms,
            total_preparation_ms
        );
    }
    exit(ExitCode::SUCCESS);
}
