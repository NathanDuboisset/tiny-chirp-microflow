"""Build firmware-side assets for the cnn_mel path.

Emits into src/nrf/generated_data/:
  - sample_audio_{0..3}.bin    int16 PCM, MEL_AUDIO_LEN samples each
  - sample_mel_{0..3}.bin      int8 log-mel reference, baked from the TFLite
  - hann_{q15,even_q15,odd_q15,f32}.bin, twiddle_f32.bin, mel_matrix_q15.bin
  - sample_input_meta.h        MEL_* sizes + input quant constants + labels

Clip selection mirrors collect_test_clips_for_rs():
non_target_1, target_1, non_target_2, target_2 (alphabetical within label).
Audio is read via librosa to avoid the tensorflow_io dep.

Run:  make gen_assets
"""

import sys
from pathlib import Path

import librosa
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "building"))

from rust_export import write_sample_input_raw_c
from utils import SAMPLE_RATE, TARGET_AUDIO_LEN_MEL


def _load(path, target_len):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size >= target_len:
        return audio[:target_len]
    out = np.zeros(target_len, dtype=np.float32)
    out[: audio.size] = audio
    return out


def main():
    testing = REPO / "dataset" / "testing"
    nt = sorted(p for p in (testing / "non_target").iterdir() if p.suffix == ".wav")[:2]
    tg = sorted(p for p in (testing / "target").iterdir()     if p.suffix == ".wav")[:2]

    picks = [
        ("non_target", nt[0]),
        ("target",     tg[0]),
        ("non_target", nt[1]),
        ("target",     tg[1]),
    ]
    clips = []
    for i, (label, wav) in enumerate(picks):
        audio = _load(wav, TARGET_AUDIO_LEN_MEL)
        clips.append((label, audio, f"dataset/testing/{label}/{wav.name}"))
        print(f"  clip {i}  {label:<10}  {wav.name}")

    write_sample_input_raw_c(
        src_dir=REPO / "src" / "nrf",
        clips=clips,
        tflite_path=REPO / "models" / "cnn_mel_tf.tflite",
        generator_name="scripts/gen_assets.py",
    )
    print(f"wrote {len(clips)} clips to src/nrf/generated_data/")


if __name__ == "__main__":
    main()
