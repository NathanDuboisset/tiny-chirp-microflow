"""Build firmware-side assets:

- raw audio + Hann + twiddle + mel matrix as .bin blobs in src/generated_data/
- sample_input_meta.h with sizes and quantization constants

Picks the same 4 testing clips collect_test_clips_for_rs would have:
non_target_1, target_1, non_target_2, target_2 (alphabetical within label).
Loads via librosa so we don't need tensorflow_io.
"""

import sys
from pathlib import Path

import librosa
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "building"))

from rust_export import write_sample_input_raw_c
from utils import SAMPLE_RATE


def _load(path, target_len):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size >= target_len:
        return audio[:target_len]
    out = np.zeros(target_len, dtype=np.float32)
    out[: audio.size] = audio
    return out


def main():
    from utils import TARGET_AUDIO_LEN_MEL

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
