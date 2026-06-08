"""Build firmware-side assets for the sincnet_chunked path.

Emits into src/nrf/generated_data/:
  - sample_sincnet_audio_{0..3}.bin   int8-quantized raw audio, cropped to
                                      NUM_CHUNKS * CHUNK_SIZE samples and laid
                                      out as the (1, 46, 1024, 1) NHWC input.
  - sample_input_sincnet_meta.h       sizes + quant constants + labels.

Clip selection matches gen_assets.py (non_target_1, target_1, non_target_2,
target_2). The TFLite input quant (scale, zp) is read directly from
models/sincnet_chunked.tflite, so the firmware sees exactly what the trained
model expects.

Run:  make gen_assets_sincnet
"""

import sys
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "building"))
from utils import SAMPLE_RATE  # noqa: E402

# Geometry mirrors the sincnet_chunked.ipynb training notebook.
CHUNK_SIZE = 1024
NUM_CHUNKS = 46
TOTAL_LEN = NUM_CHUNKS * CHUNK_SIZE

TFLITE = REPO / "models" / "sincnet_chunked.tflite"
SRC = REPO / "src" / "nrf"
DATA = SRC / "generated_data"


def load_wav(path: Path, n: int) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size >= n:
        return audio[:n]
    out = np.zeros(n, dtype=np.float32)
    out[: audio.size] = audio
    return out


def main():
    assert TFLITE.exists(), f"missing {TFLITE} — run building/sincnet_chunked.ipynb first"

    testing = REPO / "dataset" / "testing"
    nt = sorted(p for p in (testing / "non_target").iterdir() if p.suffix == ".wav")[:2]
    tg = sorted(p for p in (testing / "target").iterdir()     if p.suffix == ".wav")[:2]
    picks = [
        ("non_target", nt[0]),
        ("target",     tg[0]),
        ("non_target", nt[1]),
        ("target",     tg[1]),
    ]

    interp = tf.lite.Interpreter(model_path=str(TFLITE))
    interp.allocate_tensors()
    in_scale, in_zp = interp.get_input_details()[0]["quantization"]
    if in_scale == 0:
        raise RuntimeError("TFLite input not int8-quantized.")
    print(f"input quant: scale={in_scale:.10g}  zp={int(in_zp)}")

    DATA.mkdir(parents=True, exist_ok=True)
    labels = []
    for i, (label, wav) in enumerate(picks):
        audio = load_wav(wav, TOTAL_LEN)
        q = np.clip(np.round(audio / in_scale) + in_zp, -128, 127).astype(np.int8)
        out_bin = DATA / f"sample_sincnet_audio_{i}.bin"
        out_bin.write_bytes(q.tobytes())
        print(f"  clip {i}  {label:<10}  {wav.name}  -> {out_bin.name} ({q.size} B)")
        labels.append(label.strip())

    label_defines = "\n".join(
        f'#define SINCNET_CLIP_{i}_LABEL  "{lbl}"' for i, lbl in enumerate(labels)
    )
    meta = (
        "#pragma once\n\n"
        f"#define SINCNET_SAMPLE_RATE        {SAMPLE_RATE}\n"
        f"#define SINCNET_NUM_CHUNKS         {NUM_CHUNKS}\n"
        f"#define SINCNET_CHUNK_SIZE         {CHUNK_SIZE}\n"
        f"#define SINCNET_AUDIO_LEN          {TOTAL_LEN}\n\n"
        f"#define SINCNET_INPUT_QUANT_SCALE  ({float(in_scale):.10g}f)\n"
        f"#define SINCNET_INPUT_QUANT_ZP     {int(in_zp)}\n\n"
        f"#define N_SINCNET_CLIPS            {len(labels)}\n"
        f"{label_defines}\n"
    )
    (DATA / "sample_input_sincnet_meta.h").write_text(meta)
    print(f"wrote {DATA / 'sample_input_sincnet_meta.h'}")


if __name__ == "__main__":
    main()
