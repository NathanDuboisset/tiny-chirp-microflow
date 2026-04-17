from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Tuple, TYPE_CHECKING, cast
import os
import random
from pydantic import BaseModel, field_serializer, field_validator
import time

# Suppress TF/CUDA noise — must happen before `import tensorflow`.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # hide C++ INFO/WARNING/ERROR
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # disable oneDNN ops
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")

import wandb

import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("ERROR")  # hide Python-level TF INFO/WARNING


if TYPE_CHECKING:
    import keras
else:
    keras = tf.keras

from tensorflow.python.framework.convert_to_constants import (  # noqa: E402
    convert_variables_to_constants_v2_as_graph,
)


# GLOBAL CONFIGURATION / PATHS

SEED = 3407


def set_global_seed(seed: int = SEED) -> None:
    """Set Python, NumPy and TensorFlow RNG seeds."""

    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def configure_tf_runtime() -> None:
    """Enable GPU memory growth (call once before building any model)."""
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


# Repository paths
REPO_ROOT = Path.cwd().parent
MODELS_DIR = REPO_ROOT / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
SRC_DIR = REPO_ROOT / "src"
DATASET_ROOT = REPO_ROOT / "dataset"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelPaths:
    """Convenience bundle for per-model output paths."""

    out_tflite: Path
    out_audio_rs: Path


def get_paths(model_stem: str) -> ModelPaths:
    """Return standard output paths for a given model name (without extension)."""

    out_tflite = MODELS_DIR / f"{model_stem}.tflite"
    out_audio_rs = SRC_DIR / "audio_sample.rs"
    return ModelPaths(out_tflite=out_tflite, out_audio_rs=out_audio_rs)


# AUDIO GEOMETRY

SAMPLE_RATE = 16000
FRAME_LENGTH = 1024
FRAME_STEP = 256

# CNN TIME GEOMETRY
TARGET_FRAMES_TIME = 184
TARGET_AUDIO_LEN_TIME = (TARGET_FRAMES_TIME - 1) * FRAME_STEP + FRAME_LENGTH


# CNN mel
FFT_LENGTH_MEL = FRAME_LENGTH
NUM_MEL_BINS_MEL = 80
LOWER_EDGE_HERTZ = 80.0
UPPER_EDGE_HERTZ = 8000.0
TARGET_FRAMES_MEL = 184
TARGET_AUDIO_LEN_MEL = (TARGET_FRAMES_MEL - 1) * FRAME_STEP + FRAME_LENGTH


# DATASET HELPERS
def make_audio_datasets(
    root: Path = DATASET_ROOT,
    sample_rate: int = SAMPLE_RATE,
    batch_size: int = 32,
    seed: int = SEED,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, np.ndarray]:
    """Create raw train/val/test datasets from the TinyChirp directory layout."""

    train_ds_raw = keras.utils.audio_dataset_from_directory(
        root / "training",
        labels="inferred",
        sampling_rate=sample_rate,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    val_ds_raw = keras.utils.audio_dataset_from_directory(
        root / "validation",
        labels="inferred",
        sampling_rate=sample_rate,
        batch_size=batch_size,
        shuffle=False,
    )
    test_ds_raw = keras.utils.audio_dataset_from_directory(
        root / "testing",
        labels="inferred",
        sampling_rate=sample_rate,
        batch_size=batch_size,
        shuffle=False,
    )

    label_names = np.array(train_ds_raw.class_names)
    return train_ds_raw, val_ds_raw, test_ds_raw, label_names


def fix_audio_length_time(audio: tf.Tensor) -> tf.Tensor:
    """Match cnn_time.ipynb: crop/pad to TARGET_AUDIO_LEN_TIME and add channel."""

    # audio: [batch, time, 1] from keras audio_dataset_from_directory
    audio = tf.squeeze(audio, axis=-1)  # [batch, time]
    audio = audio[:, :TARGET_AUDIO_LEN_TIME]
    current_len = tf.shape(audio)[1]
    pad_len = tf.maximum(0, TARGET_AUDIO_LEN_TIME - current_len)
    audio = tf.pad(audio, [[0, 0], [0, pad_len]])  # ty:ignore[invalid-argument-type]
    audio = tf.ensure_shape(audio, [None, TARGET_AUDIO_LEN_TIME])
    audio = tf.expand_dims(audio, axis=-1)  # [batch, time, 1]
    return audio


def time_to_features(audio: tf.Tensor, label: tf.Tensor):
    audio = fix_audio_length_time(audio)
    return audio, label


def make_time_datasets(
    root: Path = DATASET_ROOT,
    batch_size: int = 32,
    seed: int = SEED,
) -> Tuple[
    tf.data.Dataset,
    tf.data.Dataset,
    tf.data.Dataset,
    np.ndarray,
]:
    """Return (train_ds, val_ds, test_ds, label_names, steps_per_epoch, val_steps, test_steps)
    for the CNN-time model.
    """

    train_raw, val_raw, test_raw, label_names = make_audio_datasets(
        root=root, sample_rate=SAMPLE_RATE, batch_size=batch_size, seed=seed
    )

    # Compute finite cardinalities before repeating.
    train_ds = train_raw.map(
        time_to_features, num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(2)
    val_ds = val_raw.map(
        time_to_features, num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(2)
    test_ds = test_raw.map(
        time_to_features, num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(2)

    return train_ds, val_ds, test_ds, label_names


# MEL HELPERS
def hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_rust_mel_matrix(
    num_mel_bins: int,
    fft_length_mel: int,
    frame_length: int = FRAME_LENGTH,
    sample_rate: int = SAMPLE_RATE,
    lower_edge_hz: float = LOWER_EDGE_HERTZ,
    upper_edge_hz: float = UPPER_EDGE_HERTZ,
):
    """Build the Rust-compatible mel filterbank matrix and return (fft_bins, matrix)."""

    fft_bins = fft_length_mel // 2

    mel_edges = np.zeros(num_mel_bins + 2, dtype=np.int32)
    low_mel = hz_to_mel(lower_edge_hz)
    high_mel = hz_to_mel(upper_edge_hz)
    for i in range(num_mel_bins + 2):
        frac = i / (num_mel_bins + 1)
        mel = low_mel + frac * (high_mel - low_mel)
        hz = mel_to_hz(mel)
        bin_idx = int(((frame_length + 1.0) * hz) / sample_rate)
        mel_edges[i] = min(bin_idx, fft_bins - 1)

    rust_mel_matrix_np = np.zeros((fft_bins, num_mel_bins), dtype=np.float32)
    for m in range(num_mel_bins):
        left = mel_edges[m]
        center = mel_edges[m + 1]
        right = mel_edges[m + 2]

        for k in range(left, center):
            rust_mel_matrix_np[k, m] = (k - left) / max(center - left, 1)
        for k in range(center, right):
            rust_mel_matrix_np[k, m] = (right - k) / max(right - center, 1)

    rust_mel_matrix = tf.constant(rust_mel_matrix_np, dtype=tf.float32)
    return fft_bins, rust_mel_matrix


_RUST_FFT_BINS, RUST_MEL_MATRIX = build_rust_mel_matrix(
    num_mel_bins=NUM_MEL_BINS_MEL,
    fft_length_mel=FFT_LENGTH_MEL,
)


def fix_audio_length_mel(audio: tf.Tensor) -> tf.Tensor:
    audio = tf.squeeze(audio, axis=-1)
    audio = audio[:, :TARGET_AUDIO_LEN_MEL]
    current_len = tf.shape(audio)[1]
    pad_len = tf.maximum(0, TARGET_AUDIO_LEN_MEL - current_len)
    audio = tf.pad(audio, [[0, 0], [0, pad_len]])  # ty:ignore[invalid-argument-type]
    audio = tf.ensure_shape(audio, [None, TARGET_AUDIO_LEN_MEL])
    return audio


def create_log_mel_spectrogram(audio: tf.Tensor) -> tf.Tensor:
    stfts = tf.signal.stft(
        audio,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FFT_LENGTH_MEL,
        window_fn=lambda n, dtype: tf.signal.hann_window(
            n, periodic=False, dtype=dtype
        ),
    )
    spectrograms = tf.abs(stfts)[..., :_RUST_FFT_BINS]
    mel_spectrograms = tf.tensordot(spectrograms, RUST_MEL_MATRIX, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate([NUM_MEL_BINS_MEL]))
    return tf.math.log(mel_spectrograms + 1e-6)


def make_mel_datasets(
    root: Path = DATASET_ROOT,
    batch_size: int = 32,
    seed: int = SEED,
    num_mel_bins: int = NUM_MEL_BINS_MEL,
    target_frames: int = TARGET_FRAMES_MEL,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, np.ndarray]:
    """Return (train_ds, val_ds, test_ds, label_names) for CNN-mel model.

    `num_mel_bins` and `target_frames` are threaded through the tf.data
    pipeline for shape checking, but must currently match the Rust-side
    configuration (NUM_MEL_BINS_MEL / TARGET_FRAMES_MEL).
    """

    if num_mel_bins != NUM_MEL_BINS_MEL or target_frames != TARGET_FRAMES_MEL:
        raise ValueError(
            "make_mel_datasets currently requires num_mel_bins == NUM_MEL_BINS_MEL "
            "and target_frames == TARGET_FRAMES_MEL to stay in sync with Rust."
        )

    def _mel_to_features(audio: tf.Tensor, label: tf.Tensor):
        audio_fixed = fix_audio_length_mel(audio)
        spec = create_log_mel_spectrogram(audio_fixed)
        spec = tf.ensure_shape(spec, [None, target_frames, num_mel_bins])
        spec = tf.expand_dims(spec, axis=-1)
        return spec, label

    train_raw, val_raw, test_raw, label_names = make_audio_datasets(
        root=root, sample_rate=SAMPLE_RATE, batch_size=batch_size, seed=seed
    )

    train_ds = train_raw.map(
        _mel_to_features, num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)
    val_ds = val_raw.map(
        _mel_to_features, num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)
    test_ds = test_raw.map(
        _mel_to_features, num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, label_names


# METRICS


def get_flops_native(model, batch_size=1):
    inputs = []
    for i in range(len(model.inputs)):
        base_shape = list(model.inputs[i].shape[1:])
        full_shape = [batch_size] + base_shape

        spec = tf.TensorSpec(full_shape, model.inputs[i].dtype)
        inputs.append(spec)

    real_model = tf.function(model).get_concrete_function(inputs)

    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(real_model)

    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    opts["output"] = "none"

    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph, run_meta=run_meta, cmd="op", options=opts
    )

    return flops.total_float_ops if flops is not None else 0


def plot_training_history(
    history: object,
    title: str | None = None,
) -> None:
    """Plot training/validation loss and accuracy side-by-side.

    Works with the return value of `tf.keras.Model.fit(...)` (Keras `History`)
    or with a raw `history.history` dict.
    """

    # Defer plotting dependency; most of this module is "TF only".
    import matplotlib.pyplot as plt

    if hasattr(history, "history"):
        history_dict = cast(dict[str, Any], history.history)
    else:
        history_dict = cast(dict[str, Any], history)

    if not isinstance(history_dict, dict):
        raise ValueError(
            "`history` must be a Keras History object or a dict-like `history.history`."
        )

    loss_key = "loss"
    val_loss_key = "val_loss"
    if loss_key not in history_dict or val_loss_key not in history_dict:
        raise ValueError(
            "History is missing `loss` and/or `val_loss`. "
            f"Keys: {sorted(history_dict.keys())}"
        )

    # Keras can report accuracy under different names depending on label encoding.
    accuracy_key = next(
        (
            k
            for k in ("accuracy", "sparse_categorical_accuracy", "categorical_accuracy")
            if k in history_dict
        ),
        None,
    )
    if accuracy_key is None:
        raise ValueError(
            "Could not find an accuracy key in history. "
            f"Keys: {sorted(history_dict.keys())}"
        )

    val_accuracy_key = f"val_{accuracy_key}"
    if val_accuracy_key not in history_dict:
        raise ValueError(
            f"Missing `{val_accuracy_key}` in history. Keys: {sorted(history_dict.keys())}"
        )

    def _to_1d(x: Any) -> np.ndarray:
        arr = np.asarray(x).reshape(-1)
        return arr

    train_loss = _to_1d(history_dict[loss_key])
    val_loss = _to_1d(history_dict[val_loss_key])
    train_acc = _to_1d(history_dict[accuracy_key])
    val_acc = _to_1d(history_dict[val_accuracy_key])

    n_epochs = len(train_loss)
    epochs = np.arange(1, n_epochs + 1)

    if title is None:
        title = "Training History"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title)

    axes[0].plot(epochs, train_loss, label="Train")
    axes[0].plot(epochs, val_loss, label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="Train")
    axes[1].plot(epochs, val_acc, label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


class EvalMetrics(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    threshold: float
    accuracy: float
    precision: float
    recall: float
    f2: float
    auc: float | None = None
    roc_fpr: np.ndarray | None = None
    roc_tpr: np.ndarray | None = None
    roc_thresholds: np.ndarray | None = None
    avg_inference_time_ms: float | None = None

    # Make roc arrays JSON serializable (lists) and accept lists back on load.
    @field_serializer("roc_fpr", "roc_tpr", "roc_thresholds", when_used="json")
    def _serialize_roc_arrays(self, v: np.ndarray | None):
        if v is None:
            return None
        return v.tolist()

    @field_validator("roc_fpr", "roc_tpr", "roc_thresholds", mode="before")
    @classmethod
    def _coerce_roc_arrays(cls, v: Any):
        if v is None:
            return None
        return np.asarray(v, dtype=float)

    @field_validator("avg_inference_time_ms", mode="before")
    @classmethod
    def _coerce_avg_inference_time(cls, v: Any):
        if v is None:
            return None
        return float(v)


class TFLiteEvalRecord(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    model_name: str
    timestamp: str
    train: EvalMetrics
    test: EvalMetrics
    model_size_kb: float | None = None
    flops_mflops: float | None = None
    arena_size_kb: float | None = None


def display_eval_metrics(train_metrics: EvalMetrics, test_metrics: EvalMetrics) -> None:
    print("=== Binary classifier metrics : TEST SET===")
    print(f"Threshold: {test_metrics.threshold:.4f}  (best F2 threshold)")
    print(f"Accuracy : {test_metrics.accuracy:.4f}")
    print(f"Precision: {test_metrics.precision:.4f}")
    print(f"Recall   : {test_metrics.recall:.4f}")
    print(f"F2 score : {test_metrics.f2:.4f}")

    print("=== Binary classifier metrics : TRAIN SET===")
    print(f"AUC      : {train_metrics.auc:.4f}")

    import matplotlib.pyplot as plt

    if train_metrics.roc_fpr is None or train_metrics.roc_tpr is None:
        raise ValueError("ROC curve data is missing.")

    plt.figure()
    plt.plot(
        train_metrics.roc_fpr,
        train_metrics.roc_tpr,
        label=f"AUC = {train_metrics.auc:.3f}",
    )

    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xscale("log")
    plt.title("ROC curve")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()


def evaluate_tflite_model(
    tflite_path: Path,
    model_name: str,
    train_dataset: tf.data.Dataset,
    test_dataset: tf.data.Dataset,
    results_dir: Path | None = None,
) -> tuple[EvalMetrics, EvalMetrics, float]:
    """Evaluate an INT8-quantized TFLite model with a train-chosen best F2 threshold."""
    from datetime import datetime, timezone
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        fbeta_score,
        roc_curve,
        roc_auc_score,
        precision_recall_curve,
    )

    if results_dir is None:
        results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    tflite_path = Path(tflite_path)
    if not tflite_path.exists():
        raise FileNotFoundError(str(tflite_path))

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]
    if in_scale == 0 or out_scale == 0:
        raise ValueError("TFLite quantization scale is 0.")
    qmin, qmax = (-128, 127) if inp["dtype"] == np.int8 else (0, 255)

    def run(x: np.ndarray) -> tuple[float, float]:
        x_q = np.clip(np.round(x / in_scale) + in_zp, qmin, qmax).astype(inp["dtype"])
        interpreter.set_tensor(inp["index"], x_q[None, ...])
        t0 = time.perf_counter()
        interpreter.invoke()
        elapsed = time.perf_counter() - t0
        raw = (
            (interpreter.get_tensor(out["index"]).astype(np.float32) - out_zp)
            * out_scale
        ).reshape(-1)
        if raw.size == 1:
            prob = float(1.0 / (1.0 + np.exp(-raw[0])))
        else:
            e = np.exp(raw - raw.max())
            prob = float(e[1] / e.sum())
        return prob, elapsed

    def collect(ds: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y_true, y_score, times = [], [], []
        for x_batch, y_batch in ds.unbatch():
            prob, elapsed = run(x_batch.numpy())
            y_true.append(int(y_batch.numpy()))
            y_score.append(prob)
            times.append(elapsed)
        return np.asarray(y_true), np.asarray(y_score), np.asarray(times)

    y_train, s_train, _ = collect(train_dataset)
    y_test, s_test, times_test = collect(test_dataset)

    # Best F2 threshold from train precision-recall curve
    prec_c, rec_c, thr_c = precision_recall_curve(y_train, s_train)
    denom = 4 * prec_c[:-1] + rec_c[:-1]
    f2_c = np.where(denom > 0, 5 * prec_c[:-1] * rec_c[:-1] / denom, 0.0)
    best_thr = float(thr_c[np.argmax(f2_c)])

    # ROC + AUC on train set
    fpr, tpr, roc_thr = roc_curve(y_train, s_train)
    auc = float(roc_auc_score(y_train, s_train))

    y_pred_tr = (s_train >= best_thr).astype(int)
    train_metrics = EvalMetrics(
        threshold=best_thr,
        accuracy=float(accuracy_score(y_train, y_pred_tr)),
        precision=float(precision_score(y_train, y_pred_tr, zero_division=0)),
        recall=float(recall_score(y_train, y_pred_tr, zero_division=0)),
        f2=float(fbeta_score(y_train, y_pred_tr, beta=2, zero_division=0)),
        auc=auc,
        roc_fpr=fpr,
        roc_tpr=tpr,
        roc_thresholds=roc_thr,
    )

    y_pred_test = (s_test >= best_thr).astype(int)
    avg_ms = float(np.mean(times_test) * 1000.0)
    test_metrics = EvalMetrics(
        threshold=best_thr,
        accuracy=float(accuracy_score(y_test, y_pred_test)),
        precision=float(precision_score(y_test, y_pred_test, zero_division=0)),
        recall=float(recall_score(y_test, y_pred_test, zero_division=0)),
        f2=float(fbeta_score(y_test, y_pred_test, beta=2, zero_division=0)),
        avg_inference_time_ms=avg_ms,
    )

    import io as _io
    import contextlib as _ctx
    import re as _re

    # Capture Analyzer output for both model size and arena estimation
    buf = _io.StringIO()
    with _ctx.redirect_stdout(buf):
        tf.lite.experimental.Analyzer.analyze(model_path=str(tflite_path))
    analyzer_output = buf.getvalue()

    # Flash: weight data only (INT8 weights + INT32 biases), excluding flatbuffer overhead
    m_data = _re.search(r"Total data buffer size:\s*(\d+)\s*bytes", analyzer_output)
    model_size_kb = int(m_data.group(1)) / 1024.0 if m_data else tflite_path.stat().st_size / 1024.0

    # RAM: sum all activation tensors (shape_signature = dynamic batch dim → runtime-allocated)
    _dtype_bytes = {"INT8": 1, "UINT8": 1, "INT16": 2, "INT32": 4, "FLOAT32": 4}
    activation_total = 0
    for line in analyzer_output.splitlines():
        sig_m = _re.search(r"shape_signature:\[([-\d,\s]+)\],\s*type:(\w+)", line)
        if sig_m:
            dims = [max(1, int(d)) for d in sig_m.group(1).split(",")]
            activation_total += int(np.prod(dims)) * _dtype_bytes.get(sig_m.group(2), 1)
    arena_size_kb = activation_total / 1024.0 if activation_total > 0 else None

    # MFLOPs estimate: sum shape products of all 4-D tensors (conv weights / activations)
    flops_mflops = sum(
        float(np.prod(t["shape"])) / 1e6
        for t in interpreter.get_tensor_details()
        if len(t["shape"]) == 4
    )

    print(f"Model size : {model_size_kb:.1f} KB")
    print(f"Est. MFLOPs: {flops_mflops:.3f}")
    if arena_size_kb is not None:
        print(f"Arena size : {arena_size_kb:.1f} KB")

    record = TFLiteEvalRecord(
        model_name=model_name,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        train=train_metrics,
        test=test_metrics,
        model_size_kb=model_size_kb,
        flops_mflops=flops_mflops,
        arena_size_kb=arena_size_kb,
    )
    with (results_dir / f"{model_name}.jsonl").open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")

    display_eval_metrics(train_metrics, test_metrics)
    return train_metrics, test_metrics, avg_ms


# TFLite export helpers


def build_representative_batches(
    dataset: tf.data.Dataset,
    take: int = 100,
) -> List[np.ndarray]:
    """Collect a small set of representative samples for quantization.

    Works for any input shape (time-domain or mel spectrogram): adds a batch
    dimension without reshaping, so the sample shape matches the model input.
    """

    batches: List[np.ndarray] = []
    for x_batch, _ in dataset.unbatch().take(take):
        sample = x_batch.numpy().astype(np.float32)
        sample = sample[None, ...]  # add batch dim: [...] -> [1, ...]
        batches.append(sample)
    return batches


def representative_dataset_from_batches(
    batches: Iterable[np.ndarray],
) -> Callable[[], Iterable[List[np.ndarray]]]:
    """Create a TFLite representative_dataset generator from prebuilt batches."""

    def gen():
        for sample in batches:
            yield [sample]

    return gen


def export_keras_model_to_int8_tflite(
    model: keras.Model,
    rep_batches: Iterable[np.ndarray],
    out_tflite: Path,
    tmp_dir: str = "temp_saved_model",
) -> None:
    """Export a Keras model as an INT8-quantized TFLite flatbuffer."""

    model.export(tmp_dir)
    converter = tf.lite.TFLiteConverter.from_saved_model(tmp_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_from_batches(rep_batches)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    out_tflite.write_bytes(converter.convert())


# Rust audio_sample.rs helpers
TestClip = Tuple[str, np.ndarray, str]


def collect_test_clips_for_rs(
    root: Path,
    sample_rate: int,
    target_len: int,
    num_per_label: int = 2,
) -> List[TestClip]:
    """Collect a small, fixed set of test clips for Rust."""

    raw_sample_ds = keras.utils.audio_dataset_from_directory(
        root,
        labels="inferred",
        sampling_rate=sample_rate,
        batch_size=1,
        shuffle=False,
    )

    clips_by_label: dict[int, List[np.ndarray]] = {}
    for audio_batch, label_batch in raw_sample_ds.unbatch():
        label_idx = int(label_batch.numpy())
        if label_idx not in clips_by_label:
            clips_by_label[label_idx] = []
        if len(clips_by_label[label_idx]) < num_per_label:
            fixed = (
                fix_audio_length_time(tf.expand_dims(audio_batch, 0))[0]
                .numpy()
                .astype(np.float32)
            )
            clips_by_label[label_idx].append(fixed)
        if (
            all(len(v) >= num_per_label for v in clips_by_label.values())
            and len(clips_by_label) >= 2
        ):
            break

    if len(clips_by_label) < 2:
        raise RuntimeError("Expected at least two labels in testing dataset.")

    # Assumes binary classification with labels 0 and 1, matching existing notebooks.
    ordered: List[TestClip] = []
    for i in range(num_per_label):
        for label_idx in sorted(clips_by_label.keys()):
            label_name = "target" if label_idx == 1 else "non_target"
            audio = clips_by_label[label_idx][i]
            rel_path = f"dataset/testing/{label_name}/sample_{i + 1}.wav"
            ordered.append((label_name, audio, rel_path))

    return ordered


def write_audio_sample_rs(
    out_path: Path,
    clips: List[TestClip],
    sample_rate: int,
    generator_name: str = "building_tensorflow/cnn_time.ipynb",
) -> None:
    """Write a Rust audio_sample.rs file compatible with the TinyChirp runner."""

    rs: List[str] = []
    rs.append(f"// Generated by {generator_name}\n")
    rs.append(f"pub const SAMPLE_RATE: usize = {sample_rate};\n\n")
    rs.append("pub struct TestClip {\n")
    rs.append("    pub expected_label: &'static str,\n")
    rs.append("    pub source_file: &'static str,\n")
    rs.append("    pub audio: &'static [f32],\n")
    rs.append("}\n\n")

    for i, (_label, audio, _rel_path) in enumerate(clips, 1):
        audio_vals = ", ".join(f"{float(v):.8f}" for v in audio)
        rs.append(f"pub const CLIP_{i}: &[f32] = &[{audio_vals}];\n\n")

    rs.append(f"pub const TEST_CLIPS: [TestClip; {len(clips)}] = [\n")
    for i, (label, _audio, rel_path) in enumerate(clips, 1):
        rs.append("    TestClip {\n")
        rs.append(f'        expected_label: "{label}",\n')
        rs.append(f'        source_file: "{rel_path}",\n')
        rs.append(f"        audio: CLIP_{i},\n")
        rs.append("    },\n")
    rs.append("];\n")

    out_path.write_text("".join(rs), encoding="utf-8")


# W&B helpers


def init_wandb(run_name: str, config: dict, project: str = "tinychirp") -> None:
    import wandb

    wandb.init(name=run_name, project=project, config=config)


class TimingCallback(tf.keras.callbacks.Callback):
    def __init__(self, batch_size, log_freq=10):
        super().__init__()
        self.batch_size = batch_size
        self.log_freq = log_freq

    def on_train_batch_begin(self, batch, logs=None):
        self._train_start = time.time()

    def on_train_batch_end(self, batch, logs=None):
        step_time = time.time() - self._train_start

        if batch % self.log_freq == 0:
            if wandb is not None:
                wandb.log(
                    {
                        "train/step_time": step_time,
                        "train/samples_per_sec": self.batch_size / step_time,
                    }
                )

    def on_test_begin(self, logs=None):
        self._val_epoch_start = time.time()

    def on_test_end(self, logs=None):
        val_time = time.time() - self._val_epoch_start

        log_dict = {"val/epoch_time": val_time}

        if hasattr(self, "params"):
            steps = self.params.get("steps")
            if steps:
                total = steps * self.batch_size
                log_dict["val/samples_per_sec_epoch"] = total / val_time
        wandb.log(log_dict)

    def on_test_batch_begin(self, batch, logs=None):
        self._val_start = time.time()

    def on_test_batch_end(self, batch, logs=None):
        step_time = time.time() - self._val_start

        if batch % self.log_freq == 0:
            wandb.log(
                {
                    "val/step_time": step_time,
                    "val/samples_per_sec": self.batch_size / step_time,
                }
            )


def get_callbacks(
    patience: int | None = None,
    log_freq: int = 5,
    batch_size: int = 32,
) -> list:
    from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

    callbacks = []
    callbacks.append(WandbMetricsLogger(log_freq=log_freq))
    callbacks.append(WandbModelCheckpoint(str(CHECKPOINT_DIR / "checkpoint.keras")))
    callbacks.append(TimingCallback(batch_size=batch_size, log_freq=log_freq))
    if patience is not None:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                patience=patience, restore_best_weights=True, monitor="val_loss"
            )
        )
    return callbacks


def finish_wandb() -> None:
    import wandb

    wandb.finish()
