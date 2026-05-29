## TinyChirp on MicroFlow

TinyChirp bird call demo running on the MicroFlow Rust TinyML engine.

Using microflow and ariel os as the infrastructure, for a full Rust code

- MicroFlow repo: `https://github.com/matteocarnelos/microflow-rs`
- MicroFlow paper: `https://arxiv.org/pdf/2409.19432`

Based of the following :

- TinyChirp repo: `https://github.com/TinyPART/TinyChirp`
- TinyChirp paper: `https://arxiv.org/abs/2407.21453`

## How to build models

uv from astral (https://docs.astral.sh/uv/) is used to manage python deps. Since torch and tensorflow can be conflicting on versions, there is two folders to generate, each having its own env,
In a folder, run 
```
uv sync
```
then run notebooks with the .venv created in that folder

### How to run

From the repo root, pick the input pipeline and the model via laze modules
(`-s`), then a board (`-b`):

```bash
laze build -b {your-board-ariel-id} -s input-time -s model-sincnet-multi run
```

Input modules (pick one): `input-mel`, `input-time`.
Model modules (pick one): `model-cnn-tf`, `model-cnn-torch`, `model-sincnet`,
`model-sincnet-multi`, `model-sincnet-mimic-mel`.

This builds the firmware, runs the TinyChirp model on test clips and prints
predictions + latency over serial.

#### nRF boards

ariel-os v0.4.0 supports `nrf52840dk`, `nrf5340dk-app`, `nrf9151-dk`, etc. No nRF54L support yet; `nrf5340dk-app` is the closest Cortex-M33 target (512 KB RAM / 1 MB flash).

`microflow` is pinned to the sibling `../microflow-rs` checkout on branch `feat/streaming-and-timing` (adds the `ariel-os`, streaming, and transpose features used by `#[model(...)]`).

### Check RAM / Flash usage

Binary path after build:

```text
build/bin/{your-board-ariel-id}/cargo/thumbv8m.main-none-eabihf/release/tiny-chirp-microflow
```

Example:

```bash
runtime_file_path=build/bin/{your-board-ariel-id}/cargo/thumbv8m.main-none-eabihf/release/tiny-chirp-microflow

arm-none-eabi-size "$runtime_file_path"
nm --print-size --size-sort --demangle=rust --radix=d "$runtime_file_path"
```

## nRF54LM20B / Axon NPU (Zephyr / nRF Connect SDK)

Second firmware, sitting next to the Rust code: a Zephyr app running `models/cnn_mel_tf.tflite` on the Axon NPU. SincNet doesn't fit (filter > 16 taps, input width > line buffer), cnn_mel does.

C sources live in `src/` next to the `*.rs` files. Build dir is `build_ncs/` so it doesn't clash with laze's `build/`.

### Build & flash

```bash
# from repo root — default: precomputed log-mel input
nrfutil toolchain-manager launch --ncs-version v3.3.0 -- bash -c \
  'ZEPHYR_BASE=$HOME/ncs/zephyr west build -b nrf54lm20dk/nrf54lm20b/cpuapp -p auto -d build_ncs .'
nrfutil toolchain-manager launch --ncs-version v3.3.0 -- bash -c \
  'ZEPHYR_BASE=$HOME/ncs/zephyr west flash -d build_ncs'
picocom -b 115200 /dev/ttyACM1
```

Add `-- -DMEL_FROM_RAW=ON` (with `-p always`) to compute the log-mel on the NPU from raw PCM instead. Mel and inference times are then printed on separate lines.

The DK exposes two CDC-ACM ports. App VCOM is `ttyACM1`, `ttyACM0` is the J-Link channel (silent). Reset the board after opening picocom to catch the boot print.

In nRF Connect for VS Code: open the repo root, Add Build Configuration, board `nrf54lm20dk/nrf54lm20b/cpuapp`, build directory `build_ncs`.

### Regenerate model / test input

```bash
./scripts/compile.sh   # cnn_mel_tf.tflite -> src/generated/nrf_axon_model_cnn_mel_.h
```

`building/export_audio_sample_rs.ipynb` writes `src/sample_input.[ch]` (precomputed log-mel) and, for `MEL_FROM_RAW`, `src/sample_input_raw.c` + `src/generated_data/*.inc`.