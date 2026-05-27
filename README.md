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

ariel-os v0.4.0 supports `nrf52840dk`, `nrf5340dk-app`, `nrf9151-dk`, etc.
**The nRF54L family is not yet supported by ariel-os v0.4.0.** For a Cortex-M33
target close to nRF54L, use `nrf5340dk-app` (512 KB RAM / 1 MB flash).

This crate depends on a forked `microflow` (sibling `../microflow-rs`) checked
out on the `feat/streaming-and-timing` branch — it provides the `ariel-os`,
streaming, and transpose features used by `#[model(...)]`.

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

A second, separate firmware lives alongside the Rust code: a Zephyr app that
runs `models/cnn_mel_tf.tflite` on the nRF54LM20B's Axon NPU. SincNet doesn't
fit (filter > 16 taps, input width > line buffer); cnn_mel does.

The Zephyr app uses `CMakeLists.txt`, `prj.conf`, `boards/`, and its C sources
in `src/` (alongside the Rust `*.rs` files — different file extensions, no
collision). Builds go to `build_ncs/` to stay clear of laze's `build/`.

### Build & flash

```bash
# from repo root
nrfutil toolchain-manager launch --ncs-version v3.3.0 -- bash -c \
  'ZEPHYR_BASE=$HOME/ncs/zephyr west build -b nrf54lm20dk/nrf54lm20b/cpuapp -p auto -d build_ncs .'
nrfutil toolchain-manager launch --ncs-version v3.3.0 -- bash -c \
  'ZEPHYR_BASE=$HOME/ncs/zephyr west flash -d build_ncs'
picocom -b 115200 /dev/ttyACM0
```

In the nRF Connect for VS Code extension: open this repo at the root,
**Add Build Configuration**, board `nrf54lm20dk/nrf54lm20b/cpuapp`,
**Build directory** = `build_ncs`.

### Regenerate model / test input

```bash
# from repo root, with .venv synced
./scripts/compile.sh                                   # model -> src/generated/
.venv/bin/python scripts/gen_mel_input.py --clip 1     # -> src/sample_input.[ch]
```