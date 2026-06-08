#pragma once

/*
 * Public API for on-device log-mel preprocessing.
 *
 *   mel_compute()       — full pipeline: int16 PCM -> int8 log-mel tensor.
 *   mel_fft_frame()     — single-frame power spectrum, provided by either
 *                         mel_fft_cpu.c (CMSIS-DSP) or
 *                         mel_fft_axon_split.c (Axon NPU, even/odd split).
 *   mel_fft_log_offset  — constant the backend adds in log-domain to align
 *                         its fixed-point scale with the TFLite-trained model.
 *   mel_backend_name    — "cpu" or "axon", for logging.
 */

#include <stdint.h>

int mel_compute(const int16_t *audio, int8_t *out_int8);

int mel_fft_frame(const int16_t *frame_in, int32_t *power_out);

extern const float mel_fft_log_offset;
extern const char *const mel_backend_name;
