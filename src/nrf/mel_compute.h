#pragma once

#include <stdint.h>

int mel_compute(const int16_t *audio, int8_t *out_int8);

int mel_fft_frame(const int16_t *frame_in, int32_t *power_out);

extern const float mel_fft_log_offset;
extern const char *const mel_backend_name;
