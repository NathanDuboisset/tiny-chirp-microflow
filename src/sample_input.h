#pragma once
#include <stdint.h>
#include <stddef.h>

#include "generated_data/sample_input_meta.h"

typedef struct {
	const int16_t *audio;       /* int16 PCM, MEL_AUDIO_LEN samples */
	const int8_t  *mel_baked;   /* int8 log-mel, MEL_TARGET_FRAMES * MEL_NUM_BINS */
	const char    *label;
} sample_clip_t;

extern const sample_clip_t g_sample_clips[N_SAMPLE_CLIPS];

#ifdef MEL_FROM_RAW
extern const int16_t g_hann_window_q15[];
extern const int16_t g_hann_even_q15[];
extern const int16_t g_hann_odd_q15[];
extern const float   g_hann_f32[];
extern const float   g_twiddle_f32[];
extern const int16_t g_mel_matrix_q15[];
extern const float   g_input_quant_scale;
extern const int32_t g_input_quant_zp;
#endif
