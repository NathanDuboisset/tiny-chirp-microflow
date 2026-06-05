#pragma once
#include <stdint.h>

#include "generated_data/sample_input_sincnet_meta.h"

typedef struct {
	const int8_t *audio;       /* int8-quantized, NUM_CHUNKS*CHUNK_SIZE bytes */
	const char   *label;
} sincnet_clip_t;

extern const sincnet_clip_t g_sincnet_clips[N_SINCNET_CLIPS];
