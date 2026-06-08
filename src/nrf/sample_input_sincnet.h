#pragma once

/*
 * Baked-in test clips for the sincnet_chunked path. Audio is already int8-
 * quantized and laid out as the (NUM_CHUNKS, CHUNK_SIZE) tensor the Axon
 * model expects. See scripts/gen_sincnet_assets.py.
 */

#include <stdint.h>

#include "generated_data/sample_input_sincnet_meta.h"

typedef struct {
	const int8_t *audio;       /* int8-quantized, NUM_CHUNKS*CHUNK_SIZE bytes */
	const char   *label;
} sincnet_clip_t;

extern const sincnet_clip_t g_sincnet_clips[N_SINCNET_CLIPS];
