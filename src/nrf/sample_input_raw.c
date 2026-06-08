/*
 * Pulls the cnn_mel sample blobs (raw int16 PCM, baked int8 mel, and — when
 * MEL_FROM_RAW is set — the Hann window / twiddle / mel-matrix tables) into
 * .rodata via .incbin, and exposes them through g_sample_clips[].
 */

#include "sample_input.h"

#define INCBIN(symbol, file, align) \
	__asm__(".section .rodata." #symbol ",\"a\",%progbits\n" \
		".balign " #align "\n" \
		".global " #symbol "\n" \
		".type " #symbol ",%object\n" \
		#symbol ":\n" \
		".incbin \"" file "\"\n" \
		".size " #symbol ", . - " #symbol "\n" \
		".previous\n")

INCBIN(g_clip_mel_0, "generated_data/sample_mel_0.bin", 1);
INCBIN(g_clip_mel_1, "generated_data/sample_mel_1.bin", 1);
INCBIN(g_clip_mel_2, "generated_data/sample_mel_2.bin", 1);
INCBIN(g_clip_mel_3, "generated_data/sample_mel_3.bin", 1);
extern const int8_t g_clip_mel_0[];
extern const int8_t g_clip_mel_1[];
extern const int8_t g_clip_mel_2[];
extern const int8_t g_clip_mel_3[];

#ifdef MEL_FROM_RAW
INCBIN(g_clip_audio_0, "generated_data/sample_audio_0.bin", 2);
INCBIN(g_clip_audio_1, "generated_data/sample_audio_1.bin", 2);
INCBIN(g_clip_audio_2, "generated_data/sample_audio_2.bin", 2);
INCBIN(g_clip_audio_3, "generated_data/sample_audio_3.bin", 2);
extern const int16_t g_clip_audio_0[];
extern const int16_t g_clip_audio_1[];
extern const int16_t g_clip_audio_2[];
extern const int16_t g_clip_audio_3[];

INCBIN(g_hann_even_q15,   "generated_data/hann_even_q15.bin",   2);
INCBIN(g_hann_odd_q15,    "generated_data/hann_odd_q15.bin",    2);
INCBIN(g_hann_f32,        "generated_data/hann_f32.bin",        4);
INCBIN(g_twiddle_f32,     "generated_data/twiddle_f32.bin",     4);
INCBIN(g_mel_matrix_q15,  "generated_data/mel_matrix_q15.bin",  2);

const float   g_input_quant_scale = MEL_INPUT_QUANT_SCALE;
const int32_t g_input_quant_zp    = MEL_INPUT_QUANT_ZP;
#endif

const sample_clip_t g_sample_clips[N_SAMPLE_CLIPS] = {
#ifdef MEL_FROM_RAW
	{ g_clip_audio_0, g_clip_mel_0, SAMPLE_CLIP_0_LABEL },
	{ g_clip_audio_1, g_clip_mel_1, SAMPLE_CLIP_1_LABEL },
	{ g_clip_audio_2, g_clip_mel_2, SAMPLE_CLIP_2_LABEL },
	{ g_clip_audio_3, g_clip_mel_3, SAMPLE_CLIP_3_LABEL },
#else
	{ NULL, g_clip_mel_0, SAMPLE_CLIP_0_LABEL },
	{ NULL, g_clip_mel_1, SAMPLE_CLIP_1_LABEL },
	{ NULL, g_clip_mel_2, SAMPLE_CLIP_2_LABEL },
	{ NULL, g_clip_mel_3, SAMPLE_CLIP_3_LABEL },
#endif
};
