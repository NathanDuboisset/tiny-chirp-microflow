#include "sample_input_sincnet.h"

#define INCBIN(symbol, file, align) \
	__asm__(".section .rodata." #symbol ",\"a\",%progbits\n" \
		".balign " #align "\n" \
		".global " #symbol "\n" \
		".type " #symbol ",%object\n" \
		#symbol ":\n" \
		".incbin \"" file "\"\n" \
		".size " #symbol ", . - " #symbol "\n" \
		".previous\n")

INCBIN(g_sincnet_audio_0, "generated_data/sample_sincnet_audio_0.bin", 1);
INCBIN(g_sincnet_audio_1, "generated_data/sample_sincnet_audio_1.bin", 1);
INCBIN(g_sincnet_audio_2, "generated_data/sample_sincnet_audio_2.bin", 1);
INCBIN(g_sincnet_audio_3, "generated_data/sample_sincnet_audio_3.bin", 1);
extern const int8_t g_sincnet_audio_0[];
extern const int8_t g_sincnet_audio_1[];
extern const int8_t g_sincnet_audio_2[];
extern const int8_t g_sincnet_audio_3[];

const sincnet_clip_t g_sincnet_clips[N_SINCNET_CLIPS] = {
	{ .audio = g_sincnet_audio_0, .label = SINCNET_CLIP_0_LABEL },
	{ .audio = g_sincnet_audio_1, .label = SINCNET_CLIP_1_LABEL },
	{ .audio = g_sincnet_audio_2, .label = SINCNET_CLIP_2_LABEL },
	{ .audio = g_sincnet_audio_3, .label = SINCNET_CLIP_3_LABEL },
};
