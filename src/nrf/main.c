/*
 * cnn_mel entry point: runs the Axon-compiled CNN over 4 baked-in test clips.
 * If MEL_FROM_RAW is defined the int8 log-mel input is computed on-device from
 * raw PCM (see mel_compute.c + mel_fft_{cpu,axon_split}.c); otherwise the
 * pre-computed mel from gen_assets.py is fed directly to the model.
 */

#include <string.h>

#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#include <axon/nrf_axon_platform.h>
#include <drivers/axon/nrf_axon_nn_infer.h>

#include "generated/nrf_axon_model_cnn_mel_.h"
#include "sample_input.h"

#ifdef MEL_FROM_RAW
#include "mel_compute.h"
static int8_t mel_buf_int8[MEL_TARGET_FRAMES * MEL_NUM_BINS];
#define BACKEND_NAME mel_backend_name
#else
#define BACKEND_NAME "baked"
#endif

static uint64_t cycles_to_us(uint32_t cycles)
{
	return k_cyc_to_ns_floor64(cycles) / 1000;
}

int main(void)
{
	if (nrf_axon_platform_init() || nrf_axon_nn_model_init_vars(&model_cnn_mel)) {
		printk("axon init failed\n");
		return -1;
	}

	printk("\n  cnn_mel | mel backend: %s | %d clips x %d frames x %d bins\n\n",
	       BACKEND_NAME, N_SAMPLE_CLIPS, MEL_TARGET_FRAMES, MEL_NUM_BINS);
	printk("  #  expected    pred  score(nt) / score(t)  -> result      pre        inf\n");

	const float s = (float)model_cnn_mel.output_dequant_mult /
	                (float)(1U << model_cnn_mel.output_dequant_round);
	const int32_t zp = model_cnn_mel.output_dequant_zp;

	uint64_t total_pre_us = 0;
	uint64_t total_inf_us = 0;

	for (int i = 0; i < N_SAMPLE_CLIPS; i++) {
		const sample_clip_t *c = &g_sample_clips[i];

		uint64_t pre_us = 0;
		const int8_t *model_input;
#ifdef MEL_FROM_RAW
		const uint32_t t_pre = k_cycle_get_32();
		if (mel_compute(c->audio, mel_buf_int8)) {
			printk("  %d  %-10s  mel_compute failed\n", i, c->label);
			continue;
		}
		pre_us = cycles_to_us(k_cycle_get_32() - t_pre);
		model_input = mel_buf_int8;
#else
		model_input = c->mel_baked;
#endif

		int32_t out_q[2];
		const uint32_t t_inf = k_cycle_get_32();
		nrf_axon_nn_model_infer_sync(&model_cnn_mel,
		                             (int8_t *)model_input, (int8_t *)out_q);
		const uint64_t inf_us = cycles_to_us(k_cycle_get_32() - t_inf);

		const float a = (out_q[0] - zp) * s;
		const float b = (out_q[1] - zp) * s;
		const char *pred = (b > a) ? "target" : "non_target";
		const char *mark = (strcmp(pred, c->label) == 0) ? "ok " : "BAD";

		printk("  %d  %-10s  %s  %7.3f / %7.3f  -> %-10s  %5llu ms  %5llu us\n",
		       i, c->label, mark, (double)a, (double)b, pred,
		       (unsigned long long)(pre_us / 1000),
		       (unsigned long long)inf_us);

		total_pre_us += pre_us;
		total_inf_us += inf_us;
	}

	printk("                                          avg %5llu ms  %5llu us\n",
	       (unsigned long long)(total_pre_us / N_SAMPLE_CLIPS / 1000),
	       (unsigned long long)(total_inf_us / N_SAMPLE_CLIPS));
	return 0;
}
