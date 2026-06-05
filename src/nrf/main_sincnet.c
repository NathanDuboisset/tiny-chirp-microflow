#include <assert.h>
#include <string.h>

#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#include <axon/nrf_axon_platform.h>
#include <drivers/axon/nrf_axon_nn_infer.h>

#include "generated/nrf_axon_model_sincnet_chunked_.h"
#include "sample_input_sincnet.h"

static uint64_t cycles_to_us(uint32_t cycles)
{
	return k_cyc_to_ns_floor64(cycles) / 1000;
}

int main(void)
{
	if (nrf_axon_platform_init() || nrf_axon_nn_model_init_vars(&model_sincnet_chunked)) {
		printk("axon init failed\n");
		return -1;
	}

	printk("\n  sincnet_chunked | %d clips x %d chunks x %d samples\n\n",
	       N_SINCNET_CLIPS, SINCNET_NUM_CHUNKS, SINCNET_CHUNK_SIZE);
	printk("  #  expected    pred  score(nt) / score(t)  -> result        inf\n");

	const float s = (float)model_sincnet_chunked.output_dequant_mult /
	                (float)(1U << model_sincnet_chunked.output_dequant_round);
	const int32_t zp = model_sincnet_chunked.output_dequant_zp;

	uint64_t total_inf_us = 0;

	for (int i = 0; i < N_SINCNET_CLIPS; i++) {
		const sincnet_clip_t *c = &g_sincnet_clips[i];

		int32_t out_q[2];
		const uint32_t t_inf = k_cycle_get_32();
		nrf_axon_nn_model_infer_sync(&model_sincnet_chunked,
		                             (int8_t *)c->audio, (int8_t *)out_q);
		const uint64_t inf_us = cycles_to_us(k_cycle_get_32() - t_inf);

		const float a = (out_q[0] - zp) * s;
		const float b = (out_q[1] - zp) * s;
		const char *pred = (b > a) ? "target" : "non_target";
		const char *mark = (strcmp(pred, c->label) == 0) ? "ok " : "BAD";

		printk("  %d  %-10s  %s  %7.3f / %7.3f  -> %-10s  %5llu us\n",
		       i, c->label, mark, (double)a, (double)b, pred,
		       (unsigned long long)inf_us);

		total_inf_us += inf_us;
	}

	printk("                                          avg %5llu us\n",
	       (unsigned long long)(total_inf_us / N_SINCNET_CLIPS));
	return 0;
}
