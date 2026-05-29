#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#include <axon/nrf_axon_platform.h>
#include <drivers/axon/nrf_axon_nn_infer.h>

#include "generated/nrf_axon_model_cnn_mel_.h"
#include "sample_input.h"

#ifdef MEL_FROM_RAW
#include "mel_axon.h"
static int8_t mel_buf_int8[MEL_TARGET_FRAMES * MEL_NUM_BINS];
#endif

int main(void)
{
	if (nrf_axon_platform_init() || nrf_axon_nn_model_init_vars(&model_cnn_mel)) {
		printk("axon init failed\n");
		return -1;
	}

#ifdef MEL_FROM_RAW
	const char *expected = g_sample_raw_expected_label;
	const uint32_t t_mel0 = k_cycle_get_32();
	if (mel_axon_compute(mel_buf_int8)) {
		printk("mel_axon_compute failed\n");
		return -1;
	}
	const uint32_t mel_cycles = k_cycle_get_32() - t_mel0;
	const uint64_t mel_us = k_cyc_to_ns_floor64(mel_cycles) / 1000;
	const int8_t *model_input = mel_buf_int8;
	printk("mel_axon: %llu us, %u cycles (%d frames x %d bins)\n",
	       (unsigned long long)mel_us, mel_cycles,
	       MEL_TARGET_FRAMES, MEL_NUM_BINS);
#else
	const char *expected = g_sample_expected_label;
	const int8_t *model_input = g_sample_input;
#endif

	int32_t out_q[2];
	const uint32_t t0 = k_cycle_get_32();
	nrf_axon_nn_model_infer_sync(&model_cnn_mel,
	                             (int8_t *)model_input,
	                             (int8_t *)out_q);
	const uint32_t cycles = k_cycle_get_32() - t0;
	const uint64_t us = k_cyc_to_ns_floor64(cycles) / 1000;

	const float s = (float)model_cnn_mel.output_dequant_mult /
	                (float)(1U << model_cnn_mel.output_dequant_round);
	const int32_t zp = model_cnn_mel.output_dequant_zp;
	const float a = (out_q[0] - zp) * s;
	const float b = (out_q[1] - zp) * s;

	printk("non_target=%.3f  target=%.3f  -> %s (expected %s, %llu us, %u cycles)\n",
	       (double)a, (double)b, (b > a) ? "target" : "non_target",
	       expected, (unsigned long long)us, cycles);
	return 0;
}
