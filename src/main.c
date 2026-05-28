/* cnn_mel on axon (nRF54LM20B). sincnet doesn't fit:
 * 65-tap filters > 16, raw input > line buffer. */

#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#include <axon/nrf_axon_platform.h>
#include <drivers/axon/nrf_axon_nn_infer.h>

#include "generated/nrf_axon_model_cnn_mel_.h"
#include "sample_input.h"

int main(void)
{
	if (nrf_axon_platform_init() || nrf_axon_nn_model_init_vars(&model_cnn_mel)) {
		printk("axon init failed\n");
		return -1;
	}

	int32_t out_q[2];
	const uint32_t t0 = k_cycle_get_32();
	nrf_axon_nn_model_infer_sync(&model_cnn_mel,
	                             (int8_t *)g_sample_input,
	                             (int8_t *)out_q);
	const uint32_t cycles = k_cycle_get_32() - t0;
	const uint64_t us = k_cyc_to_ns_floor64(cycles) / 1000;

	/* disable_op_quantization=True -> int32 logits, dequant = q * mult / 2^round */
	const float s = (float)model_cnn_mel.output_dequant_mult /
	                (float)(1U << model_cnn_mel.output_dequant_round);
	const int32_t zp = model_cnn_mel.output_dequant_zp;
	const float a = (out_q[0] - zp) * s;
	const float b = (out_q[1] - zp) * s;

	printk("non_target=%.3f  target=%.3f  -> %s (expected %s, %llu us, %u cycles)\n",
	       (double)a, (double)b, (b > a) ? "target" : "non_target",
	       g_sample_expected_label, (unsigned long long)us, cycles);
	return 0;
}
