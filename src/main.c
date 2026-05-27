/*
 * cnn_mel on the Axon NPU (nRF54LM20B). SincNet variants do not fit Axon —
 * their raw-audio 1D convs blow the line buffer and 65-tap filters exceed
 * the 16-tap conv kernel limit. cnn_mel is the deployable model in this repo.
 *
 * Pipeline:
 *   models/cnn_mel_tf.tflite -> scripts/compile.sh -> generated header
 *   audio_samples/<any>_tf.rs CLIP_N -> scripts/gen_mel_input.py
 *     (dequantize -> log-mel via building/utils.py -> int8 quant for cnn_mel)
 *     -> src/sample_input.[ch]
 *
 * The Axon-compiled model keeps the tflite's int8 quantization, so we feed
 * the int8 sample bytes straight into the input tensor and dequantize the
 * int8 output using the params baked into the model struct.
 */

#include <assert.h>

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/printk.h>

#include <axon/nrf_axon_platform.h>
#include <drivers/axon/nrf_axon_driver.h>
#include <drivers/axon/nrf_axon_nn_infer.h>

#include "generated/nrf_axon_model_cnn_mel_.h"
#include "sample_input.h"

LOG_MODULE_REGISTER(tinychirp, LOG_LEVEL_INF);

static const char *const k_class_names[] = {"non_target", "target"};

static void dequantize(const int8_t *q, size_t n, float *out,
		       const nrf_axon_nn_compiled_model_s *model)
{
	const float scale = (float)model->output_dequant_mult /
			    (float)(1U << model->output_dequant_round);
	const int8_t zp = model->output_dequant_zp;

	for (size_t i = 0; i < n; i++) {
		out[i] = (float)(q[i] - zp) * scale;
	}
}

int main(void)
{
	printk("\n=== tiny-chirp / cnn_mel on nRF54LM20B Axon NPU ===\n");
	printk("expected: %s\n", g_sample_expected_label);

	const nrf_axon_nn_compiled_model_input_s *in =
		&model_cnn_mel.inputs[model_cnn_mel.external_input_ndx];
	const size_t in_elems = in->dimensions.height * in->dimensions.width *
				in->dimensions.channel_cnt;

	printk("model input: %ux%ux%u, byte_width=%u (total=%zu)\n",
	       in->dimensions.height, in->dimensions.width,
	       in->dimensions.channel_cnt, in->dimensions.byte_width, in_elems);

	if (in->dimensions.byte_width != 1) {
		printk("ERR: expected int8 input (byte_width=1), got %u\n",
		       in->dimensions.byte_width);
		return -1;
	}
	if (in_elems != g_sample_input_len) {
		printk("ERR: model wants %zu samples, sample has %zu\n",
		       in_elems, g_sample_input_len);
		return -1;
	}

	nrf_axon_result_e r = nrf_axon_nn_model_validate(&model_cnn_mel);
	if (r != NRF_AXON_RESULT_SUCCESS) {
		printk("ERR: model validate failed (%d)\n", r);
		return -1;
	}

	r = nrf_axon_platform_init();
	if (r != NRF_AXON_RESULT_SUCCESS) {
		printk("ERR: platform init failed (%d)\n", r);
		return -1;
	}
	int err = nrf_axon_nn_model_init_vars(&model_cnn_mel);
	if (err) {
		printk("ERR: model init_vars failed (%d)\n", err);
		return -1;
	}

	const size_t out_elems = model_cnn_mel.output_dimensions.height *
				 model_cnn_mel.output_dimensions.width *
				 model_cnn_mel.output_dimensions.channel_cnt;
	int8_t output_q[8];
	if (out_elems > ARRAY_SIZE(output_q)) {
		printk("ERR: output bigger than scratch (%zu)\n", out_elems);
		return -1;
	}

	const uint32_t t0 = k_cycle_get_32();
	r = nrf_axon_nn_model_infer_sync(&model_cnn_mel,
					 (int8_t *)g_sample_input, output_q);
	const uint32_t t1 = k_cycle_get_32();
	if (r != NRF_AXON_RESULT_SUCCESS) {
		printk("ERR: infer failed (%d)\n", r);
		return -1;
	}

	const uint64_t cycles = (uint64_t)(t1 - t0);
	const uint64_t ns = k_cyc_to_ns_floor64(cycles);
	printk("inference: %llu us (%llu cycles)\n",
	       (unsigned long long)(ns / 1000), (unsigned long long)cycles);

	float scores[8];
	dequantize(output_q, out_elems, scores, &model_cnn_mel);

	int best = 0;
	for (size_t i = 0; i < out_elems; i++) {
		const char *name = (i < ARRAY_SIZE(k_class_names)) ? k_class_names[i] : "?";
		printk("  %s: %.4f\n", name, (double)scores[i]);
		if (scores[i] > scores[best]) { best = (int)i; }
	}

	const char *predicted =
		((size_t)best < ARRAY_SIZE(k_class_names)) ? k_class_names[best] : "?";
	printk("predicted: %s   expected: %s\n", predicted, g_sample_expected_label);
	printk("=== done ===\n");
	return 0;
}
