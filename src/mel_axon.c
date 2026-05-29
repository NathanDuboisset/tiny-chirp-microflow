#include <math.h>
#include <stdint.h>
#include <string.h>

#include <zephyr/sys/printk.h>

#include <drivers/axon/nrf_axon_dsp_intrinsics.h>
#include <drivers/axon/nrf_axon_driver.h>

#include "sample_input.h"
#include "mel_axon.h"

#ifndef MEL_FROM_RAW
#error "mel_axon.c requires MEL_FROM_RAW; data symbols are guarded by it."
#endif

#define XTY_CHUNK 512
#define VEC_CAP   512

#define ROUND_WINDOW 8
#define ROUND_FFT    0
#define ROUND_MEL    15
#define MEL_RADIX_BITS 12

#define POLL NRF_AXON_SYNC_MODE_BLOCKING_POLLING

static int32_t fft_buf[2 * MEL_FRAME_LENGTH];
static int32_t power_buf[MEL_FFT_BINS];
static int32_t mag_buf[MEL_FFT_BINS];
static int32_t mel_buf[MEL_NUM_BINS];
static int32_t mel_q1112[MEL_NUM_BINS];
static int32_t log_q1112[MEL_NUM_BINS];

static int compute_one_frame(const int16_t *frame_in, int32_t *mel_out)
{
	nrf_axon_result_e r;

	r = axon_xty_16_16_32_output_stride(
		&frame_in[0], &g_hann_window_q15[0],
		&fft_buf[0], XTY_CHUNK, ROUND_WINDOW, 1, POLL, true);
	if (r) return r;
	r = axon_xty_16_16_32_output_stride(
		&frame_in[XTY_CHUNK], &g_hann_window_q15[XTY_CHUNK],
		&fft_buf[2 * XTY_CHUNK], XTY_CHUNK, ROUND_WINDOW, 1, POLL, true);
	if (r) return r;

	r = axon_memset_32_output_stride(0, &fft_buf[1], XTY_CHUNK, 1, POLL, true);
	if (r) return r;
	r = axon_memset_32_output_stride(0, &fft_buf[2 * XTY_CHUNK + 1],
	                                 XTY_CHUNK, 1, POLL, true);
	if (r) return r;

	for (int off = 0; off < 2 * MEL_FRAME_LENGTH; off += VEC_CAP) {
		int n = (off + VEC_CAP <= 2 * MEL_FRAME_LENGTH)
		            ? VEC_CAP : (2 * MEL_FRAME_LENGTH - off);
		r = axon_saturate_32_24(&fft_buf[off], &fft_buf[off], n, POLL, true);
		if (r) return r;
	}

	r = axon_fft_power_24(&fft_buf[0], &power_buf[0],
	                      MEL_FFT_LENGTH_LOG2, true,
	                      ROUND_FFT, POLL, true);
	if (r) return r;

	r = axon_sqrt_24(&power_buf[0], &mag_buf[0], MEL_FFT_BINS, POLL, true);
	if (r) return r;

	for (int m = 0; m < MEL_NUM_BINS; m++) {
		bool keep = (m < MEL_NUM_BINS - 1);
		r = axon_mar_16_24_32(
			&g_mel_matrix_q15[m * MEL_FFT_BINS], &mag_buf[0],
			&mel_out[m], MEL_FFT_BINS, ROUND_MEL, POLL, keep);
		if (r) return r;
	}
	return 0;
}

int mel_axon_compute(int8_t *out_int8)
{
	const float in_scale = g_input_quant_scale;
	const int32_t in_zp = g_input_quant_zp;
	const float log_offset = -(float)(MEL_RADIX_BITS - 12) * logf(2.0f);

	for (int f = 0; f < MEL_TARGET_FRAMES; f++) {
		const int16_t *frame_in =
			&g_sample_raw_audio[f * MEL_FRAME_STEP];

		int err = compute_one_frame(frame_in, mel_buf);
		if (err) {
			printk("mel_axon frame %d failed: %d\n", f, err);
			return err;
		}

		for (int m = 0; m < MEL_NUM_BINS; m++) {
			int32_t v = mel_buf[m];
			if (v < 1) v = 1;
			mel_q1112[m] = v;
		}

		nrf_axon_result_e r = axon_logn_11p12(
			mel_q1112, log_q1112, MEL_NUM_BINS, POLL, false);
		if (r) {
			printk("axon_logn_11p12 frame %d failed: %d\n", f, r);
			return r;
		}

		int8_t *row = &out_int8[f * MEL_NUM_BINS];
		for (int m = 0; m < MEL_NUM_BINS; m++) {
			float log_mel = (float)log_q1112[m] / 4096.0f + log_offset;
			int32_t q = (int32_t)lroundf(log_mel / in_scale) + in_zp;
			if (q < -128) q = -128;
			if (q > 127)  q = 127;
			row[m] = (int8_t)q;
		}
	}
	return 0;
}
