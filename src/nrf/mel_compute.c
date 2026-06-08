/*
 * On-device log-mel pipeline shared by both FFT backends.
 *
 * Per frame:
 *   raw int16 PCM
 *     -> mel_fft_frame()                    (backend-specific, power spectrum)
 *     -> axon_sqrt_24  -> magnitude
 *     -> mar/marx_*    -> mel-filterbank projection
 *     -> axon_logn     -> natural log (Q11.12)
 *     -> requantize to int8 using the TFLite input quant (scale, zp)
 *
 * mel_fft_log_offset and the input quant constants are supplied by the
 * selected backend / generated_data.
 */

#include <math.h>
#include <stdint.h>

#include <zephyr/sys/printk.h>

#include <drivers/axon/nrf_axon_dsp_intrinsics.h>
#include <drivers/axon/nrf_axon_driver.h>

#include "sample_input.h"
#include "mel_compute.h"

#define ROUND_MEL  15
#define POLL       NRF_AXON_SYNC_MODE_BLOCKING_POLLING

static int32_t power_buf[MEL_FFT_BINS];
static int32_t mag_buf[MEL_FFT_BINS];
static int32_t mel_buf[MEL_NUM_BINS];
static int32_t mel_q1112[MEL_NUM_BINS];
static int32_t log_q1112[MEL_NUM_BINS];

static int power_to_mel(int32_t *mel_out)
{
	nrf_axon_result_e r;

	r = axon_sqrt_24(power_buf, mag_buf, MEL_FFT_BINS, POLL, true);
	if (r) { printk("  sqrt -> %d\n", (int)r); return r; }

	r = axon_mar_16_24_32(
		&g_mel_matrix_q15[0], mag_buf,
		&mel_out[0], MEL_FFT_BINS, ROUND_MEL, POLL, true);
	if (r) { printk("  mar[0] -> %d\n", (int)r); return r; }

	for (int m = 1; m < MEL_NUM_BINS; m++) {
		bool keep = (m < MEL_NUM_BINS - 1);
		r = axon_marx_16_32(
			&g_mel_matrix_q15[m * MEL_FFT_BINS],
			&mel_out[m], MEL_FFT_BINS, ROUND_MEL, POLL, keep);
		if (r) { printk("  marx[%d] -> %d\n", m, (int)r); return r; }
	}
	return 0;
}

int mel_compute(const int16_t *audio, int8_t *out_int8)
{
	const float in_scale = g_input_quant_scale;
	const int32_t in_zp = g_input_quant_zp;

	for (int f = 0; f < MEL_TARGET_FRAMES; f++) {
		int err = mel_fft_frame(&audio[f * MEL_FRAME_STEP], power_buf);
		if (err) {
			printk("  frame %d: fft -> %d\n", f, err);
			return err;
		}

		err = power_to_mel(mel_buf);
		if (err) {
			printk("  frame %d: mel -> %d\n", f, err);
			return err;
		}

		for (int m = 0; m < MEL_NUM_BINS; m++) {
			int32_t v = mel_buf[m];
			mel_q1112[m] = v < 1 ? 1 : v;
		}

		nrf_axon_result_e r = axon_logn_11p12(
			mel_q1112, log_q1112, MEL_NUM_BINS, POLL, false);
		if (r) {
			printk("  frame %d: logn -> %d\n", f, (int)r);
			return r;
		}

		int8_t *row = &out_int8[f * MEL_NUM_BINS];
		for (int m = 0; m < MEL_NUM_BINS; m++) {
			float log_mel = (float)log_q1112[m] / 4096.0f + mel_fft_log_offset;
			int32_t q = (int32_t)lroundf(log_mel / in_scale) + in_zp;
			if (q < -128) q = -128;
			if (q > 127)  q = 127;
			row[m] = (int8_t)q;
		}
	}
	return 0;
}
