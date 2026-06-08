/*
 * CMSIS-DSP FFT frontend for mel_compute (reference baseline; runs entirely
 * on the CPU). Used as ground truth for the Axon-split backend.
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#include <arm_math.h>

#include "sample_input.h"
#include "mel_compute.h"

#define POWER_SHIFT_CPU  8

const float mel_fft_log_offset =
	(12.0f - 0.5f * (30.0f - (float)POWER_SHIFT_CPU)) * 0.69314718056f;

const char *const mel_backend_name = "cpu";

static const float   k_power_scale = 1.0f / (float)(1u << POWER_SHIFT_CPU);
static const int32_t k_int24_cap   = (1 << 23) - 1;

static arm_rfft_fast_instance_f32 rfft_inst;
static bool rfft_inited;

static float win_f[MEL_FRAME_LENGTH];
static float fft_out[MEL_FRAME_LENGTH];
static float power_f[MEL_FFT_BINS];

int mel_fft_frame(const int16_t *frame_in, int32_t *power_out)
{
	if (!rfft_inited) {
		if (arm_rfft_fast_init_f32(&rfft_inst, MEL_FRAME_LENGTH) != ARM_MATH_SUCCESS) {
			return -1;
		}
		rfft_inited = true;
	}

	for (int n = 0; n < MEL_FRAME_LENGTH; n++) {
		win_f[n] = (float)frame_in[n] * g_hann_f32[n];
	}

	arm_rfft_fast_f32(&rfft_inst, win_f, fft_out, 0);

	/* arm_rfft_fast_f32 packs DC at [0] and Nyquist at [1]; we drop Nyquist. */
	power_f[0] = fft_out[0] * fft_out[0];
	arm_cmplx_mag_squared_f32(&fft_out[2], &power_f[1], MEL_FFT_BINS - 1);

	for (int k = 0; k < MEL_FFT_BINS; k++) {
		int32_t q = (int32_t)lroundf(power_f[k] * k_power_scale);
		if (q > k_int24_cap)  q = k_int24_cap;
		if (q < -k_int24_cap) q = -k_int24_cap;
		power_out[k] = q;
	}
	return 0;
}
