/*
 * Axon-backed FFT frontend for mel_compute (alternative to mel_fft_cpu.c).
 *
 * The Axon FFT primitive caps at 512 points (axon_fft_24 rejects length_log2
 * >= 10), so we synthesize the 1024-point real FFT from two 512-point complex
 * FFTs on the even/odd samples and recombine with the twiddle table:
 *   X[k] = E[k] + W_N^k * O[k]
 *
 * The xty/memset output-stride helpers also cap at length=256 with stride=1,
 * so the windowing pass is chunked. See project_axon_fft_512_limit.md.
 */

#include <math.h>
#include <stdint.h>

#include <zephyr/sys/printk.h>

#include <drivers/axon/nrf_axon_dsp_intrinsics.h>
#include <drivers/axon/nrf_axon_driver.h>

#include "sample_input.h"
#include "mel_compute.h"

#define XTY_CHUNK         256
#define SAT_CHUNK         512
#define HALF_LEN          (MEL_FRAME_LENGTH / 2)
#define HALF_LOG2         9
#define ROUND_WINDOW      8
#define POWER_SHIFT_AXON  22
#define POLL              NRF_AXON_SYNC_MODE_BLOCKING_POLLING

const float mel_fft_log_offset =
	(0.5f * (float)POWER_SHIFT_AXON - 10.0f) * 0.69314718056f;

const char *const mel_backend_name = "axon";

static const float   k_power_scale = 1.0f / (float)(1ull << POWER_SHIFT_AXON);
static const int32_t k_int24_cap   = (1 << 23) - 1;

static int16_t even_in[HALF_LEN];
static int16_t odd_in[HALF_LEN];
static int32_t e_fft[2 * HALF_LEN];
static int32_t o_fft[2 * HALF_LEN];
static float   power_f[MEL_FFT_BINS];

#define AXON_TRY(step, expr) do { \
	r = (expr); \
	if (r) { printk("  %s -> %d\n", (step), (int)r); return r; } \
} while (0)

static int window_half(const int16_t *src, const int16_t *hann, int32_t *dst)
{
	nrf_axon_result_e r;
	for (int off = 0; off < HALF_LEN; off += XTY_CHUNK) {
		AXON_TRY("xty", axon_xty_16_16_32_output_stride(
			&src[off], &hann[off],
			&dst[2 * off], XTY_CHUNK, ROUND_WINDOW, 1, POLL, true));
		AXON_TRY("memset", axon_memset_32_output_stride(
			0, &dst[2 * off + 1], XTY_CHUNK, 1, POLL, true));
	}
	for (int off = 0; off < 2 * HALF_LEN; off += SAT_CHUNK) {
		AXON_TRY("saturate", axon_saturate_32_24(
			&dst[off], &dst[off], SAT_CHUNK, POLL, true));
	}
	return 0;
}

int mel_fft_frame(const int16_t *frame_in, int32_t *power_out)
{
	for (int n = 0; n < HALF_LEN; n++) {
		even_in[n] = frame_in[2 * n];
		odd_in[n]  = frame_in[2 * n + 1];
	}

	int err = window_half(even_in, g_hann_even_q15, e_fft);
	if (err) return err;
	err = window_half(odd_in, g_hann_odd_q15, o_fft);
	if (err) return err;

	nrf_axon_result_e r;
	AXON_TRY("fft_e", axon_fft_24(e_fft, e_fft, HALF_LOG2, false, false, POLL, true));
	AXON_TRY("fft_o", axon_fft_24(o_fft, o_fft, HALF_LOG2, false, false, POLL, true));

	for (int k = 0; k < MEL_FFT_BINS; k++) {
		float Ere = (float)e_fft[2 * k];
		float Eim = (float)e_fft[2 * k + 1];
		float Ore = (float)o_fft[2 * k];
		float Oim = (float)o_fft[2 * k + 1];
		float Wre = g_twiddle_f32[2 * k];
		float Wim = g_twiddle_f32[2 * k + 1];
		float Tre = Wre * Ore - Wim * Oim;
		float Tim = Wre * Oim + Wim * Ore;
		float Xre = Ere + Tre;
		float Xim = Eim + Tim;
		power_f[k] = Xre * Xre + Xim * Xim;
	}

	for (int k = 0; k < MEL_FFT_BINS; k++) {
		int32_t q = (int32_t)lroundf(power_f[k] * k_power_scale);
		if (q > k_int24_cap)  q = k_int24_cap;
		if (q < -k_int24_cap) q = -k_int24_cap;
		power_out[k] = q;
	}
	return 0;
}
