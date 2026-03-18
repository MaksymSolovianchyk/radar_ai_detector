/*
 * fft.c
 *
 * CMSIS-DSP complex FFT module for IQ radar data.
 *
 * Library used: libarm_cortexM7lfdp_math.a  (M7 hard-float double-precision)
 * Runs correctly on Cortex-M55 via backward compatibility.
 * Link flag required in STM32CubeIDE:
 *   Project → Properties → C/C++ Build → Settings → MCU GCC Linker →
 *   Libraries: add  arm_cortexM7lfdp_math
 *   Library search path: add path to the .a file
 */

#include "fft.h"
#include "main.h"           /* huart1, Error_Handler, GPIO defines          */

/* CMSIS-DSP ----------------------------------------------------------------
 * Define the target before including arm_math.h so the correct
 * instruction set and twiddle tables are compiled in.
 * ARM_MATH_CM7 is correct for libarm_cortexM7lfdp_math.a
 * (The library runs on M55 via backward compatibility.)
 * ---------------------------------------------------------------------- */
#define ARM_MATH_CM7
#define __FPU_PRESENT 1
#include "arm_math.h"

/* Pre-built FFT instance structs (CMSIS-DSP < v1.9 pattern).
 * arm_cfft_init_f32() does not exist in older SDK versions;
 * use the pre-declared const structs instead.                         */
#if   FFT_N == 256
extern const arm_cfft_instance_f32 arm_cfft_sR_f32_len256;
#define FFT_INSTANCE  arm_cfft_sR_f32_len256
#elif FFT_N == 512
extern const arm_cfft_instance_f32 arm_cfft_sR_f32_len512;
#define FFT_INSTANCE  arm_cfft_sR_f32_len512
#elif FFT_N == 1024
extern const arm_cfft_instance_f32 arm_cfft_sR_f32_len1024;
#define FFT_INSTANCE  arm_cfft_sR_f32_len1024
#elif FFT_N == 2048
extern const arm_cfft_instance_f32 arm_cfft_sR_f32_len2048;
#define FFT_INSTANCE  arm_cfft_sR_f32_len2048
#elif FFT_N == 4096
extern const arm_cfft_instance_f32 arm_cfft_sR_f32_len4096;
#define FFT_INSTANCE  arm_cfft_sR_f32_len4096
#else
#error "FFT_N must be 256, 512, 1024, 2048 or 4096"
#endif

#include <string.h>
#include <math.h>

/* ── External handles from main.c ──────────────────────────────────────── */
extern UART_HandleTypeDef huart1;

/* ── Private constants ──────────────────────────────────────────────────── */
#define FFT_N2          (FFT_N * 2u)        /* interleaved re/im length     */
#define DC_GUARD_BINS   2u                  /* bins around DC to skip when
                                               searching for peak frequency  */

/* ── Private buffers ────────────────────────────────────────────────────── */

/* Interleaved IQ input / in-place FFT working buffer.
 * Layout: [re0, im0, re1, im1, ... re(N-1), im(N-1)]
 * Marked volatile for the fields written from ISR context.             */
static float          fft_input[FFT_N2];

/* Magnitude output after arm_cmplx_mag_f32, fftshifted.                */
static float          fft_mag[FFT_N];

/* Precomputed Hamming window coefficients.                              */
static float          fft_window[FFT_N];

/* UART TX packet buffer.  Built by FFT_Process(), sent by FFT_Transmit().*/
static uint8_t        fft_tx_buf[FFT_PACKET_BYTES];

/* FFT_INSTANCE macro points to the pre-built CMSIS-DSP const struct.   */

/* ── Private state ──────────────────────────────────────────────────────── */
static volatile uint32_t  sample_count  = 0;   /* samples written so far   */
static volatile bool      frame_ready   = false;/* N samples collected flag */
static volatile bool      tx_done       = true; /* UART DMA idle flag        */


/* ── Private helpers ────────────────────────────────────────────────────── */

/**
 * @brief  XOR checksum over a byte array.
 */
static uint8_t checksum_xor(const uint8_t *buf, uint32_t len)
{
    uint8_t cs = 0;
    for (uint32_t i = 0; i < len; i++) cs ^= buf[i];
    return cs;
}

/**
 * @brief  In-place FFT shift: swap left and right halves of fft_mag so that
 *         DC moves from index 0 to index FFT_N/2, matching the Python
 *         convention used in the existing live_fft.py script.
 */
static void fftshift(float *buf, uint32_t n)
{
    uint32_t half = n / 2u;
    float    tmp;
    for (uint32_t i = 0; i < half; i++)
    {
        tmp            = buf[i];
        buf[i]         = buf[i + half];
        buf[i + half]  = tmp;
    }
}


/* ── Public API implementation ──────────────────────────────────────────── */

void FFT_Init(void)
{
    /* 1. Precompute Hamming window: w[n] = 0.54 - 0.46*cos(2π*n/(N-1))   */
    for (uint32_t n = 0; n < FFT_N; n++)
    {
        fft_window[n] = 0.54f - 0.46f * cosf(2.0f * (float)M_PI
                                              * (float)n / (float)(FFT_N - 1u));
    }

    /* 2. FFT instance: pre-built const struct selected at compile time
     *    via FFT_INSTANCE macro above. No runtime init required.         */

    /* 3. Clear buffers and state.                                         */
    memset(fft_input,  0, sizeof(fft_input));
    memset(fft_mag,    0, sizeof(fft_mag));
    memset(fft_tx_buf, 0, sizeof(fft_tx_buf));
    sample_count = 0;
    frame_ready  = false;
    tx_done      = true;
}


void FFT_PushSample(float i, float q)
{
    /* Ignore incoming samples if buffer is full and not yet processed.    */
    if (frame_ready) return;

    uint32_t idx = sample_count;
    if (idx < FFT_N)
    {
        fft_input[idx * 2u]      = i;   /* real part                       */
        fft_input[idx * 2u + 1u] = q;   /* imaginary part                  */
        sample_count++;

        if (sample_count >= FFT_N)
        {
            frame_ready = true;
        }
    }
}


bool FFT_IsReady(void)
{
    return frame_ready;
}


void FFT_Process(void)
{
    if (!frame_ready) return;

    // ── 0. Remove DC: subtract mean of I and Q separately ────────────────
    float sum_i = 0.0f, sum_q = 0.0f;
    for (uint32_t n = 0; n < FFT_N; n++)
    {
        sum_i += fft_input[n * 2u];
        sum_q += fft_input[n * 2u + 1u];
    }
    float mean_i = sum_i / (float)FFT_N;
    float mean_q = sum_q / (float)FFT_N;
    for (uint32_t n = 0; n < FFT_N; n++)
    {
        fft_input[n * 2u]      -= mean_i;
        fft_input[n * 2u + 1u] -= mean_q;
    }

    for (uint32_t n = 0; n < FFT_N; n++)
    {
        fft_input[n * 2u]      *= fft_window[n];   /* real                 */
        fft_input[n * 2u + 1u] *= fft_window[n];   /* imag                 */
    }

    /* ── 2. In-place complex FFT ─────────────────────────────────────── */
    /* ifftFlag = 0  → forward FFT
     * bitReverseFlag = 1  → apply bit-reversal (required for correct output) */
    arm_cfft_f32(&FFT_INSTANCE, fft_input, 0, 1);

    /* ── 3. Compute magnitude: sqrt(re^2 + im^2) for each bin ────────── */
    /* fft_input still holds the complex spectrum [re0,im0, re1,im1, ...] */
    arm_cmplx_mag_f32(fft_input, fft_mag, FFT_N);

    /* ── 4. FFT shift: move DC from index 0 to index FFT_N/2 ─────────── */
    fftshift(fft_mag, FFT_N);

    /* ── 5. Build UART TX packet ─────────────────────────────────────── */
    /*
     * Packet layout (little-endian):
     *  [0..1]   sync word   0x55 0xAA  (LSB first)
     *  [2..3]   bin count   FFT_N as uint16_t
     *  [4..4+FFT_N*4-1]  FFT_N × float32 magnitudes
     *  [last]   XOR checksum of bytes [4..4+FFT_N*4-1]
     */
    uint16_t sync  = (uint16_t)FFT_SYNC_WORD;
    uint16_t count = (uint16_t)FFT_N;

    fft_tx_buf[0] = (uint8_t)(sync  & 0xFFu);
    fft_tx_buf[1] = (uint8_t)(sync  >> 8u);
    fft_tx_buf[2] = (uint8_t)(count & 0xFFu);
    fft_tx_buf[3] = (uint8_t)(count >> 8u);

    /* Copy float magnitudes as raw bytes.                                 */
    memcpy(&fft_tx_buf[4], fft_mag, FFT_N * sizeof(float));

    /* Checksum over the magnitude payload only.                           */
    fft_tx_buf[FFT_PACKET_BYTES - 1u] =
        checksum_xor(&fft_tx_buf[4], FFT_N * sizeof(float));
}


void FFT_Transmit(void)
{
    tx_done = false;
    HAL_StatusTypeDef status = HAL_UART_Transmit_DMA(
        &huart1,
        fft_tx_buf,
        (uint16_t)FFT_PACKET_BYTES);

    if (status != HAL_OK)
    {
        /* DMA busy or error — mark done so caller can retry.              */
        tx_done = true;
    }
}


bool FFT_TransmitDone(void)
{
    return tx_done;
}


void FFT_Reset(void)
{
    /* Clear the input buffer to avoid stale data leaking into next frame. */
    memset(fft_input, 0, sizeof(fft_input));
    sample_count = 0;
    frame_ready  = false;
}


void FFT_TxCompleteCallback(void)
{
    tx_done = true;
}


const float* FFT_GetMagnitude(void)
{
    return fft_mag;
}


float FFT_GetPeakFrequency(void)
{
    float    peak_mag = 0.0f;
    uint32_t peak_idx = FFT_N / 2u;   /* start at DC bin                  */
    float    bin_hz   = (float)FFT_FS / (float)FFT_N;

    /* After fftshift: DC is at index FFT_N/2.
     * Skip DC_GUARD_BINS on either side of DC.                           */
    uint32_t dc = FFT_N / 2u;

    for (uint32_t k = 0; k < FFT_N; k++)
    {
        /* Skip bins close to DC.                                          */
        if (k >= (dc - DC_GUARD_BINS) && k <= (dc + DC_GUARD_BINS)) continue;

        if (fft_mag[k] > peak_mag)
        {
            peak_mag = fft_mag[k];
            peak_idx = k;
        }
    }

    /* Convert bin index to signed Hz:
     * index 0        → -(FFT_FS/2) Hz
     * index FFT_N/2  → 0 Hz
     * index FFT_N-1  → +(FFT_FS/2 - bin_hz) Hz                          */
    float freq_hz = ((float)peak_idx - (float)dc) * bin_hz;
    return freq_hz;
}
