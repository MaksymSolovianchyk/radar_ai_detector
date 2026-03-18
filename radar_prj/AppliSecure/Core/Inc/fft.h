/*
 * fft.h
 *
 * CMSIS-DSP complex FFT module for IQ radar data.
 * CH0 → real (I),  CH1 → imaginary (Q)
 *
 * Usage:
 *   1. Call FFT_Init() once at startup.
 *   2. Call FFT_PushSample(i, q) from the DMA ISR for every new sample.
 *   3. In the main loop, check FFT_IsReady() — when true call FFT_Process()
 *      which computes the FFT, shifts bins to centre at 0 Hz, and prepares
 *      the UART TX packet.  Then call FFT_Transmit() to send it.
 *   4. After transmit completes call FFT_Reset() to start next frame.
 */

#ifndef FFT_H_
#define FFT_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

/* ── Configuration ──────────────────────────────────────────────────────────
 * FFT_N must be a power of two: 256 / 512 / 1024 / 2048.
 * Increasing N improves frequency resolution but increases collect time:
 *   N=512  → 7.6 Hz/bin,  ~131 ms collect,  ~6 frames/s
 *   N=1024 → 3.8 Hz/bin,  ~262 ms collect,  ~3 frames/s   ← default
 *   N=2048 → 1.9 Hz/bin,  ~525 ms collect,  ~1.5 frames/s
 * ──────────────────────────────────────────────────────────────────────────*/
#define FFT_N           1024u

/* Sampling rate must match what the ADS131M04 is configured to output.    */
#define FFT_FS          3904u       /* Hz                                   */

/* Packet framing sent over UART for each FFT frame.                       */
#define FFT_SYNC_WORD   0xAA55u     /* 2-byte little-endian sync header     */

/* Total UART packet size in bytes:
 *   2  sync
 *   2  bin count  (= FFT_N, uint16)
 *   FFT_N * 4  magnitude floats
 *   1  checksum  (XOR of all payload bytes)
 * ── ────────────────────────────────────────────────────────────────────── */
#define FFT_PACKET_BYTES  (2u + 2u + (FFT_N * 4u) + 1u)


/* ── Public API ─────────────────────────────────────────────────────────── */

/**
 * @brief  Initialise the FFT module.
 *         Precomputes the Hamming window, initialises the CMSIS-DSP instance.
 *         Call once before the main loop.
 */
void FFT_Init(void);

/**
 * @brief  Push one IQ sample into the collection buffer.
 *         Call from the DMA/EXTI ISR after each ADC conversion.
 * @param  i   Channel 0 value converted to float (volts)
 * @param  q   Channel 1 value converted to float (volts)
 */
void FFT_PushSample(float i, float q);

/**
 * @brief  Returns true when FFT_N samples have been collected and the
 *         FFT has not yet been processed for this frame.
 *         Poll this in the main loop.
 */
bool FFT_IsReady(void);

/**
 * @brief  Run the FFT pipeline on the collected buffer:
 *           1. Apply Hamming window
 *           2. arm_cfft_f32  (in-place complex FFT)
 *           3. arm_cmplx_mag_f32  (magnitude)
 *           4. fftshift  (centre DC at index N/2)
 *           5. Build UART TX packet into fft_tx_buf[]
 *         Call from the main loop when FFT_IsReady() is true.
 *         This function is NOT safe to call from an ISR.
 */
void FFT_Process(void);

/**
 * @brief  Start DMA UART transmission of the packet built by FFT_Process().
 *         Returns immediately; transmission runs in background.
 *         Call from the main loop immediately after FFT_Process().
 */
void FFT_Transmit(void);

/**
 * @brief  Returns true when the UART DMA transmission has completed and
 *         the module is ready to accept a new frame.
 */
bool FFT_TransmitDone(void);

/**
 * @brief  Reset the sample counter so the next frame can be collected.
 *         Call from the main loop after FFT_TransmitDone() is true.
 */
void FFT_Reset(void);

/**
 * @brief  Called from HAL_UART_TxCpltCallback to signal TX complete.
 *         Do not call directly.
 */
void FFT_TxCompleteCallback(void);


/* ── Read-only access to results (optional, for debug) ─────────────────── */

/**
 * @brief  Returns pointer to the magnitude array after FFT_Process().
 *         Array has FFT_N elements, index 0 = most negative frequency,
 *         index FFT_N/2 = DC, index FFT_N-1 = most positive frequency.
 */
const float* FFT_GetMagnitude(void);

/**
 * @brief  Returns the frequency (Hz) of the bin with maximum magnitude,
 *         excluding the DC bin.  Positive = approaching, negative = receding.
 */
float FFT_GetPeakFrequency(void);


#ifdef __cplusplus
}
#endif

#endif /* FFT_H_ */
