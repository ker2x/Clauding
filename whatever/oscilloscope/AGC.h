/*
 * AGC.h - Automatic Gain Control
 *
 * Pure C implementation of AGC algorithm for testability.
 * Used by AudioCaptureManager for real-time audio level normalization.
 *
 * Algorithm Overview:
 * 1. Peak Detection: Fast attack, slow decay envelope follower
 * 2. Gain Calculation: Target level / peak level, clamped to [MIN, MAX]
 * 3. Gain Smoothing: Asymmetric smoothing (slower attack, faster release)
 * 4. Soft Clipping: tanh() to prevent harsh digital distortion
 */

#ifndef AGC_H
#define AGC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// AGC Configuration Constants
#define AGC_TARGET_LEVEL    0.5f   // Target peak level after normalization
#define AGC_MAX_GAIN        50.0f  // Maximum gain multiplier
#define AGC_MIN_GAIN        1.0f   // Minimum gain (no attenuation below 1x)
#define AGC_PEAK_THRESHOLD  0.0001f // Below this, signal is considered silence

// Peak tracking coefficients (fast attack, slow decay)
#define AGC_ATTACK_COEFF    0.5f   // Fast attack: 50% new, 50% old
#define AGC_DECAY_COEFF     0.02f  // Slow decay: 2% new, 98% old

// Gain smoothing coefficients (asymmetric)
#define AGC_GAIN_ATTACK     0.05f  // Gain decrease (loud signal): slow
#define AGC_GAIN_RELEASE    0.10f  // Gain increase (quiet signal): faster

// AGC state structure
typedef struct {
    float peak_level;      // Current tracked peak level
    float current_gain;    // Current smoothed gain value
    float target_level;    // Target output level (configurable)
    float max_gain;        // Maximum gain limit
    float min_gain;        // Minimum gain limit
} AGCState;

/**
 * Initialize AGC state with default parameters.
 * @param state Pointer to AGCState structure
 */
void agc_init(AGCState *state);

/**
 * Initialize AGC state with custom parameters.
 * @param state Pointer to AGCState structure
 * @param target_level Desired output level (0.0 to 1.0)
 * @param min_gain Minimum gain multiplier
 * @param max_gain Maximum gain multiplier
 * @param initial_peak Initial peak level estimate
 */
void agc_init_custom(AGCState *state, float target_level, float min_gain,
                     float max_gain, float initial_peak);

/**
 * Find peak (maximum absolute value) in a buffer.
 * @param samples Input audio buffer
 * @param count Number of samples
 * @return Peak absolute value in range [0, inf)
 */
float agc_find_peak(const float *samples, uint32_t count);

/**
 * Update peak level using envelope follower (fast attack, slow decay).
 * @param state AGC state
 * @param new_peak Current frame's peak value
 * @return Updated peak level
 */
float agc_update_peak(AGCState *state, float new_peak);

/**
 * Calculate desired gain based on current peak level.
 * @param state AGC state
 * @return Desired gain value (clamped to [min_gain, max_gain])
 */
float agc_calculate_gain(const AGCState *state);

/**
 * Smooth gain transition (asymmetric attack/release).
 * @param state AGC state
 * @param desired_gain Target gain value
 * @return Smoothed gain value
 */
float agc_smooth_gain(AGCState *state, float desired_gain);

/**
 * Apply soft clipping using tanh() to prevent harsh distortion.
 * @param sample Input sample
 * @return Soft-clipped sample in range (-1, 1)
 */
float agc_soft_clip(float sample);

/**
 * Process audio buffer with AGC.
 * Applies gain and soft clipping in-place.
 * @param state AGC state (updated during processing)
 * @param samples Audio buffer (modified in-place)
 * @param count Number of samples
 * @param agc_enabled If false, uses fixed gain from state->current_gain
 */
void agc_process(AGCState *state, float *samples, uint32_t count,
                 int agc_enabled);

/**
 * Get current gain value (for display/debugging).
 * @param state AGC state
 * @return Current gain multiplier
 */
float agc_get_gain(const AGCState *state);

/**
 * Set manual gain value (used when AGC is disabled).
 * @param state AGC state
 * @param gain Manual gain value
 */
void agc_set_manual_gain(AGCState *state, float gain);

#ifdef __cplusplus
}
#endif

#endif // AGC_H
