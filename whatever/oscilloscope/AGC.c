/*
 * AGC.c - Automatic Gain Control Implementation
 *
 * Real-time safe AGC for audio normalization.
 * No allocations, no locks - suitable for audio thread callbacks.
 */

#include "AGC.h"
#include <math.h>

void agc_init(AGCState *state) {
    if (!state) return;

    state->peak_level = 0.05f;  // Start with small non-zero to avoid division issues
    state->current_gain = 10.0f; // Start with moderate gain
    state->target_level = AGC_TARGET_LEVEL;
    state->max_gain = AGC_MAX_GAIN;
    state->min_gain = AGC_MIN_GAIN;
}

void agc_init_custom(AGCState *state, float target_level, float min_gain,
                     float max_gain, float initial_peak) {
    if (!state) return;

    state->peak_level = initial_peak > 0.0f ? initial_peak : 0.05f;
    state->current_gain = target_level / state->peak_level;
    state->target_level = target_level;
    state->max_gain = max_gain;
    state->min_gain = min_gain;

    // Clamp initial gain
    if (state->current_gain > max_gain) state->current_gain = max_gain;
    if (state->current_gain < min_gain) state->current_gain = min_gain;
}

float agc_find_peak(const float *samples, uint32_t count) {
    if (!samples || count == 0) return 0.0f;

    float peak = 0.0f;
    for (uint32_t i = 0; i < count; i++) {
        float abs_val = fabsf(samples[i]);
        if (abs_val > peak) {
            peak = abs_val;
        }
    }
    return peak;
}

float agc_update_peak(AGCState *state, float new_peak) {
    if (!state) return 0.0f;

    // Fast attack, slow decay envelope follower
    // When signal gets louder, track quickly to prevent clipping
    // When signal gets quieter, decay slowly to avoid pumping
    if (new_peak > state->peak_level) {
        // Fast attack: 50% new + 50% old
        state->peak_level = state->peak_level * (1.0f - AGC_ATTACK_COEFF)
                          + new_peak * AGC_ATTACK_COEFF;
    } else {
        // Slow decay: 2% new + 98% old
        state->peak_level = state->peak_level * (1.0f - AGC_DECAY_COEFF)
                          + new_peak * AGC_DECAY_COEFF;
    }

    return state->peak_level;
}

float agc_calculate_gain(const AGCState *state) {
    if (!state) return 1.0f;

    // Avoid division by zero / near-zero (silence detection)
    if (state->peak_level < AGC_PEAK_THRESHOLD) {
        return state->max_gain;  // Max gain for silence (will be soft-clipped anyway)
    }

    // Calculate desired gain to reach target level
    float desired_gain = state->target_level / state->peak_level;

    // Clamp to valid range
    if (desired_gain > state->max_gain) desired_gain = state->max_gain;
    if (desired_gain < state->min_gain) desired_gain = state->min_gain;

    return desired_gain;
}

float agc_smooth_gain(AGCState *state, float desired_gain) {
    if (!state) return desired_gain;

    // Asymmetric smoothing:
    // - Gain DECREASE (loud signal): slower to avoid sudden volume drops
    // - Gain INCREASE (quiet signal): faster to recover from loud transients
    if (desired_gain < state->current_gain) {
        // Decreasing gain (attacking loud signal)
        state->current_gain = state->current_gain * (1.0f - AGC_GAIN_ATTACK)
                            + desired_gain * AGC_GAIN_ATTACK;
    } else {
        // Increasing gain (releasing after loud signal)
        state->current_gain = state->current_gain * (1.0f - AGC_GAIN_RELEASE)
                            + desired_gain * AGC_GAIN_RELEASE;
    }

    return state->current_gain;
}

float agc_soft_clip(float sample) {
    // tanh provides smooth saturation curve:
    // - Linear near zero (preserves quiet signals)
    // - Gradually compresses toward ±1 (prevents harsh clipping)
    // - Output always in range (-1, 1)
    return tanhf(sample);
}

void agc_process(AGCState *state, float *samples, uint32_t count,
                 int agc_enabled) {
    if (!state || !samples || count == 0) return;

    if (agc_enabled) {
        // Step 1: Find peak in current buffer
        float new_peak = agc_find_peak(samples, count);

        // Step 2: Update peak level with envelope follower
        agc_update_peak(state, new_peak);

        // Step 3: Calculate desired gain
        float desired_gain = agc_calculate_gain(state);

        // Step 4: Smooth gain transition
        agc_smooth_gain(state, desired_gain);
    }
    // If AGC disabled, current_gain remains at manually set value

    // Step 5: Apply gain and soft clipping to all samples
    for (uint32_t i = 0; i < count; i++) {
        samples[i] *= state->current_gain;
        samples[i] = agc_soft_clip(samples[i]);
    }
}

float agc_get_gain(const AGCState *state) {
    if (!state) return 1.0f;
    return state->current_gain;
}

void agc_set_manual_gain(AGCState *state, float gain) {
    if (!state) return;
    state->current_gain = gain;
}
