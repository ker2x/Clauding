/*
 * Unit Tests for AGC (Automatic Gain Control)
 *
 * Tests cover:
 * - Initialization (default and custom)
 * - Peak detection in buffers
 * - Peak tracking (fast attack, slow decay envelope)
 * - Gain calculation and clamping
 * - Gain smoothing (asymmetric attack/release)
 * - Soft clipping behavior
 * - Full AGC processing pipeline
 * - Edge cases and NULL handling
 */

#include "AGC.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

// Test framework macros (same as test_ringbuffer.c for consistency)
#define ANSI_RED "\x1b[31m"
#define ANSI_GREEN "\x1b[32m"
#define ANSI_YELLOW "\x1b[33m"
#define ANSI_RESET "\x1b[0m"

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name)                                                             \
  do {                                                                         \
    tests_run++;                                                               \
    printf("  %-50s ", name);                                                  \
  } while (0)

#define PASS()                                                                 \
  do {                                                                         \
    tests_passed++;                                                            \
    printf(ANSI_GREEN "[PASS]" ANSI_RESET "\n");                               \
  } while (0)

#define FAIL(msg)                                                              \
  do {                                                                         \
    printf(ANSI_RED "[FAIL]" ANSI_RESET " %s\n", msg);                         \
  } while (0)

#define ASSERT(cond, msg)                                                      \
  do {                                                                         \
    if (!(cond)) {                                                             \
      FAIL(msg);                                                               \
      return;                                                                  \
    }                                                                          \
  } while (0)

#define ASSERT_FLOAT_EQ(a, b, eps, msg)                                        \
  ASSERT(fabsf((a) - (b)) < (eps), msg)

// ============================================================================
// Test: Initialization
// ============================================================================

void test_init_default_values(void) {
    TEST("init: default values are set correctly");

    AGCState state;
    agc_init(&state);

    ASSERT_FLOAT_EQ(state.target_level, AGC_TARGET_LEVEL, 0.001f,
                    "target_level should be AGC_TARGET_LEVEL");
    ASSERT_FLOAT_EQ(state.max_gain, AGC_MAX_GAIN, 0.001f,
                    "max_gain should be AGC_MAX_GAIN");
    ASSERT_FLOAT_EQ(state.min_gain, AGC_MIN_GAIN, 0.001f,
                    "min_gain should be AGC_MIN_GAIN");
    ASSERT(state.peak_level > 0.0f, "peak_level should be non-zero");
    ASSERT(state.current_gain > 0.0f, "current_gain should be positive");
    PASS();
}

void test_init_null_safe(void) {
    TEST("init: handles NULL pointer");

    // Should not crash
    agc_init(NULL);
    agc_init_custom(NULL, 0.5f, 1.0f, 50.0f, 0.1f);
    PASS();
}

void test_init_custom_values(void) {
    TEST("init_custom: sets custom parameters");

    AGCState state;
    agc_init_custom(&state, 0.7f, 0.5f, 100.0f, 0.2f);

    ASSERT_FLOAT_EQ(state.target_level, 0.7f, 0.001f, "target_level mismatch");
    ASSERT_FLOAT_EQ(state.min_gain, 0.5f, 0.001f, "min_gain mismatch");
    ASSERT_FLOAT_EQ(state.max_gain, 100.0f, 0.001f, "max_gain mismatch");
    ASSERT_FLOAT_EQ(state.peak_level, 0.2f, 0.001f, "peak_level mismatch");
    PASS();
}

void test_init_custom_clamps_gain(void) {
    TEST("init_custom: clamps initial gain to range");

    AGCState state;
    // Very small peak would give huge gain, should be clamped to max
    agc_init_custom(&state, 0.5f, 1.0f, 10.0f, 0.001f);

    ASSERT(state.current_gain <= 10.0f, "gain should be clamped to max");
    ASSERT(state.current_gain >= 1.0f, "gain should be at least min");
    PASS();
}

// ============================================================================
// Test: Peak Detection
// ============================================================================

void test_find_peak_positive_values(void) {
    TEST("find_peak: detects max in positive values");

    float samples[] = {0.1f, 0.5f, 0.3f, 0.8f, 0.2f};
    float peak = agc_find_peak(samples, 5);

    ASSERT_FLOAT_EQ(peak, 0.8f, 0.001f, "peak should be 0.8");
    PASS();
}

void test_find_peak_negative_values(void) {
    TEST("find_peak: detects max absolute in negative values");

    float samples[] = {-0.1f, -0.9f, -0.3f, -0.2f};
    float peak = agc_find_peak(samples, 4);

    ASSERT_FLOAT_EQ(peak, 0.9f, 0.001f, "peak should be 0.9 (absolute)");
    PASS();
}

void test_find_peak_mixed_values(void) {
    TEST("find_peak: handles mixed positive/negative");

    float samples[] = {0.3f, -0.7f, 0.5f, -0.2f};
    float peak = agc_find_peak(samples, 4);

    ASSERT_FLOAT_EQ(peak, 0.7f, 0.001f, "peak should be 0.7");
    PASS();
}

void test_find_peak_silence(void) {
    TEST("find_peak: returns 0 for silence");

    float samples[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float peak = agc_find_peak(samples, 4);

    ASSERT_FLOAT_EQ(peak, 0.0f, 0.001f, "peak should be 0 for silence");
    PASS();
}

void test_find_peak_single_sample(void) {
    TEST("find_peak: works with single sample");

    float sample = -0.42f;
    float peak = agc_find_peak(&sample, 1);

    ASSERT_FLOAT_EQ(peak, 0.42f, 0.001f, "peak should be 0.42");
    PASS();
}

void test_find_peak_null_buffer(void) {
    TEST("find_peak: returns 0 for NULL buffer");

    float peak = agc_find_peak(NULL, 10);

    ASSERT_FLOAT_EQ(peak, 0.0f, 0.001f, "should return 0 for NULL");
    PASS();
}

void test_find_peak_zero_count(void) {
    TEST("find_peak: returns 0 for zero count");

    float samples[] = {0.5f, 0.8f};
    float peak = agc_find_peak(samples, 0);

    ASSERT_FLOAT_EQ(peak, 0.0f, 0.001f, "should return 0 for count=0");
    PASS();
}

// ============================================================================
// Test: Peak Tracking (Envelope Follower)
// ============================================================================

void test_update_peak_fast_attack(void) {
    TEST("update_peak: fast attack on rising signal");

    AGCState state;
    agc_init(&state);
    state.peak_level = 0.1f;  // Start low

    // New peak is much higher - should attack quickly
    float result = agc_update_peak(&state, 0.9f);

    // Fast attack: 50% old + 50% new = 0.1*0.5 + 0.9*0.5 = 0.5
    ASSERT_FLOAT_EQ(result, 0.5f, 0.01f, "should attack quickly");
    ASSERT_FLOAT_EQ(state.peak_level, 0.5f, 0.01f, "state should be updated");
    PASS();
}

void test_update_peak_slow_decay(void) {
    TEST("update_peak: slow decay on falling signal");

    AGCState state;
    agc_init(&state);
    state.peak_level = 0.9f;  // Start high

    // New peak is much lower - should decay slowly
    float result = agc_update_peak(&state, 0.1f);

    // Slow decay: 98% old + 2% new = 0.9*0.98 + 0.1*0.02 = 0.884
    ASSERT_FLOAT_EQ(result, 0.884f, 0.01f, "should decay slowly");
    PASS();
}

void test_update_peak_multiple_frames(void) {
    TEST("update_peak: tracks signal over multiple frames");

    AGCState state;
    agc_init(&state);
    state.peak_level = 0.5f;

    // Simulate sustained loud signal
    for (int i = 0; i < 10; i++) {
        agc_update_peak(&state, 0.8f);
    }

    // Should converge toward 0.8
    ASSERT(state.peak_level > 0.75f, "should approach 0.8 after repeated frames");
    ASSERT(state.peak_level <= 0.8f, "should not exceed input peak");
    PASS();
}

void test_update_peak_null_state(void) {
    TEST("update_peak: returns 0 for NULL state");

    float result = agc_update_peak(NULL, 0.5f);

    ASSERT_FLOAT_EQ(result, 0.0f, 0.001f, "should return 0 for NULL");
    PASS();
}

// ============================================================================
// Test: Gain Calculation
// ============================================================================

void test_calculate_gain_normal(void) {
    TEST("calculate_gain: computes target/peak ratio");

    AGCState state;
    agc_init(&state);
    state.target_level = 0.5f;
    state.peak_level = 0.1f;  // Quiet signal

    float gain = agc_calculate_gain(&state);

    // Expected: 0.5 / 0.1 = 5.0
    ASSERT_FLOAT_EQ(gain, 5.0f, 0.01f, "gain should be 5.0");
    PASS();
}

void test_calculate_gain_clamps_to_max(void) {
    TEST("calculate_gain: clamps to max_gain");

    AGCState state;
    agc_init(&state);
    state.target_level = 0.5f;
    state.peak_level = 0.001f;  // Very quiet - would give gain of 500
    state.max_gain = 50.0f;

    float gain = agc_calculate_gain(&state);

    ASSERT_FLOAT_EQ(gain, 50.0f, 0.01f, "should clamp to max_gain");
    PASS();
}

void test_calculate_gain_clamps_to_min(void) {
    TEST("calculate_gain: clamps to min_gain");

    AGCState state;
    agc_init(&state);
    state.target_level = 0.5f;
    state.peak_level = 2.0f;  // Very loud - would give gain of 0.25
    state.min_gain = 1.0f;

    float gain = agc_calculate_gain(&state);

    ASSERT_FLOAT_EQ(gain, 1.0f, 0.01f, "should clamp to min_gain");
    PASS();
}

void test_calculate_gain_silence_returns_max(void) {
    TEST("calculate_gain: returns max for silence");

    AGCState state;
    agc_init(&state);
    state.peak_level = 0.00001f;  // Below threshold
    state.max_gain = 50.0f;

    float gain = agc_calculate_gain(&state);

    ASSERT_FLOAT_EQ(gain, 50.0f, 0.01f, "should return max_gain for silence");
    PASS();
}

void test_calculate_gain_null_state(void) {
    TEST("calculate_gain: returns 1.0 for NULL state");

    float gain = agc_calculate_gain(NULL);

    ASSERT_FLOAT_EQ(gain, 1.0f, 0.001f, "should return 1.0 for NULL");
    PASS();
}

// ============================================================================
// Test: Gain Smoothing
// ============================================================================

void test_smooth_gain_decrease_slow(void) {
    TEST("smooth_gain: slow attack on gain decrease");

    AGCState state;
    agc_init(&state);
    state.current_gain = 10.0f;

    // Decreasing to 2.0 (loud signal detected)
    float result = agc_smooth_gain(&state, 2.0f);

    // Slow attack: 95% old + 5% new = 10*0.95 + 2*0.05 = 9.6
    ASSERT_FLOAT_EQ(result, 9.6f, 0.01f, "should decrease slowly");
    PASS();
}

void test_smooth_gain_increase_faster(void) {
    TEST("smooth_gain: faster release on gain increase");

    AGCState state;
    agc_init(&state);
    state.current_gain = 2.0f;

    // Increasing to 10.0 (quiet signal after loud)
    float result = agc_smooth_gain(&state, 10.0f);

    // Faster release: 90% old + 10% new = 2*0.9 + 10*0.1 = 2.8
    ASSERT_FLOAT_EQ(result, 2.8f, 0.01f, "should increase faster than decrease");
    PASS();
}

void test_smooth_gain_convergence(void) {
    TEST("smooth_gain: converges to target over time");

    AGCState state;
    agc_init(&state);
    state.current_gain = 1.0f;

    // Apply same target repeatedly
    for (int i = 0; i < 50; i++) {
        agc_smooth_gain(&state, 10.0f);
    }

    // Should be very close to target after many iterations
    ASSERT(state.current_gain > 9.5f, "should converge to target");
    PASS();
}

void test_smooth_gain_null_state(void) {
    TEST("smooth_gain: returns input for NULL state");

    float result = agc_smooth_gain(NULL, 5.0f);

    ASSERT_FLOAT_EQ(result, 5.0f, 0.001f, "should return desired_gain for NULL");
    PASS();
}

// ============================================================================
// Test: Soft Clipping
// ============================================================================

void test_soft_clip_small_values_linear(void) {
    TEST("soft_clip: near-linear for small values");

    // tanh(x) ≈ x for small x
    float result = agc_soft_clip(0.1f);

    ASSERT_FLOAT_EQ(result, 0.1f, 0.01f, "small values should be near-linear");
    PASS();
}

void test_soft_clip_large_values_compressed(void) {
    TEST("soft_clip: compresses large values");

    float result = agc_soft_clip(3.0f);

    // tanh(3.0) ≈ 0.995
    ASSERT(result > 0.99f, "should be close to 1.0");
    ASSERT(result < 1.0f, "should never reach 1.0");
    PASS();
}

void test_soft_clip_negative_values(void) {
    TEST("soft_clip: handles negative values symmetrically");

    float pos = agc_soft_clip(2.0f);
    float neg = agc_soft_clip(-2.0f);

    ASSERT_FLOAT_EQ(neg, -pos, 0.001f, "should be symmetric around zero");
    PASS();
}

void test_soft_clip_zero(void) {
    TEST("soft_clip: zero passes through");

    float result = agc_soft_clip(0.0f);

    ASSERT_FLOAT_EQ(result, 0.0f, 0.001f, "zero should stay zero");
    PASS();
}

void test_soft_clip_extreme_values(void) {
    TEST("soft_clip: bounds extreme values");

    float huge_pos = agc_soft_clip(1000.0f);
    float huge_neg = agc_soft_clip(-1000.0f);

    // tanh(1000) is essentially 1.0 in float precision
    ASSERT(huge_pos >= 0.999f && huge_pos <= 1.0f, "large positive bounded to ~1");
    ASSERT(huge_neg <= -0.999f && huge_neg >= -1.0f, "large negative bounded to ~-1");
    PASS();
}

// ============================================================================
// Test: Full Processing Pipeline
// ============================================================================

void test_process_applies_gain(void) {
    TEST("process: applies gain to samples");

    AGCState state;
    agc_init(&state);
    state.current_gain = 2.0f;
    state.peak_level = 0.25f;  // Will calculate gain as 2.0

    float samples[] = {0.1f, 0.2f, 0.3f};
    agc_process(&state, samples, 3, 0);  // AGC disabled, use fixed gain

    // With gain=2 and soft clipping:
    // 0.1 * 2 = 0.2 -> tanh(0.2) ≈ 0.197
    // 0.2 * 2 = 0.4 -> tanh(0.4) ≈ 0.380
    // 0.3 * 2 = 0.6 -> tanh(0.6) ≈ 0.537
    ASSERT_FLOAT_EQ(samples[0], tanhf(0.2f), 0.001f, "sample 0 incorrect");
    ASSERT_FLOAT_EQ(samples[1], tanhf(0.4f), 0.001f, "sample 1 incorrect");
    ASSERT_FLOAT_EQ(samples[2], tanhf(0.6f), 0.001f, "sample 2 incorrect");
    PASS();
}

void test_process_agc_enabled_adjusts_gain(void) {
    TEST("process: AGC enabled adjusts gain based on input");

    AGCState state;
    agc_init(&state);
    state.current_gain = 10.0f;  // Start with high gain
    state.peak_level = 0.1f;     // Low initial peak

    // Loud signal should cause gain to decrease over multiple frames
    float loud_samples[256];
    for (int i = 0; i < 256; i++) {
        loud_samples[i] = 0.9f * ((i % 2) ? 1.0f : -1.0f);  // Alternating ±0.9
    }

    float initial_gain = state.current_gain;

    // Process multiple frames to allow gain to change significantly
    for (int frame = 0; frame < 10; frame++) {
        // Regenerate samples since they get modified in-place
        for (int i = 0; i < 256; i++) {
            loud_samples[i] = 0.9f * ((i % 2) ? 1.0f : -1.0f);
        }
        agc_process(&state, loud_samples, 256, 1);  // AGC enabled
    }

    // Gain should have decreased significantly from 10.0 toward target
    ASSERT(state.current_gain < initial_gain * 0.9f, "gain should have decreased");
    PASS();
}

void test_process_prevents_clipping(void) {
    TEST("process: output stays within [-1, 1]");

    AGCState state;
    agc_init(&state);
    state.current_gain = 100.0f;  // Extreme gain

    float samples[] = {0.5f, -0.5f, 1.0f, -1.0f};
    agc_process(&state, samples, 4, 0);

    for (int i = 0; i < 4; i++) {
        ASSERT(samples[i] >= -1.0f && samples[i] <= 1.0f,
               "all samples should be in [-1, 1]");
    }
    PASS();
}

void test_process_silence_handling(void) {
    TEST("process: handles silence gracefully");

    AGCState state;
    agc_init(&state);

    float silence[100] = {0};
    agc_process(&state, silence, 100, 1);

    // Should not crash, gain should go to max
    ASSERT(state.current_gain > 0.0f, "gain should remain positive");
    for (int i = 0; i < 100; i++) {
        ASSERT_FLOAT_EQ(silence[i], 0.0f, 0.001f, "silence should stay silent");
    }
    PASS();
}

void test_process_null_handling(void) {
    TEST("process: handles NULL inputs safely");

    AGCState state;
    agc_init(&state);
    float samples[] = {0.5f};

    // None of these should crash
    agc_process(NULL, samples, 1, 1);
    agc_process(&state, NULL, 1, 1);
    agc_process(&state, samples, 0, 1);
    PASS();
}

// ============================================================================
// Test: Getters and Setters
// ============================================================================

void test_get_gain(void) {
    TEST("get_gain: returns current gain");

    AGCState state;
    agc_init(&state);
    state.current_gain = 7.5f;

    float gain = agc_get_gain(&state);

    ASSERT_FLOAT_EQ(gain, 7.5f, 0.001f, "should return current_gain");
    PASS();
}

void test_get_gain_null(void) {
    TEST("get_gain: returns 1.0 for NULL");

    float gain = agc_get_gain(NULL);

    ASSERT_FLOAT_EQ(gain, 1.0f, 0.001f, "should return 1.0 for NULL");
    PASS();
}

void test_set_manual_gain(void) {
    TEST("set_manual_gain: sets current gain");

    AGCState state;
    agc_init(&state);

    agc_set_manual_gain(&state, 15.0f);

    ASSERT_FLOAT_EQ(state.current_gain, 15.0f, 0.001f, "gain should be set");
    PASS();
}

void test_set_manual_gain_null(void) {
    TEST("set_manual_gain: handles NULL safely");

    // Should not crash
    agc_set_manual_gain(NULL, 10.0f);
    PASS();
}

// ============================================================================
// Test: Edge Cases and Stress
// ============================================================================

void test_stress_many_frames(void) {
    TEST("stress: processes 10000 frames without issues");

    AGCState state;
    agc_init(&state);

    float buffer[256];
    for (int frame = 0; frame < 10000; frame++) {
        // Generate varying signal
        float amplitude = 0.1f + 0.8f * (float)(frame % 100) / 100.0f;
        for (int i = 0; i < 256; i++) {
            buffer[i] = amplitude * sinf((float)i * 0.1f);
        }
        agc_process(&state, buffer, 256, 1);
    }

    // Should have valid state after all processing
    ASSERT(state.current_gain > 0.0f, "gain should remain positive");
    ASSERT(state.current_gain <= AGC_MAX_GAIN, "gain should not exceed max");
    ASSERT(state.peak_level >= 0.0f, "peak should be non-negative");
    PASS();
}

void test_rapid_gain_changes(void) {
    TEST("stress: handles rapid loud/quiet transitions");

    AGCState state;
    agc_init(&state);

    float loud[64], quiet[64];
    for (int i = 0; i < 64; i++) {
        loud[i] = 0.95f * sinf((float)i * 0.2f);
        quiet[i] = 0.01f * sinf((float)i * 0.2f);
    }

    // Rapidly alternate between loud and quiet
    for (int cycle = 0; cycle < 100; cycle++) {
        agc_process(&state, loud, 64, 1);
        agc_process(&state, quiet, 64, 1);
    }

    // State should remain valid
    ASSERT(state.current_gain >= AGC_MIN_GAIN, "gain >= min");
    ASSERT(state.current_gain <= AGC_MAX_GAIN, "gain <= max");
    PASS();
}

void test_dc_offset_handling(void) {
    TEST("edge: handles DC offset in signal");

    AGCState state;
    agc_init(&state);

    // Signal with DC offset: peak value is 0.5 + 0.1 = 0.6
    float samples[256];
    for (int i = 0; i < 256; i++) {
        samples[i] = 0.5f + 0.1f * sinf((float)i * 0.1f);  // DC offset of 0.5
    }

    // Process multiple frames to let envelope follower converge
    for (int frame = 0; frame < 10; frame++) {
        for (int i = 0; i < 256; i++) {
            samples[i] = 0.5f + 0.1f * sinf((float)i * 0.1f);
        }
        agc_process(&state, samples, 256, 1);
    }

    // Peak detection should track toward ~0.6 over multiple frames
    ASSERT(state.peak_level > 0.3f, "should detect peak including DC");
    PASS();
}

void test_impulse_response(void) {
    TEST("edge: single impulse followed by silence");

    AGCState state;
    agc_init(&state);
    state.peak_level = 0.1f;
    state.current_gain = 5.0f;

    // Single loud impulse
    float impulse[64] = {0};
    impulse[32] = 1.0f;
    agc_process(&state, impulse, 64, 1);

    float gain_after_impulse = state.current_gain;

    // Followed by silence - peak level decays, gain increases toward max
    // Need many frames for slow decay to take effect
    float silence[64] = {0};
    for (int i = 0; i < 100; i++) {
        agc_process(&state, silence, 64, 1);
    }

    // With silence, peak decays and gain increases toward max
    // The gain should have increased from post-impulse level
    ASSERT(state.current_gain >= gain_after_impulse,
           "gain should recover or stay same after impulse");
    PASS();
}

// ============================================================================
// Main
// ============================================================================

int main(void) {
    printf("\n" ANSI_YELLOW "═══════════════════════════════════════════════════"
                            "═══════════════════════\n");
    printf(" AGC (Automatic Gain Control) Unit Tests\n");
    printf("═══════════════════════════════════════════════════════════════════"
           "════════\n" ANSI_RESET);

    printf("\n[Initialization]\n");
    test_init_default_values();
    test_init_null_safe();
    test_init_custom_values();
    test_init_custom_clamps_gain();

    printf("\n[Peak Detection]\n");
    test_find_peak_positive_values();
    test_find_peak_negative_values();
    test_find_peak_mixed_values();
    test_find_peak_silence();
    test_find_peak_single_sample();
    test_find_peak_null_buffer();
    test_find_peak_zero_count();

    printf("\n[Peak Tracking (Envelope Follower)]\n");
    test_update_peak_fast_attack();
    test_update_peak_slow_decay();
    test_update_peak_multiple_frames();
    test_update_peak_null_state();

    printf("\n[Gain Calculation]\n");
    test_calculate_gain_normal();
    test_calculate_gain_clamps_to_max();
    test_calculate_gain_clamps_to_min();
    test_calculate_gain_silence_returns_max();
    test_calculate_gain_null_state();

    printf("\n[Gain Smoothing]\n");
    test_smooth_gain_decrease_slow();
    test_smooth_gain_increase_faster();
    test_smooth_gain_convergence();
    test_smooth_gain_null_state();

    printf("\n[Soft Clipping]\n");
    test_soft_clip_small_values_linear();
    test_soft_clip_large_values_compressed();
    test_soft_clip_negative_values();
    test_soft_clip_zero();
    test_soft_clip_extreme_values();

    printf("\n[Full Processing Pipeline]\n");
    test_process_applies_gain();
    test_process_agc_enabled_adjusts_gain();
    test_process_prevents_clipping();
    test_process_silence_handling();
    test_process_null_handling();

    printf("\n[Getters and Setters]\n");
    test_get_gain();
    test_get_gain_null();
    test_set_manual_gain();
    test_set_manual_gain_null();

    printf("\n[Edge Cases and Stress]\n");
    test_stress_many_frames();
    test_rapid_gain_changes();
    test_dc_offset_handling();
    test_impulse_response();

    printf("\n" ANSI_YELLOW "═══════════════════════════════════════════════════"
                            "═══════════════════════\n" ANSI_RESET);
    if (tests_passed == tests_run) {
        printf(ANSI_GREEN " ✓ All %d tests passed\n" ANSI_RESET, tests_run);
    } else {
        printf(ANSI_RED " ✗ %d/%d tests passed (%d failed)\n" ANSI_RESET,
               tests_passed, tests_run, tests_run - tests_passed);
    }
    printf(ANSI_YELLOW "═══════════════════════════════════════════════════════"
                       "════════════════════\n\n" ANSI_RESET);

    return tests_passed == tests_run ? 0 : 1;
}
