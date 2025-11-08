#!/usr/bin/env python3
"""
Test that RL state (especially slip ratios) is stable without oscillations.
"""

import numpy as np
from preprocessing import make_carracing_env

def test_rl_state_stability():
    """Test that slip ratios in RL state don't oscillate."""
    print("=" * 70)
    print("Testing RL State Stability (Slip Ratio Oscillations)")
    print("=" * 70)

    # Create environment in vector mode (what RL uses)
    env = make_carracing_env(
        stack_size=4,
        terminate_stationary=False,
        stationary_patience=100,
        render_mode=None,
        state_mode='vector'
    )

    # Reset environment
    state, _ = env.reset()

    # Apply moderate throttle for 100 steps
    action = np.array([0.0, 0.5], dtype=np.float32)  # [steering, accel]

    # Track slip ratio history for rear left wheel (index 2 in state)
    # State structure (47 dims total):
    # 0-10: basic car state
    # 11-15: track segment
    # 16-35: waypoints
    # 36: speed
    # 37-38: accelerations
    # 39-42: slip angles [FL, FR, RL, RR]
    # 43-46: slip ratios [FL, FR, RL, RR]

    rl_slip_idx = 45  # Rear left slip ratio index
    rr_slip_idx = 46  # Rear right slip ratio index

    slip_history_rl = []
    slip_history_rr = []

    print("\nRunning 100 steps with moderate throttle (50%)...")
    print("Tracking rear wheel slip ratios (driven wheels)\n")

    for step in range(100):
        state, reward, terminated, truncated, _ = env.step(action)

        rl_slip = state[rl_slip_idx]
        rr_slip = state[rr_slip_idx]

        slip_history_rl.append(rl_slip)
        slip_history_rr.append(rr_slip)

        if step % 20 == 0:
            print(f"Step {step:3d}: RL slip = {rl_slip:+.4f}, RR slip = {rr_slip:+.4f}")

    env.close()

    # Analyze oscillations
    print("\n" + "=" * 70)
    print("OSCILLATION ANALYSIS")
    print("=" * 70)

    # Count sign changes (indicator of oscillation)
    sign_changes_rl = sum(1 for i in range(1, len(slip_history_rl))
                          if np.sign(slip_history_rl[i]) != np.sign(slip_history_rl[i-1])
                          and abs(slip_history_rl[i]) > 0.01)

    sign_changes_rr = sum(1 for i in range(1, len(slip_history_rr))
                          if np.sign(slip_history_rr[i]) != np.sign(slip_history_rr[i-1])
                          and abs(slip_history_rr[i]) > 0.01)

    print(f"\nSign changes in RL wheel: {sign_changes_rl}")
    print(f"Sign changes in RR wheel: {sign_changes_rr}")

    # Calculate variance (should be low for stable values)
    var_rl = np.var(slip_history_rl[50:])  # Skip transient
    var_rr = np.var(slip_history_rr[50:])  # Skip transient

    print(f"\nVariance (after settling):")
    print(f"  RL wheel: {var_rl:.6f}")
    print(f"  RR wheel: {var_rr:.6f}")

    # Final values
    final_rl = slip_history_rl[-1]
    final_rr = slip_history_rr[-1]

    print(f"\nFinal slip ratios:")
    print(f"  RL wheel: {final_rl:+.4f}")
    print(f"  RR wheel: {final_rr:+.4f}")

    # Check for stability
    print("\n" + "=" * 70)
    print("STABILITY ASSESSMENT")
    print("=" * 70)

    oscillation_free = (sign_changes_rl < 5 and sign_changes_rr < 5)
    low_variance = (var_rl < 0.01 and var_rr < 0.01)
    positive_slip = (final_rl > 0 and final_rr > 0)  # Should be positive during acceleration

    print(f"\n✓ No oscillations (sign changes < 5): {oscillation_free}")
    print(f"✓ Low variance (< 0.01): {low_variance}")
    print(f"✓ Positive slip during acceleration: {positive_slip}")

    if oscillation_free and low_variance and positive_slip:
        print("\n🎉 PASS: RL state is stable and consistent!")
        return True
    else:
        print("\n❌ FAIL: RL state shows oscillations or instability")
        if not oscillation_free:
            print(f"   Issue: Too many sign changes ({sign_changes_rl}, {sign_changes_rr})")
        if not low_variance:
            print(f"   Issue: High variance ({var_rl:.6f}, {var_rr:.6f})")
        if not positive_slip:
            print(f"   Issue: Slip should be positive during acceleration")
        return False

if __name__ == "__main__":
    import sys
    success = test_rl_state_stability()
    sys.exit(0 if success else 1)
