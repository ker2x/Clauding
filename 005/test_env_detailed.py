#!/usr/bin/env python3
"""
Detailed test of slip ratios in the full environment.
"""

import numpy as np
from preprocessing import make_carracing_env

env = make_carracing_env(
    stack_size=4,
    terminate_stationary=False,
    stationary_patience=100,
    render_mode=None,
    state_mode='vector'
)

state, _ = env.reset()
action = np.array([0.0, 0.5], dtype=np.float32)  # [steering, accel]

RL_SLIP_IDX = 45

print('Step | State SR | Stored SR | Wheel Omega | Ground V | Match?')
print('-'*70)

for step in range(20):
    state, _, _, _, _ = env.step(action)

    # Get slip from state
    state_slip = state[RL_SLIP_IDX]

    # Get slip from stored forces
    car = env.unwrapped.car
    stored_slip = car.last_tire_forces[2]['slip_ratio'] if car.last_tire_forces else 0.0

    # Get wheel data
    wheel_omega = car.wheels[2].omega
    ground_v = car.vx

    match = '✓' if abs(state_slip - stored_slip) < 0.001 else '✗ MISMATCH!'

    print(f'{step:4d} | {state_slip:+8.4f} | {stored_slip:+9.4f} | {wheel_omega:11.2f} | {ground_v:8.2f} | {match}')

env.close()
