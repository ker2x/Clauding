"""Quick test of trained agent with the loophole fix."""

import sys
from preprocessing import make_carracing_env
from ddqn_agent import DDQNAgent
import torch

checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/final_model.pt"

# Detect state mode from checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu')
if 'state_mode' in checkpoint:
    state_mode = checkpoint['state_mode']
else:
    policy_state = checkpoint['policy_net_state_dict']
    state_mode = 'visual' if 'conv1.weight' in policy_state else 'vector'

print(f"Testing agent from: {checkpoint_path}")
print(f"Detected state mode: {state_mode}")

# Create environment
env = make_carracing_env(
    stack_size=4,
    discretize_actions=True,
    steering_bins=3,
    gas_brake_bins=3,
    terminate_stationary=True,
    stationary_patience=100,
    render_mode=None,
    state_mode=state_mode
)

# Load agent
state_shape = env.observation_space.shape
n_actions = env.action_space.n
agent = DDQNAgent(state_shape=state_shape, n_actions=n_actions, state_mode=state_mode)
agent.load(checkpoint_path, load_optimizer=False)

# Run 1 episode with max 300 steps
state, _ = env.reset()
total_reward = 0
step = 0
done = False
max_steps = 300

print(f"\nRunning episode (max {max_steps} steps)...")

while not done and step < max_steps:
    action = agent.select_action(state, training=False)
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    state = next_state
    step += 1

    if step % 50 == 0:
        print(f"  Step {step}: reward={reward:.2f}, total={total_reward:.2f}")

env.close()

print(f"\n{'='*60}")
print(f"Episode finished!")
print(f"  Total steps: {step}")
print(f"  Total reward: {total_reward:.2f}")
print(f"  Terminated: {terminated}")
print(f"  Truncated: {truncated}")

if 'stationary_termination' in info and info['stationary_termination']:
    print(f"  ⚠️  Stationary termination: Agent was stuck!")
    print(f"  The agent learned the wiggle loophole, but it's now FIXED.")
elif 'off_track' in info and info['off_track']:
    print(f"  ℹ️  Off-track termination: Agent went off track")
elif step >= max_steps:
    print(f"  ⏱️  Reached max steps limit")
else:
    print(f"  ✅ Normal termination")

print(f"{'='*60}")
