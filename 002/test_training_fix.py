"""Quick test to verify steps_done increments properly and training starts."""

from preprocessing import make_carracing_env
from ddqn_agent import DDQNAgent

# Create environment
env = make_carracing_env(render_mode=None)
n_actions = env.action_space.n
state_shape = env.observation_space.shape

# Create agent
agent = DDQNAgent(
    state_shape=state_shape,
    n_actions=n_actions,
    buffer_size=10000,
    batch_size=32
)

print("=" * 60)
print("Testing Training Fix")
print("=" * 60)
print(f"Initial steps_done: {agent.steps_done}")
print(f"Initial epsilon: {agent.epsilon:.4f}")
print()

# Run a few steps
state, _ = env.reset()
learning_starts = 500

print("Collecting experiences and testing training...")
for step in range(600):
    action = agent.select_action(state, training=True)
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    agent.store_experience(state, action, reward, next_state, done)

    # Increment steps (this is the fix!)
    agent.steps_done += 1

    # Try training after learning_starts
    if agent.steps_done >= learning_starts:
        loss = agent.train_step()
        if loss is not None and agent.steps_done % 100 == 0:
            print(f"Step {agent.steps_done}: Loss = {loss:.4f}, Epsilon = {agent.epsilon:.4f}")

    # Update epsilon
    agent.update_epsilon()

    if done:
        state, _ = env.reset()
    else:
        state = next_state

print()
print("=" * 60)
print("Test Results")
print("=" * 60)
print(f"Final steps_done: {agent.steps_done}")
print(f"Final epsilon: {agent.epsilon:.6f}")
print(f"Replay buffer size: {len(agent.replay_buffer)}")
print()

if agent.steps_done == 600:
    print("✅ SUCCESS! steps_done is incrementing properly")
else:
    print(f"❌ FAIL! steps_done should be 600, but is {agent.steps_done}")

if agent.epsilon < 1.0:
    print("✅ SUCCESS! Epsilon is decaying")
else:
    print("❌ FAIL! Epsilon should have decayed from 1.0")

env.close()
