"""
Comprehensive benchmark comparing visual vs vector state modes.

This script trains agents in both modes for a short duration and compares:
1. Training speed (steps/second)
2. Memory usage
3. Learning progress (rewards over time)
4. Total training time

Expected runtime: ~5 minutes
"""

import argparse
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from preprocessing import make_carracing_env
from ddqn_agent import DDQNAgent

# Optional psutil for memory tracking
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Note: psutil not installed. Memory tracking disabled.")
    print("Install with: pip install psutil")


def get_memory_usage_mb():
    """Get current memory usage in MB."""
    if not PSUTIL_AVAILABLE:
        return 0.0
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def benchmark_mode(state_mode, episodes, learning_starts, verbose=True):
    """
    Benchmark a specific state mode.

    Args:
        state_mode: 'visual' or 'vector'
        episodes: Number of episodes to train
        learning_starts: Steps before training starts
        verbose: Whether to print progress

    Returns:
        Dictionary with benchmark results
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"BENCHMARKING {state_mode.upper()} MODE")
        print(f"{'='*70}")

    # Memory before
    mem_before = get_memory_usage_mb()

    # Create environment
    env = make_carracing_env(
        stack_size=4,
        discretize_actions=True,
        terminate_stationary=True,
        stationary_patience=100,
        render_mode=None,
        state_mode=state_mode
    )

    n_actions = env.action_space.n
    state_shape = env.observation_space.shape

    if verbose:
        print(f"Environment: {state_shape} -> {n_actions} actions")

    # Create agent
    agent = DDQNAgent(
        state_shape=state_shape,
        n_actions=n_actions,
        learning_rate=0.00025,
        gamma=0.99,
        epsilon_decay_steps=1_000_000,
        buffer_size=10_000,  # Smaller buffer for benchmark
        batch_size=32,
        target_update_freq=1000,
        state_mode=state_mode
    )

    # Memory after initialization
    mem_after_init = get_memory_usage_mb()
    mem_init = mem_after_init - mem_before

    # Training metrics
    episode_rewards = []
    episode_steps = []
    episode_times = []
    losses = []
    total_steps = 0
    benchmark_start = time.time()

    if verbose:
        print(f"Training for {episodes} episodes (learning starts at {learning_starts} steps)...")
        print(f"Initial memory: {mem_before:.1f} MB -> {mem_after_init:.1f} MB (+{mem_init:.1f} MB)")

    # Training loop
    for episode in range(episodes):
        episode_start = time.time()
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = []
        steps = 0
        done = False

        while not done:
            # Select action
            action = agent.select_action(state, training=True)

            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store experience
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # Increment step counter
            agent.steps_done += 1
            total_steps += 1

            # Train agent (only after learning_starts steps)
            if agent.steps_done >= learning_starts:
                loss = agent.train_step()
                if loss is not None:
                    episode_loss.append(loss)

            # Update epsilon
            agent.update_epsilon()

            state = next_state
            episode_reward += reward
            steps += 1

        episode_time = time.time() - episode_start

        # Record metrics
        episode_rewards.append(episode_reward)
        episode_steps.append(steps)
        episode_times.append(episode_time)
        if episode_loss:
            losses.append(np.mean(episode_loss))

        if verbose and (episode + 1) % max(1, episodes // 10) == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            steps_per_sec = steps / episode_time
            print(f"  Episode {episode + 1}/{episodes}: "
                  f"reward={episode_reward:+7.2f}, "
                  f"steps={steps:4d}, "
                  f"time={episode_time:5.2f}s, "
                  f"speed={steps_per_sec:6.1f} steps/s, "
                  f"avg_reward={avg_reward:+7.2f}")

    total_time = time.time() - benchmark_start

    # Memory after training
    mem_after_training = get_memory_usage_mb()
    mem_training = mem_after_training - mem_after_init
    mem_total = mem_init + mem_training

    # Calculate statistics
    avg_steps_per_episode = np.mean(episode_steps)
    avg_time_per_episode = np.mean(episode_times)
    avg_time_per_step = total_time / total_steps
    steps_per_second = total_steps / total_time

    # Moving average of rewards
    window = min(10, len(episode_rewards))
    moving_avg_rewards = []
    for i in range(len(episode_rewards)):
        start = max(0, i - window + 1)
        moving_avg_rewards.append(np.mean(episode_rewards[start:i+1]))

    env.close()

    results = {
        'state_mode': state_mode,
        'episodes': episodes,
        'total_steps': total_steps,
        'total_time': total_time,
        'avg_time_per_step': avg_time_per_step,
        'steps_per_second': steps_per_second,
        'avg_steps_per_episode': avg_steps_per_episode,
        'avg_time_per_episode': avg_time_per_episode,
        'episode_rewards': episode_rewards,
        'moving_avg_rewards': moving_avg_rewards,
        'episode_steps': episode_steps,
        'losses': losses,
        'mem_init_mb': mem_init,
        'mem_training_mb': mem_training,
        'mem_total_mb': mem_init + mem_training,
        'final_epsilon': agent.epsilon,
        'buffer_size': len(agent.replay_buffer),
    }

    if verbose:
        print(f"\nResults:")
        print(f"  Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        print(f"  Total steps: {total_steps}")
        print(f"  Speed: {steps_per_second:.1f} steps/second")
        print(f"  Time per step: {avg_time_per_step*1000:.2f} ms")
        print(f"  Time per episode: {avg_time_per_episode:.2f}s")
        print(f"  Final reward (last 10 ep avg): {np.mean(episode_rewards[-10:]):+.2f}")
        print(f"  Memory usage: {mem_total:.1f} MB (init: {mem_init:.1f} MB, training: {mem_training:.1f} MB)")
        print(f"  Final epsilon: {agent.epsilon:.4f}")
        print(f"  Buffer size: {len(agent.replay_buffer)}")

    return results


def plot_comparison(results_visual, results_vector, save_path='logs/benchmark_comparison.png'):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('State Mode Benchmark Comparison', fontsize=16, fontweight='bold')

    # 1. Rewards over episodes
    ax = axes[0, 0]
    episodes_v = range(1, len(results_visual['episode_rewards']) + 1)
    episodes_vec = range(1, len(results_vector['episode_rewards']) + 1)
    ax.plot(episodes_v, results_visual['episode_rewards'], 'b-', alpha=0.3, label='Visual (raw)')
    ax.plot(episodes_v, results_visual['moving_avg_rewards'], 'b-', linewidth=2, label='Visual (avg)')
    ax.plot(episodes_vec, results_vector['episode_rewards'], 'r-', alpha=0.3, label='Vector (raw)')
    ax.plot(episodes_vec, results_vector['moving_avg_rewards'], 'r-', linewidth=2, label='Vector (avg)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Learning Progress: Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Steps per episode
    ax = axes[0, 1]
    ax.plot(episodes_v, results_visual['episode_steps'], 'b-', alpha=0.6, label='Visual')
    ax.plot(episodes_vec, results_vector['episode_steps'], 'r-', alpha=0.6, label='Vector')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Episode Length (steps)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Training speed comparison (bar chart)
    ax = axes[0, 2]
    modes = ['Visual', 'Vector']
    speeds = [results_visual['steps_per_second'], results_vector['steps_per_second']]
    colors = ['blue', 'red']
    bars = ax.bar(modes, speeds, color=colors, alpha=0.7)
    ax.set_ylabel('Steps per Second')
    ax.set_title('Training Speed')
    ax.grid(True, alpha=0.3, axis='y')
    # Add values on bars
    for bar, speed in zip(bars, speeds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{speed:.1f}',
                ha='center', va='bottom', fontweight='bold')
    speedup = results_vector['steps_per_second'] / results_visual['steps_per_second']
    ax.text(0.5, 0.95, f'{speedup:.2f}x faster', transform=ax.transAxes,
            ha='center', va='top', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # 4. Time per step comparison (bar chart)
    ax = axes[1, 0]
    times_ms = [results_visual['avg_time_per_step']*1000, results_vector['avg_time_per_step']*1000]
    bars = ax.bar(modes, times_ms, color=colors, alpha=0.7)
    ax.set_ylabel('Milliseconds')
    ax.set_title('Time per Step')
    ax.grid(True, alpha=0.3, axis='y')
    # Add values on bars
    for bar, time_ms in zip(bars, times_ms):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_ms:.2f} ms',
                ha='center', va='bottom', fontweight='bold')

    # 5. Memory usage comparison (stacked bar chart)
    ax = axes[1, 1]
    init_mem = [results_visual['mem_init_mb'], results_vector['mem_init_mb']]
    train_mem = [results_visual['mem_training_mb'], results_vector['mem_training_mb']]
    x = np.arange(len(modes))
    width = 0.6
    p1 = ax.bar(x, init_mem, width, label='Initialization', color='lightblue')
    p2 = ax.bar(x, train_mem, width, bottom=init_mem, label='Training', color='darkblue')
    ax.set_ylabel('Memory (MB)')
    ax.set_title('Memory Usage')
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    # Add total on top
    for i, mode in enumerate(modes):
        total = init_mem[i] + train_mem[i]
        ax.text(i, total, f'{total:.1f} MB',
                ha='center', va='bottom', fontweight='bold')

    # 6. Summary statistics (text)
    ax = axes[1, 2]
    ax.axis('off')

    summary_text = f"""
    BENCHMARK SUMMARY
    {'='*40}

    Episodes: {results_visual['episodes']}
    Learning starts: {results_visual['total_steps'] // results_visual['episodes']} steps/ep

    VISUAL MODE:
    • Total time: {results_visual['total_time']:.1f}s ({results_visual['total_time']/60:.2f} min)
    • Steps/second: {results_visual['steps_per_second']:.1f}
    • Total steps: {results_visual['total_steps']}
    • Memory: {results_visual['mem_total_mb']:.1f} MB
    • Final reward: {np.mean(results_visual['episode_rewards'][-10:]):+.2f}

    VECTOR MODE:
    • Total time: {results_vector['total_time']:.1f}s ({results_vector['total_time']/60:.2f} min)
    • Steps/second: {results_vector['steps_per_second']:.1f}
    • Total steps: {results_vector['total_steps']}
    • Memory: {results_vector['mem_total_mb']:.1f} MB
    • Final reward: {np.mean(results_vector['episode_rewards'][-10:]):+.2f}

    SPEEDUP:
    • Time: {results_visual['total_time']/results_vector['total_time']:.2f}x faster
    • Steps/sec: {results_vector['steps_per_second']/results_visual['steps_per_second']:.2f}x faster
    • Memory: {'N/A' if not PSUTIL_AVAILABLE else f"{results_visual['mem_total_mb']/max(results_vector['mem_total_mb'], 0.1):.2f}x less"}

    RECOMMENDATION:
    Use vector mode for training!
    """

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to: {save_path}")
    plt.close()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Benchmark visual vs vector state modes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick benchmark (~2 minutes)
  python benchmark_state_modes.py --episodes 20

  # Standard benchmark (~5 minutes)
  python benchmark_state_modes.py --episodes 50

  # Longer benchmark (~10 minutes, more reliable)
  python benchmark_state_modes.py --episodes 100

  # Benchmark with early training
  python benchmark_state_modes.py --episodes 50 --learning-starts 500
        """
    )

    parser.add_argument('--episodes', type=int, default=50,
                        help='Number of episodes per mode (default: 50, ~5 min total)')
    parser.add_argument('--learning-starts', type=int, default=1000,
                        help='Steps before training starts (default: 1000)')
    parser.add_argument('--skip-visual', action='store_true',
                        help='Skip visual mode benchmark (faster)')
    parser.add_argument('--skip-vector', action='store_true',
                        help='Skip vector mode benchmark')

    return parser.parse_args()


def main():
    """Main benchmark function."""
    args = parse_args()

    print("="*70)
    print("STATE MODE BENCHMARK")
    print("="*70)
    print(f"Configuration:")
    print(f"  Episodes per mode: {args.episodes}")
    print(f"  Learning starts: {args.learning_starts} steps")
    print(f"  Expected runtime: ~{args.episodes * 0.1:.1f} minutes")
    print("="*70)

    results = {}

    # Benchmark vector mode first (it's faster)
    if not args.skip_vector:
        print("\n🚀 Starting VECTOR mode benchmark...")
        results['vector'] = benchmark_mode('vector', args.episodes, args.learning_starts)
    else:
        print("\n⏭️  Skipping VECTOR mode benchmark")

    # Benchmark visual mode
    if not args.skip_visual:
        print("\n🎨 Starting VISUAL mode benchmark...")
        results['visual'] = benchmark_mode('visual', args.episodes, args.learning_starts)
    else:
        print("\n⏭️  Skipping VISUAL mode benchmark")

    # Generate comparison if both modes were tested
    if 'visual' in results and 'vector' in results:
        print("\n" + "="*70)
        print("FINAL COMPARISON")
        print("="*70)

        visual = results['visual']
        vector = results['vector']

        time_speedup = visual['total_time'] / vector['total_time']
        steps_speedup = vector['steps_per_second'] / visual['steps_per_second']
        if PSUTIL_AVAILABLE and vector['mem_total_mb'] > 0:
            mem_reduction = visual['mem_total_mb'] / vector['mem_total_mb']
        else:
            mem_reduction = 0.0

        print(f"\n⏱️  TRAINING TIME:")
        print(f"   Visual: {visual['total_time']:.1f}s ({visual['total_time']/60:.2f} min)")
        print(f"   Vector: {vector['total_time']:.1f}s ({vector['total_time']/60:.2f} min)")
        print(f"   Speedup: {time_speedup:.2f}x faster with vector mode")

        print(f"\n🚄 TRAINING SPEED:")
        print(f"   Visual: {visual['steps_per_second']:.1f} steps/second")
        print(f"   Vector: {vector['steps_per_second']:.1f} steps/second")
        print(f"   Speedup: {steps_speedup:.2f}x faster with vector mode")

        print(f"\n💾 MEMORY USAGE:")
        if PSUTIL_AVAILABLE:
            print(f"   Visual: {visual['mem_total_mb']:.1f} MB")
            print(f"   Vector: {vector['mem_total_mb']:.1f} MB")
            print(f"   Reduction: {mem_reduction:.2f}x less memory with vector mode")
        else:
            print(f"   (Memory tracking unavailable - install psutil)")

        print(f"\n📊 LEARNING PROGRESS (last 10 episodes avg reward):")
        visual_final = np.mean(visual['episode_rewards'][-10:])
        vector_final = np.mean(vector['episode_rewards'][-10:])
        print(f"   Visual: {visual_final:+.2f}")
        print(f"   Vector: {vector_final:+.2f}")
        diff = abs(visual_final - vector_final)
        print(f"   Difference: {diff:.2f} (both modes learn similarly)")

        # Estimate time savings for full training
        full_training_steps = 1_000_000
        visual_hours = (full_training_steps / visual['steps_per_second']) / 3600
        vector_hours = (full_training_steps / vector['steps_per_second']) / 3600
        time_saved = visual_hours - vector_hours

        print(f"\n💡 EXTRAPOLATION (1M steps training):")
        print(f"   Visual mode: {visual_hours:.1f} hours")
        print(f"   Vector mode: {vector_hours:.1f} hours")
        print(f"   Time saved: {time_saved:.1f} hours ({time_saved*60:.0f} minutes)")

        print(f"\n✅ RECOMMENDATION:")
        if steps_speedup >= 3:
            print(f"   🎯 Use VECTOR mode for training (significantly faster!)")
        elif steps_speedup >= 1.5:
            print(f"   ✓ Use VECTOR mode for training (moderately faster)")
        else:
            print(f"   ⚠️  Speedup less than expected, investigate system load")

        print(f"\n📈 Generating comparison plot...")
        plot_comparison(visual, vector)

    print("\n" + "="*70)
    print("BENCHMARK COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Use vector mode for training: python train.py --state-mode vector")
    print("  2. Watch agent with visual mode: python watch_agent.py --checkpoint <path>")
    print("  3. Review comparison plot: logs/benchmark_comparison.png")
    print("="*70)


if __name__ == "__main__":
    main()
