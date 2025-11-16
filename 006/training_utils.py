"""
Shared utilities for training SAC agents on CarRacing-v3.

This module contains common functions used across different training scripts:
- evaluate_agent: Evaluate agent performance
- configure_cpu_threading: Optimize CPU threading for training
- get_device: Determine which device to use (CPU/CUDA/MPS)
- setup_logging: Configure logging infrastructure for training
"""

import os
import csv
import multiprocessing
import torch
import numpy as np
from datetime import datetime


def get_device(device_arg='auto'):
    """
    Determine which device to use for training.

    Args:
        device_arg: 'auto', 'cpu', 'cuda', or 'mps'

    Returns:
        torch.device object
    """
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        return torch.device(device_arg)


def configure_cpu_threading(device):
    """
    Configure PyTorch CPU threading for optimal performance.

    Critical for CPU training performance! Without this, PyTorch defaults to using
    only 1-2 threads, resulting in ~100-120% CPU usage instead of utilizing all cores.

    Args:
        device: torch device (cuda/mps/cpu)
    """
    if device.type == 'cpu':
        # Get number of physical CPU cores (not hyperthreads)
        num_cores = multiprocessing.cpu_count()

        # For CPU training, use all available cores
        # Set intra-op parallelism (within a single operation like matrix multiply)
        torch.set_num_threads(num_cores)

        # Set inter-op parallelism (across independent operations)
        # Using num_cores // 2 to balance with intra-op threads
        torch.set_num_interop_threads(max(1, num_cores // 2))

        print(f"\nCPU Threading Configuration:")
        print(f"  Physical cores detected: {num_cores}")
        print(f"  PyTorch intra-op threads: {torch.get_num_threads()}")
        print(f"  PyTorch inter-op threads: {torch.get_num_interop_threads()}")
    else:
        print(f"\nDevice is {device.type}, skipping CPU threading configuration\n")


def evaluate_agent(agent, env, n_episodes=5, log_handle=None, seed_offset=None,
                   max_steps_per_episode=None, return_details=False):
    """
    Evaluate agent performance over multiple episodes.

    Args:
        agent: SAC agent
        env: CarRacing environment
        n_episodes: Number of episodes to evaluate
        log_handle: Optional file handle for logging
        seed_offset: Optional seed offset for reproducibility
        max_steps_per_episode: Optional max steps per episode (safety timeout)
        return_details: If True, return (mean, std, all_rewards). If False, return just mean.

    Returns:
        If return_details=True: Tuple of (mean_reward, std_reward, all_rewards)
        If return_details=False: mean_reward (float)
    """
    total_rewards = []
    total_steps = []

    for ep in range(n_episodes):
        if seed_offset is not None:
            state, _ = env.reset(seed=seed_offset + ep)
        else:
            state, _ = env.reset()

        episode_reward = 0
        episode_steps = 0
        done = False

        while not done:
            # Use deterministic policy (mean action)
            action = agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_steps += 1
            done = terminated or truncated

            # Safety timeout to prevent infinite loops
            if max_steps_per_episode and episode_steps >= max_steps_per_episode:
                print(f"WARNING: Evaluation episode {ep} exceeded {max_steps_per_episode} steps, terminating")
                break

        total_rewards.append(episode_reward)
        total_steps.append(episode_steps)

        msg = f"  Eval episode {ep + 1}/{n_episodes}: reward = {episode_reward:.2f}, steps = {episode_steps}"
        print(msg)
        if log_handle:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_handle.write(f"[{timestamp}] {msg}\n")

    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    mean_steps = np.mean(total_steps)

    # Add summary with average steps
    summary_msg = f"  Average: reward = {mean_reward:.2f} ± {std_reward:.2f}, steps = {mean_steps:.1f}"
    print(summary_msg)
    if log_handle:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_handle.write(f"[{timestamp}] {summary_msg}\n")

    if return_details:
        return mean_reward, std_reward, total_rewards
    else:
        return mean_reward


def setup_logging(log_dir, args, mode='standard', env=None, agent=None, config=None):
    """
    Setup logging infrastructure for training.

    Args:
        log_dir: Directory to save log files
        args: Training arguments (argparse Namespace)
        mode: 'standard' for full logging or 'selection' for parallel selection training
        env: CarRacing environment (required for standard mode)
        agent: SAC agent (required for standard mode)
        config: Dict with environment configuration values (required for standard mode)

    Returns:
        For mode='standard': Tuple of (training_csv_path, eval_csv_path, log_file_handle)
        For mode='selection': Tuple of (training_csv_path, selection_csv_path)
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if mode == 'selection':
        # Simplified logging for parallel selection training
        training_csv = os.path.join(log_dir, f'training_{timestamp}.csv')
        with open(training_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode', 'generation', 'agent_id', 'reward',
                'best_agent_id', 'best_avg_reward'
            ])

        # Selection log for tournament results
        selection_csv = os.path.join(log_dir, f'selection_{timestamp}.csv')
        with open(selection_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'generation', 'episode', 'winner_id', 'winner_eval_reward',
                'avg_eval_reward', 'min_eval_reward', 'max_eval_reward'
            ])

        print(f"\nLogging initialized:")
        print(f"  Training log: {training_csv}")
        print(f"  Selection log: {selection_csv}")
        return training_csv, selection_csv

    elif mode == 'standard':
        # Full logging for standard training
        if env is None or agent is None or config is None:
            raise ValueError("env, agent, and config are required for standard mode")

        # Create CSV for training metrics
        training_csv = os.path.join(log_dir, 'training_metrics.csv')
        with open(training_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode', 'total_steps', 'episode_steps', 'reward',
                'actor_loss', 'critic_1_loss', 'critic_2_loss', 'alpha_loss',
                'alpha', 'mean_q1', 'mean_q2', 'mean_log_prob',
                'elapsed_time_sec', 'avg_reward_100', 'timestamp'
            ])

        # Create CSV for evaluation metrics
        eval_csv = os.path.join(log_dir, 'evaluation_metrics.csv')
        with open(eval_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode', 'total_steps', 'eval_mean_reward', 'eval_std_reward',
                'eval_rewards', 'is_best', 'elapsed_time_sec', 'timestamp'
            ])

        # Create human-readable log file
        log_file = os.path.join(log_dir, 'training.log')
        log_handle = open(log_file, 'w', buffering=1)  # Line buffering for real-time updates

        # Write header to log file
        log_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_handle.write("=" * 70 + "\n")
        log_handle.write("SAC Training Session Started\n")
        log_handle.write("=" * 70 + "\n")
        log_handle.write(f"Timestamp: {log_timestamp}\n")
        log_handle.write(f"Device: {agent.device}\n")
        log_handle.write(f"State mode: vector (71D)\n")
        log_handle.write(f"State dimension: {env.observation_space.shape[0]}\n")
        log_handle.write(f"Action space: Continuous (3D)\n")
        log_handle.write(f"Episodes: {args.episodes}\n")
        log_handle.write(f"Learning starts: {args.learning_starts} steps\n")
        log_handle.write(f"Auto entropy tuning: {args.auto_entropy_tuning}\n")
        log_handle.write(f"Early termination: enabled (patience={config['stationary_patience']})\n")
        log_handle.write(f"Reward shaping: enabled (penalty {config['short_episode_penalty']} for episodes < {config['min_episode_steps']} steps)\n")
        if hasattr(args, 'resume') and args.resume:
            log_handle.write(f"Resumed from: {args.resume}\n")
        log_handle.write("=" * 70 + "\n\n")

        # Create system info file
        system_info_path = os.path.join(log_dir, 'system_info.txt')
        with open(system_info_path, 'w') as f:
            # Import reward constants from centralized config
            try:
                from config.constants import (
                    PROGRESS_REWARD_SCALE, LAP_COMPLETION_REWARD, ONTRACK_REWARD,
                    FORWARD_SPEED_REWARD_SCALE, STEP_PENALTY, OFFTRACK_PENALTY,
                    OFFTRACK_THRESHOLD, OFFTRACK_TERMINATION_PENALTY
                )

                f.write("Training Configuration\n")
                f.write("=" * 70 + "\n")
                f.write(f"Date: {log_timestamp}\n")
                f.write(f"Device: {agent.device}\n")
                f.write(f"State mode: vector (71D)\n")
                f.write(f"State dimension: {env.observation_space.shape[0]}\n\n")

                f.write("Environment:\n")
                f.write(f"  Name: CarRacing-v3\n")
                f.write(f"  Actions: Continuous [steering, acceleration]\n")
                f.write(f"  Early termination: True (patience={config['stationary_patience']})\n")
                f.write(f"  Reward shaping: True (penalty {config['short_episode_penalty']} for < {config['min_episode_steps']} steps)\n\n")

                f.write("Reward Structure (from config/constants.py):\n")
                f.write(f"  Progress reward: {PROGRESS_REWARD_SCALE} points for full lap (continuous/dense)\n")
                f.write(f"  Lap completion: {LAP_COMPLETION_REWARD} points (bonus for finishing)\n")
                f.write(f"  On-track reward: {ONTRACK_REWARD} per frame (encourages staying on track)\n")
                f.write(f"  Forward speed reward: {FORWARD_SPEED_REWARD_SCALE}×speed per frame (encourages racing, capped at +2.0)\n")
                f.write(f"  Step penalty: {STEP_PENALTY} per frame (time pressure)\n")
                f.write(f"  Off-track penalty: {OFFTRACK_PENALTY} per wheel (>{OFFTRACK_THRESHOLD} wheels)\n")
                f.write(f"  Off-track termination: {OFFTRACK_TERMINATION_PENALTY} (all wheels off)\n\n")
            except ImportError:
                f.write("Training Configuration\n")
                f.write("=" * 70 + "\n")
                f.write(f"Date: {log_timestamp}\n")
                f.write(f"Device: {agent.device}\n")
                f.write(f"State mode: vector (71D)\n")
                f.write(f"State dimension: {env.observation_space.shape[0]}\n\n")

            f.write("Agent Hyperparameters:\n")
            f.write(f"  Actor learning rate: {args.lr_actor}\n")
            f.write(f"  Critic learning rate: {args.lr_critic}\n")
            f.write(f"  Gamma (discount): {args.gamma}\n")
            f.write(f"  Tau (soft update): {args.tau}\n")
            f.write(f"  Batch size: {args.batch_size}\n")
            f.write(f"  Buffer size: {args.buffer_size}\n")
            f.write(f"  Learning starts: {args.learning_starts} steps\n")
            f.write(f"  Auto entropy tuning: {args.auto_entropy_tuning}\n\n")

            f.write("Training Schedule:\n")
            f.write(f"  Total episodes: {args.episodes}\n")
            if hasattr(args, 'eval_frequency'):
                f.write(f"  Eval frequency: {args.eval_frequency} episodes\n")
            if hasattr(args, 'checkpoint_frequency'):
                f.write(f"  Checkpoint frequency: {args.checkpoint_frequency} episodes\n\n")

        print(f"\nLogging initialized:")
        print(f"  Training metrics: {training_csv}")
        print(f"  Evaluation metrics: {eval_csv}")
        print(f"  Training log: {log_file}")
        print(f"  System info: {system_info_path}")

        return training_csv, eval_csv, log_handle

    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'standard' or 'selection'")
