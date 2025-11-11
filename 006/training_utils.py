"""
Shared utilities for training SAC agents on CarRacing-v3.

This module contains common functions used across different training scripts:
- evaluate_agent: Evaluate agent performance
- configure_cpu_threading: Optimize CPU threading for training
- get_device: Determine which device to use (CPU/CUDA/MPS)
"""

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
