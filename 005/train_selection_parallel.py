#!/usr/bin/env python3
"""
Parallel Selection Training for CarRacing-v3 SAC Agent

PRIMARY TRAINING METHOD for this project.

Evolutionary training with N independent agents running in parallel:
- Each agent trains independently on a separate CPU core
- Every M episodes: evaluate all agents and select the best performer
- Clone winner's weights to all other agents
- Continue parallel training from winner

Advantages:
- True parallel execution (N× CPU utilization)
- Evolutionary selection pressure improves convergence
- Sample efficient (N× data collection)
- Wall-clock speedup: ~N× compared to single-agent training

Recommended: Use 8 agents with 8+ CPU cores for optimal performance.
"""

import argparse
import csv
import os
import time
from datetime import datetime
import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Queue, Event
import queue

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    print(f"Warning: matplotlib not available ({e})")
    MATPLOTLIB_AVAILABLE = False

from preprocessing import make_carracing_env
from sac_agent import SACAgent, ReplayBuffer
from env.car_racing import (
    PROGRESS_REWARD_SCALE, LAP_COMPLETION_REWARD,
    STEP_PENALTY, OFFTRACK_PENALTY, OFFTRACK_THRESHOLD,
    OFFTRACK_TERMINATION_PENALTY, ONTRACK_REWARD, FORWARD_SPEED_REWARD_SCALE
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train SAC agents with TRUE PARALLEL selection-based evolution'
    )

    # Selection parameters
    parser.add_argument('--num-agents', type=int, default=8,
                        help='Number of parallel agents (default: 8)')
    parser.add_argument('--selection-frequency', type=int, default=50,
                        help='Select best agent every N episodes (default: 50)')
    parser.add_argument('--eval-episodes', type=int, default=5,
                        help='Episodes to evaluate each agent during selection (default: 5)')
    parser.add_argument('--sync-seeds', action='store_true',
                        help='Use synchronized seeds (same track for all agents per episode)')

    # Training parameters
    parser.add_argument('--episodes', type=int, default=2000,
                        help='Total training episodes (default: 2000)')
    parser.add_argument('--learning-starts', type=int, default=5000,
                        help='Steps before training starts (default: 5000)')
    parser.add_argument('--checkpoint-frequency', type=int, default=100,
                        help='Save checkpoint every N episodes (default: 100)')

    # Agent hyperparameters
    parser.add_argument('--lr-actor', type=float, default=3e-4,
                        help='Actor learning rate (default: 3e-4)')
    parser.add_argument('--lr-critic', type=float, default=3e-4,
                        help='Critic learning rate (default: 3e-4)')
    parser.add_argument('--lr-alpha', type=float, default=3e-4,
                        help='Alpha learning rate (default: 3e-4)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='Target network update rate (default: 0.005)')
    parser.add_argument('--auto-entropy-tuning', action='store_true', default=True,
                        help='Automatically tune entropy coefficient')
    parser.add_argument('--buffer-size', type=int, default=1000000,
                        help='Replay buffer size (default: 1000000)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size (default: 256)')

    # Environment parameters
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (must be cpu for multiprocessing, default: cpu)')
    parser.add_argument('--threads-per-agent', type=int, default=None,
                        help='CPU threads per agent (default: auto-calculated as num_cores/num_agents)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_selection_parallel',
                        help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs_selection_parallel',
                        help='Directory to save logs')

    return parser.parse_args()


def worker_process(agent_id, args, result_queue, command_queue, state_dict_queue, episode_offset, threads_per_agent=1, checkpoint_queue=None):
    """
    Worker process that trains a single agent.

    Args:
        agent_id: Unique ID for this agent
        args: Training arguments
        result_queue: Queue to send results back to coordinator
        command_queue: Queue to receive commands from coordinator
        state_dict_queue: Queue to receive new model weights
        episode_offset: Base episode number for seeding
        threads_per_agent: Number of CPU threads this agent should use
        checkpoint_queue: Queue to receive next checkpoint value
    """
    import torch

    # Set thread count to prevent CPU oversubscription
    torch.set_num_threads(threads_per_agent)

    # Environment constants
    MAX_EPISODE_STEPS = 2500
    STATIONARY_PATIENCE = 50
    STATIONARY_MIN_STEPS = 50
    SHORT_EPISODE_PENALTY = -50.0
    MIN_EPISODE_STEPS = 150

    # Create environment
    env = make_carracing_env(
        state_mode='vector',
        max_episode_steps=MAX_EPISODE_STEPS,
        terminate_stationary=True,
        stationary_patience=STATIONARY_PATIENCE,
        stationary_min_steps=STATIONARY_MIN_STEPS,
        reward_shaping=True,
        min_episode_steps=MIN_EPISODE_STEPS,
        short_episode_penalty=SHORT_EPISODE_PENALTY,
        verbose=args.verbose
    )

    state_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    device = torch.device('cpu')  # Force CPU for multiprocessing

    # Create agent
    agent = SACAgent(
        state_shape=state_shape,
        action_dim=action_dim,
        state_mode='vector',
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        lr_alpha=args.lr_alpha,
        gamma=args.gamma,
        tau=args.tau,
        auto_entropy_tuning=args.auto_entropy_tuning,
        device=device
    )

    # Load initial weights if provided
    if args.resume:
        agent.load(args.resume)

    # Create replay buffer
    buffer = ReplayBuffer(
        capacity=args.buffer_size,
        state_shape=state_shape,
        action_dim=action_dim,
        device=device
    )

    total_steps = 0
    episode_count = 0
    next_checkpoint = args.selection_frequency  # Track the next tournament checkpoint

    # Training loop
    while True:
        # Check for commands (non-blocking)
        try:
            command = command_queue.get_nowait()

            if command == 'STOP':
                break
            elif command == 'LOAD_WEIGHTS':
                # Load new weights from coordinator
                new_state_dict = state_dict_queue.get()
                agent.load_state_dict(new_state_dict)
                # Clear replay buffer on selection (fresh start with new weights)
                buffer = ReplayBuffer(
                    capacity=args.buffer_size,
                    state_shape=state_shape,
                    action_dim=action_dim,
                    device=device
                )
            elif command == 'EVALUATE':
                # Evaluate agent and send results back
                eval_reward = evaluate_agent(agent, env, args.eval_episodes, seed_offset=10000 + episode_offset)
                result_queue.put(('EVAL_RESULT', agent_id, eval_reward))
                continue
            elif command == 'GET_WEIGHTS':
                # Send current weights to coordinator
                state_dict = agent.get_state_dict()
                result_queue.put(('WEIGHTS', agent_id, state_dict))
                continue
            # Note: RESUME commands are ONLY processed in the blocking wait loop at checkpoint
            # If processed here, agents might skip the checkpoint wait

        except queue.Empty:
            pass

        # Train one episode
        if args.sync_seeds:
            seed = 1000 + episode_offset + episode_count
        else:
            seed = 1000 + episode_offset + episode_count + agent_id * 10000

        obs, _ = env.reset(seed=seed)
        episode_reward = 0.0
        terminated = False
        truncated = False
        steps = 0

        # Episode metrics
        episode_metrics = {
            'actor_loss': [],
            'critic_1_loss': [],
            'critic_2_loss': [],
            'alpha': [],
            'mean_q1': [],
            'mean_q2': [],
            'mean_log_prob': []
        }

        while not (terminated or truncated):
            action = agent.select_action(obs, evaluate=False)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            buffer.push(obs, action, float(reward), next_obs, float(terminated or truncated))
            episode_reward += reward

            # Train agent
            if (total_steps >= args.learning_starts and
                len(buffer) >= args.batch_size):
                metrics = agent.update(buffer, args.batch_size)
                episode_metrics['actor_loss'].append(metrics['actor_loss'])
                episode_metrics['critic_1_loss'].append(metrics['critic_1_loss'])
                episode_metrics['critic_2_loss'].append(metrics['critic_2_loss'])
                episode_metrics['alpha'].append(metrics['alpha'])
                episode_metrics['mean_q1'].append(metrics.get('mean_q1', 0.0))
                episode_metrics['mean_q2'].append(metrics.get('mean_q2', 0.0))
                episode_metrics['mean_log_prob'].append(metrics.get('mean_log_prob', 0.0))

            obs = next_obs
            steps += 1
            total_steps += 1

        episode_count += 1

        # Calculate average metrics
        avg_metrics = {}
        for key, values in episode_metrics.items():
            avg_metrics[key] = np.mean(values) if values else 0.0

        # Report episode result with metrics
        result_queue.put((
            'EPISODE_DONE', agent_id, episode_reward, episode_count,
            steps, total_steps, len(buffer), avg_metrics
        ))

        # Check if we've reached the tournament checkpoint
        if episode_count >= next_checkpoint:
            # Wait at checkpoint for tournament and RESUME command (blocking)
            while True:
                command = command_queue.get()  # Blocking wait

                if command == 'STOP':
                    env.close()
                    return
                elif command == 'LOAD_WEIGHTS':
                    new_state_dict = state_dict_queue.get()
                    agent.load_state_dict(new_state_dict)
                    buffer = ReplayBuffer(
                        capacity=args.buffer_size,
                        state_shape=state_shape,
                        action_dim=action_dim,
                        device=device
                    )
                elif command == 'EVALUATE':
                    eval_reward = evaluate_agent(agent, env, args.eval_episodes, seed_offset=10000 + episode_offset)
                    result_queue.put(('EVAL_RESULT', agent_id, eval_reward))
                elif command == 'GET_WEIGHTS':
                    state_dict = agent.get_state_dict()
                    result_queue.put(('WEIGHTS', agent_id, state_dict))
                elif command == 'RESUME':
                    # Update checkpoint and break out of wait loop
                    next_checkpoint = checkpoint_queue.get()
                    break

    env.close()


def evaluate_agent(agent, env, num_episodes, seed_offset=10000, max_steps_per_episode=2500):
    """Evaluate an agent over multiple episodes with timeout protection."""
    total_reward = 0.0

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        episode_reward = 0.0
        terminated = False
        truncated = False
        steps = 0

        while not (terminated or truncated):
            action = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1

            # Safety timeout to prevent infinite loops
            if steps >= max_steps_per_episode:
                print(f"WARNING: Evaluation episode {ep} exceeded {max_steps_per_episode} steps, terminating")
                break

        total_reward += episode_reward

    return total_reward / num_episodes


def setup_logging(log_dir, args):
    """Set up logging files."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Training log
    training_csv = os.path.join(log_dir, f'training_{timestamp}.csv')
    with open(training_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'episode', 'generation', 'agent_id', 'reward',
            'best_agent_id', 'best_avg_reward'
        ])

    # Selection log
    selection_csv = os.path.join(log_dir, f'selection_{timestamp}.csv')
    with open(selection_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'generation', 'episode', 'winner_id', 'winner_eval_reward',
            'avg_eval_reward', 'min_eval_reward', 'max_eval_reward'
        ])

    return training_csv, selection_csv


def main():
    args = parse_args()

    # Force CPU for multiprocessing (GPU sharing across processes is complex)
    if args.device != 'cpu':
        print("Warning: Parallel training requires CPU. Setting device to 'cpu'.")
        args.device = 'cpu'

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Calculate threads per agent to prevent CPU oversubscription
    num_cores = mp.cpu_count()
    if args.threads_per_agent is not None:
        threads_per_agent = max(1, args.threads_per_agent)
        thread_mode = "manual"
    else:
        threads_per_agent = max(1, num_cores // args.num_agents)
        thread_mode = "auto"

    print(f"\n{'='*60}")
    print("PARALLEL Selection-Based Training Configuration")
    print(f"{'='*60}")
    print(f"Number of agents: {args.num_agents}")
    print(f"CPU cores available: {num_cores}")
    print(f"Threads per agent: {threads_per_agent} ({thread_mode})")
    print(f"Training mode: TRUE PARALLEL (multiprocessing)")
    print(f"Seed mode: {'SYNCHRONIZED' if args.sync_seeds else 'UNSYNCHRONIZED'}")
    print(f"Selection frequency: every {args.selection_frequency} episodes")
    print(f"Evaluation episodes: {args.eval_episodes}")
    print(f"Total episodes: {args.episodes}")
    print(f"Wall-clock speedup: ~{args.num_agents}×")
    print(f"{'='*60}\n")

    # Initialize logging
    training_csv, selection_csv = setup_logging(args.log_dir, args)

    # Create communication queues
    result_queue = mp.Queue()
    command_queues = [mp.Queue() for _ in range(args.num_agents)]
    state_dict_queues = [mp.Queue() for _ in range(args.num_agents)]
    checkpoint_queues = [mp.Queue() for _ in range(args.num_agents)]

    # Start worker processes
    workers = []
    for agent_id in range(args.num_agents):
        p = Process(
            target=worker_process,
            args=(agent_id, args, result_queue, command_queues[agent_id],
                  state_dict_queues[agent_id], 0, threads_per_agent, checkpoint_queues[agent_id])
        )
        p.start()
        workers.append(p)

    print(f"Started {args.num_agents} parallel worker processes")
    print(f"Each agent using {threads_per_agent} CPU thread(s)")
    print(f"{'='*60}\n")

    # Training state
    generation = 0
    agent_episode_counts = [0] * args.num_agents
    agent_recent_rewards = [[] for _ in range(args.num_agents)]
    agent_total_steps = [0] * args.num_agents
    agent_episode_steps = [0] * args.num_agents
    agent_buffer_sizes = [0] * args.num_agents
    agent_metrics = [{'actor_loss': 0.0, 'critic_1_loss': 0.0, 'critic_2_loss': 0.0,
                      'alpha': 0.0, 'mean_q1': 0.0, 'mean_q2': 0.0, 'mean_log_prob': 0.0}
                     for _ in range(args.num_agents)]
    best_overall_reward = float('-inf')

    # Tournament checkpoint tracking
    next_tournament_checkpoint = args.selection_frequency
    tournament_pending = False

    start_time = time.time()

    # Main coordination loop
    try:
        while True:
            # Collect results from workers
            try:
                msg_type, agent_id, *data = result_queue.get(timeout=1.0)

                if msg_type == 'EPISODE_DONE':
                    episode_reward, episode_count, steps, total_steps, buffer_size, metrics = data
                    agent_episode_counts[agent_id] = episode_count
                    agent_recent_rewards[agent_id].append(episode_reward)
                    agent_total_steps[agent_id] = total_steps
                    agent_episode_steps[agent_id] = steps
                    agent_buffer_sizes[agent_id] = buffer_size
                    agent_metrics[agent_id] = metrics

                    # Log episode
                    with open(training_csv, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            episode_count,
                            generation,
                            agent_id,
                            f"{episode_reward:.2f}",
                            0,  # Will be updated
                            0.0  # Will be updated
                        ])

                    # Print progress every 10 episodes (any agent)
                    total_episodes_sum = sum(agent_episode_counts)
                    min_episodes = min(agent_episode_counts)
                    max_episodes = max(agent_episode_counts)

                    # Only print when min episodes is a multiple of 10
                    if min_episodes % 10 == 0 and min_episodes > 0:
                        elapsed = time.time() - start_time
                        hours = elapsed / 3600

                        # Calculate reward stats
                        current_rewards = []
                        avg_rewards_10 = []
                        for rewards in agent_recent_rewards:
                            if len(rewards) > 0:
                                current_rewards.append(rewards[-1])
                            else:
                                current_rewards.append(0.0)
                            if len(rewards) >= 10:
                                avg_rewards_10.append(np.mean(rewards[-10:]))
                            elif len(rewards) > 0:
                                avg_rewards_10.append(np.mean(rewards))
                            else:
                                avg_rewards_10.append(0.0)

                        best_agent = np.argmax(avg_rewards_10)

                        # Get metrics from best agent
                        best_metrics = agent_metrics[best_agent]

                        print(f"\nEpisode {min_episodes}/{args.episodes} (per agent) | Gen {generation} | Time: {hours:.2f}h")
                        print(f"  Episode range: [{min_episodes}, {max_episodes}] across {args.num_agents} agents")
                        print(f"  Agents: [{', '.join([f'{r:6.1f}' for r in current_rewards])}]")
                        print(f"  Best: Agent {best_agent} ({avg_rewards_10[best_agent]:7.2f}) | "
                              f"Avg: {np.mean(avg_rewards_10):7.2f} | "
                              f"Range: [{min(current_rewards):.1f}, {max(current_rewards):.1f}]")
                        print(f"  Reward avg(10): {np.mean(avg_rewards_10):7.2f}")
                        print(f"  Steps: {agent_episode_steps[best_agent]:4d} | "
                              f"Total: {agent_total_steps[best_agent]:7d} | "
                              f"Buffer: {agent_buffer_sizes[best_agent]:6d}")

                        if agent_total_steps[best_agent] >= args.learning_starts:
                            print(f"  Loss: Actor={best_metrics['actor_loss']:.4f} "
                                  f"Critic1={best_metrics['critic_1_loss']:.4f} "
                                  f"Critic2={best_metrics['critic_2_loss']:.4f}")
                            print(f"  Alpha: {best_metrics['alpha']:.4f} | "
                                  f"Q1: {best_metrics['mean_q1']:.2f} | "
                                  f"Q2: {best_metrics['mean_q2']:.2f} | "
                                  f"LogProb: {best_metrics['mean_log_prob']:.4f}")
                        else:
                            warmup_remaining = args.learning_starts - agent_total_steps[best_agent]
                            print(f"  Warmup: {warmup_remaining} steps remaining before training starts")

                    # Check if selection should occur
                    min_episodes = min(agent_episode_counts)
                    max_episodes = max(agent_episode_counts)

                    # Check if slowest agent has reached the tournament checkpoint
                    if not tournament_pending and min_episodes >= next_tournament_checkpoint:
                        tournament_pending = True
                        print(f"\n{'='*60}")
                        print(f"Tournament checkpoint reached: {next_tournament_checkpoint} episodes")
                        print(f"Waiting for all agents to reach checkpoint...")
                        print(f"Agent progress: {agent_episode_counts}")
                        print(f"{'='*60}\n")

                    # Trigger selection when all agents have reached the checkpoint
                    if tournament_pending and all(count >= next_tournament_checkpoint for count in agent_episode_counts):
                        # All agents have reached the tournament checkpoint
                        generation += 1
                        tournament_pending = False

                        print(f"\n{'='*60}")
                        print(f"GENERATION {generation}: Selection Tournament")
                        print(f"{'='*60}")

                        # Request evaluation from all workers
                        print(f"  Requesting evaluation from all {args.num_agents} agents...")
                        for aid in range(args.num_agents):
                            command_queues[aid].put('EVALUATE')
                        print(f"  Evaluation commands sent, waiting for results...")

                        # Collect evaluation results with timeout
                        eval_rewards = {}
                        eval_timeout = 120.0  # 2 minutes total timeout for all evaluations
                        eval_start_time = time.time()

                        while len(eval_rewards) < args.num_agents:
                            remaining_time = eval_timeout - (time.time() - eval_start_time)
                            if remaining_time <= 0:
                                print(f"  ERROR: Evaluation timeout! Only received {len(eval_rewards)}/{args.num_agents} results")
                                print(f"  Missing agents: {[i for i in range(args.num_agents) if i not in eval_rewards]}")
                                # Use zero reward for missing agents
                                for aid in range(args.num_agents):
                                    if aid not in eval_rewards:
                                        eval_rewards[aid] = float('-inf')
                                break

                            try:
                                msg_type, aid, *data = result_queue.get(timeout=min(10.0, remaining_time))
                                if msg_type == 'EVAL_RESULT':
                                    eval_reward = data[0]
                                    eval_rewards[aid] = eval_reward
                                    print(f"  Agent {aid}: {eval_reward:.2f} avg reward ({len(eval_rewards)}/{args.num_agents})")
                                # Ignore other messages during evaluation
                            except queue.Empty:
                                elapsed = time.time() - eval_start_time
                                print(f"  Still waiting... {len(eval_rewards)}/{args.num_agents} received (elapsed: {elapsed:.1f}s)")
                                continue

                        # Select winner
                        winner_id = max(eval_rewards.keys(), key=lambda k: eval_rewards[k])
                        winner_reward = eval_rewards[winner_id]

                        print(f"\n  🏆 WINNER: Agent {winner_id} ({winner_reward:.2f})")
                        print(f"  Cloning Agent {winner_id} to all positions...")

                        # Get winner's weights
                        command_queues[winner_id].put('GET_WEIGHTS')
                        while True:
                            msg_type, aid, *data = result_queue.get(timeout=10.0)
                            if msg_type == 'WEIGHTS' and aid == winner_id:
                                winner_state_dict = data[0]
                                break
                            # Ignore other messages while waiting for weights

                        # Broadcast winner's weights to all agents
                        for aid in range(args.num_agents):
                            if aid != winner_id:
                                state_dict_queues[aid].put(winner_state_dict)
                                command_queues[aid].put('LOAD_WEIGHTS')

                        print(f"  All agents now copies of Agent {winner_id}")
                        print(f"{'='*60}\n")

                        # Update next tournament checkpoint
                        next_tournament_checkpoint += args.selection_frequency

                        # Resume all agents with new checkpoint
                        for aid in range(args.num_agents):
                            checkpoint_queues[aid].put(next_tournament_checkpoint)
                            command_queues[aid].put('RESUME')

                        # Log selection
                        with open(selection_csv, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                generation,
                                min_episodes,
                                winner_id,
                                f"{winner_reward:.2f}",
                                f"{np.mean(list(eval_rewards.values())):.2f}",
                                f"{min(eval_rewards.values()):.2f}",
                                f"{max(eval_rewards.values()):.2f}"
                            ])

                        # Update best overall
                        if winner_reward > best_overall_reward:
                            best_overall_reward = winner_reward
                            best_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
                            # Save winner's state dict to file
                            import torch
                            torch.save(winner_state_dict, best_path)
                            print(f"✓ New best model saved: {best_overall_reward:.2f}")

            except queue.Empty:
                pass

            # Check if done
            if min(agent_episode_counts) >= args.episodes:
                print(f"\n{'='*60}")
                print("Training Complete!")
                print(f"{'='*60}")
                print(f"Total generations: {generation}")
                print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
                print(f"Best overall reward: {best_overall_reward:.2f}")
                print(f"{'='*60}\n")
                break

    finally:
        # Stop all workers
        for q in command_queues:
            q.put('STOP')

        # Wait for workers to finish
        for p in workers:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()


if __name__ == '__main__':
    # Required for multiprocessing on Windows/MacOS
    mp.set_start_method('spawn', force=True)
    main()
