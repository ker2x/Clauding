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
from training_utils import evaluate_agent, setup_logging
from constants import *


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train SAC agents with TRUE PARALLEL selection-based evolution'
    )

    # Selection parameters
    parser.add_argument('--num-agents', type=int, default=DEFAULT_NUM_AGENTS,
                        help=f'Number of parallel agents (default: {DEFAULT_NUM_AGENTS})')
    parser.add_argument('--selection-frequency', type=int, default=DEFAULT_SELECTION_FREQUENCY,
                        help=f'Select best agent every N episodes (default: {DEFAULT_SELECTION_FREQUENCY})')
    parser.add_argument('--eval-episodes', type=int, default=DEFAULT_EVAL_EPISODES,
                        help=f'Episodes to evaluate each agent during selection (default: {DEFAULT_EVAL_EPISODES})')
    parser.add_argument('--elite-count', type=int, default=DEFAULT_ELITE_COUNT,
                        help=f'Number of top agents to preserve (1=winner-takes-all, 2+=elite preservation, default: {DEFAULT_ELITE_COUNT})')
    parser.add_argument('--sync-seeds', action='store_true',
                        help='Use synchronized seeds (same track for all agents per episode)')

    # Training parameters
    parser.add_argument('--episodes', type=int, default=DEFAULT_EPISODES,
                        help=f'Total training episodes (default: {DEFAULT_EPISODES})')
    parser.add_argument('--learning-starts', type=int, default=DEFAULT_LEARNING_STARTS,
                        help=f'Steps before training starts (default: {DEFAULT_LEARNING_STARTS})')
    parser.add_argument('--checkpoint-frequency', type=int, default=DEFAULT_CHECKPOINT_FREQUENCY,
                        help=f'Save checkpoint every N episodes (default: {DEFAULT_CHECKPOINT_FREQUENCY})')

    # Agent hyperparameters
    parser.add_argument('--lr-actor', type=float, default=DEFAULT_LR_ACTOR,
                        help=f'Actor learning rate (default: {DEFAULT_LR_ACTOR})')
    parser.add_argument('--lr-critic', type=float, default=DEFAULT_LR_CRITIC,
                        help=f'Critic learning rate (default: {DEFAULT_LR_CRITIC})')
    parser.add_argument('--lr-alpha', type=float, default=DEFAULT_LR_ALPHA,
                        help=f'Alpha learning rate (default: {DEFAULT_LR_ALPHA})')
    parser.add_argument('--gamma', type=float, default=DEFAULT_GAMMA,
                        help=f'Discount factor (default: {DEFAULT_GAMMA})')
    parser.add_argument('--tau', type=float, default=DEFAULT_TAU,
                        help=f'Target network update rate (default: {DEFAULT_TAU})')
    parser.add_argument('--auto-entropy-tuning', action='store_true', default=True,
                        help='Automatically tune entropy coefficient')
    parser.add_argument('--buffer-size', type=int, default=DEFAULT_BUFFER_SIZE,
                        help=f'Replay buffer size (default: {DEFAULT_BUFFER_SIZE})')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'Batch size (default: {DEFAULT_BATCH_SIZE})')

    # Environment parameters
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (must be cpu for multiprocessing, default: cpu)')
    parser.add_argument('--threads-per-agent', type=int, default=None,
                        help='CPU threads per agent (default: auto-calculated as num_cores/num_agents)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--checkpoint-dir', type=str, default=DEFAULT_SELECTION_CHECKPOINT_DIR,
                        help=f'Directory to save checkpoints (default: {DEFAULT_SELECTION_CHECKPOINT_DIR})')
    parser.add_argument('--log-dir', type=str, default=DEFAULT_SELECTION_LOG_DIR,
                        help=f'Directory to save logs (default: {DEFAULT_SELECTION_LOG_DIR})')

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

    # Environment constants (using constants from constants.py)
    MAX_EPISODE_STEPS = DEFAULT_MAX_EPISODE_STEPS
    STATIONARY_PATIENCE = DEFAULT_STATIONARY_PATIENCE
    STATIONARY_MIN_STEPS = DEFAULT_STATIONARY_MIN_STEPS
    SHORT_EPISODE_PENALTY = DEFAULT_SHORT_EPISODE_PENALTY
    MIN_EPISODE_STEPS = DEFAULT_MIN_EPISODE_STEPS

    # Create environment
    env = make_carracing_env(
        max_episode_steps=MAX_EPISODE_STEPS,
        terminate_stationary=True,
        stationary_patience=STATIONARY_PATIENCE,
        stationary_min_steps=STATIONARY_MIN_STEPS,
        reward_shaping=True,
        min_episode_steps=MIN_EPISODE_STEPS,
        short_episode_penalty=SHORT_EPISODE_PENALTY,
        verbose=args.verbose
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = torch.device('cpu')  # Force CPU for multiprocessing

    # Create agent
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
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
        state_shape=state_dim,
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
                    state_shape=state_dim,
                    action_dim=action_dim,
                    device=device
                )
            elif command == 'EVALUATE':
                # Evaluate agent and send results back
                eval_reward = evaluate_agent(agent, env, n_episodes=args.eval_episodes, seed_offset=10000 + episode_offset, max_steps_per_episode=DEFAULT_MAX_STEPS_PER_EPISODE)
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
                        state_shape=state_dim,
                        action_dim=action_dim,
                        device=device
                    )
                elif command == 'EVALUATE':
                    eval_reward = evaluate_agent(agent, env, n_episodes=args.eval_episodes, seed_offset=10000 + episode_offset, max_steps_per_episode=DEFAULT_MAX_STEPS_PER_EPISODE)
                    result_queue.put(('EVAL_RESULT', agent_id, eval_reward))
                elif command == 'GET_WEIGHTS':
                    state_dict = agent.get_state_dict()
                    result_queue.put(('WEIGHTS', agent_id, state_dict))
                elif command == 'RESUME':
                    # Update checkpoint and break out of wait loop
                    next_checkpoint = checkpoint_queue.get()
                    break

    env.close()


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

    elite_count = min(args.elite_count, args.num_agents - 1)
    if elite_count > 1:
        print(f"Selection strategy: Elite preservation (top {elite_count} agents)")
    else:
        print(f"Selection strategy: Winner-takes-all")

    print(f"Total episodes: {args.episodes}")
    print(f"Wall-clock speedup: ~{args.num_agents}×")
    print(f"{'='*60}\n")

    # Initialize logging
    training_csv, selection_csv = setup_logging(args.log_dir, args, mode='selection')

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

                        print(f"\n{'='*60}", flush=True)
                        print(f"GENERATION {generation}: Selection Tournament", flush=True)
                        print(f"{'='*60}", flush=True)

                        # Request evaluation from all workers
                        print(f"  Requesting evaluation from all {args.num_agents} agents...", flush=True)

                        try:
                            for aid in range(args.num_agents):
                                command_queues[aid].put('EVALUATE')
                                print(f"    Sent EVALUATE to agent {aid}", flush=True)
                        except Exception as e:
                            print(f"  ERROR sending EVALUATE commands: {e}", flush=True)
                            raise

                        print(f"  Evaluation commands sent, waiting for results...", flush=True)

                        # Collect evaluation results with timeout
                        eval_rewards = {}
                        eval_timeout = 600.0  # 10 minutes total timeout for all evaluations
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
                                    print(f"  Agent {aid}: {eval_reward:.2f} avg reward ({len(eval_rewards)}/{args.num_agents})", flush=True)
                                else:
                                    print(f"  Unexpected message during evaluation: {msg_type} from agent {aid}", flush=True)
                            except queue.Empty:
                                elapsed = time.time() - eval_start_time
                                print(f"  Still waiting... {len(eval_rewards)}/{args.num_agents} received (elapsed: {elapsed:.1f}s)", flush=True)
                                continue
                            except Exception as e:
                                print(f"  ERROR during evaluation result collection: {e}", flush=True)
                                import traceback
                                traceback.print_exc()
                                raise

                        # Sort agents by performance
                        sorted_agents = sorted(eval_rewards.items(), key=lambda x: x[1], reverse=True)

                        # Determine elite and non-elite agents
                        elite_count = min(args.elite_count, args.num_agents - 1)  # At least 1 agent must be replaced
                        elite_agents = sorted_agents[:elite_count]
                        non_elite_agents = sorted_agents[elite_count:]

                        winner_id, winner_reward = sorted_agents[0]

                        print(f"\n  🏆 WINNER: Agent {winner_id} ({winner_reward:.2f})", flush=True)

                        if elite_count > 1:
                            # Elite preservation mode
                            elite_str = ", ".join([f"Agent {aid} ({r:.2f})" for aid, r in elite_agents])
                            print(f"  Elite preserved ({elite_count}): {elite_str}", flush=True)
                            print(f"  Cloning winner to {len(non_elite_agents)} positions...", flush=True)
                        else:
                            # Winner-takes-all mode (default)
                            print(f"  Cloning Agent {winner_id} to all positions...", flush=True)

                        # Get winner's weights
                        try:
                            command_queues[winner_id].put('GET_WEIGHTS')
                            print(f"  Requested weights from agent {winner_id}", flush=True)

                            while True:
                                msg_type, aid, *data = result_queue.get(timeout=10.0)
                                if msg_type == 'WEIGHTS' and aid == winner_id:
                                    winner_state_dict = data[0]
                                    print(f"  Received weights from agent {winner_id}", flush=True)
                                    break
                                # Ignore other messages while waiting for weights

                            # Broadcast winner's weights to non-elite agents only
                            cloned_count = 0
                            for loser_id, _ in non_elite_agents:
                                state_dict_queues[loser_id].put(winner_state_dict)
                                command_queues[loser_id].put('LOAD_WEIGHTS')
                                print(f"  Sent weights to agent {loser_id}", flush=True)
                                cloned_count += 1

                            if elite_count > 1:
                                print(f"  Cloned winner to {cloned_count} agents, preserved {elite_count} elite", flush=True)
                            else:
                                print(f"  All agents now copies of Agent {winner_id}", flush=True)
                            print(f"{'='*60}\n", flush=True)
                        except Exception as e:
                            print(f"  ERROR during weight cloning: {e}", flush=True)
                            import traceback
                            traceback.print_exc()
                            raise

                        # Update next tournament checkpoint
                        next_tournament_checkpoint += args.selection_frequency
                        print(f"  Next tournament at episode {next_tournament_checkpoint}", flush=True)

                        # Resume all agents with new checkpoint
                        try:
                            for aid in range(args.num_agents):
                                checkpoint_queues[aid].put(next_tournament_checkpoint)
                                command_queues[aid].put('RESUME')
                                print(f"  Resumed agent {aid}", flush=True)
                            print(f"  All agents resumed successfully", flush=True)
                        except Exception as e:
                            print(f"  ERROR during agent resume: {e}", flush=True)
                            import traceback
                            traceback.print_exc()
                            raise

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

                        # Save checkpoint after every tournament
                        import torch

                        # Save generation-specific checkpoint
                        gen_path = os.path.join(args.checkpoint_dir, f'generation_{generation}.pt')
                        torch.save(winner_state_dict, gen_path)
                        print(f"  💾 Saved generation {generation} checkpoint", flush=True)

                        # Save latest generation checkpoint (for easy resume)
                        latest_path = os.path.join(args.checkpoint_dir, 'latest_generation.pt')
                        torch.save(winner_state_dict, latest_path)

                        # Update best overall if improved
                        if winner_reward > best_overall_reward:
                            best_overall_reward = winner_reward
                            best_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
                            torch.save(winner_state_dict, best_path)
                            print(f"  ✓ New best model saved: {best_overall_reward:.2f}", flush=True)

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
