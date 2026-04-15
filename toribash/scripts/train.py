#!/usr/bin/env python3
"""PPO training script for Toribash 2D with self-play support.

This script provides RL training using Proximal Policy Optimization (PPO).
It supports:
- Training against fixed opponents (hold/random/mirror)
- Self-play training (agent plays against copies of itself)
- Model watching (visualize trained agents)
- TensorBoard logging

Usage:
    # Train against hold opponent (100k steps)
    ../.venv/bin/python scripts/train.py --timesteps 100000 --opponent hold

    # Self-play training (500k steps)
    ../.venv/bin/python scripts/train.py --opponent selfplay --timesteps 500000

    # Watch trained agent
    ../.venv/bin/python scripts/train.py --watch toribash_ppo.zip --episodes 5

    # AI vs AI (two trained models)
    ../.venv/bin/python scripts/train.py --watch model_a.zip --opponent selfplay \
           --opponent-model model_b.zip --episodes 5

Command Line Arguments:
    --timesteps: Total training steps (default: 100,000)
    --opponent: Opponent type - "hold", "random", "mirror", or "selfplay"
    --turns: Max turns per episode (default: 20)
    --save: Model save path (default: toribash_ppo)
    --eval-freq: Evaluation frequency (default: 10,000)
    --update-opponent: How often to update opponent in selfplay (default: 10,000)
    --watch: Path to model to watch (skips training)
    --opponent-model: Path to opponent model (for AI vs AI watching)
    --episodes: Number of episodes to watch (default: 5)
"""

import sys
sys.path.insert(0, sys.path[0] + '/..')

import os
import time
import copy
import collections
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack
from config.env_config import EnvConfig
from config.body_config import JointState
from env.toribash_env import ToribashEnv

# Number of stacked observations for temporal memory
N_STACK = 3


class SelfPlayWrapper(gym.Wrapper):
    """Wrapper that uses an opponent model for player 1 in self-play.
    
    This wrapper:
    1. Gets opponent's observation (from player 1's perspective)
    2. Runs opponent's policy to get opponent's action
    3. Sets opponent's actions before stepping the environment
    
    The opponent model is stored via a reference, allowing it to be
    updated during training.
    """
    
    def __init__(self, env, opponent_ref, vec_normalize_ref=None):
        """Initialize the self-play wrapper.
        
        Args:
            env: The underlying ToribashEnv.
            opponent_ref: Mutable list containing the opponent model [PPO or None].
            vec_normalize_ref: Mutable list containing VecNormalize [VecNormalize or None].
        """
        super().__init__(env)
        self.opponent_ref = opponent_ref
        self.vec_normalize_ref = vec_normalize_ref
        self._opp_obs_dim = env.observation_space.shape[0]
        self._opp_frames = collections.deque(maxlen=N_STACK)

    def reset(self, seed=None, options=None):
        """Reset environment and opponent frame buffer."""
        obs, info = self.env.reset(seed=seed, options=options)
        # Initialize opponent frame buffer with zeros
        self._opp_frames.clear()
        for _ in range(N_STACK):
            self._opp_frames.append(np.zeros(self._opp_obs_dim, dtype=np.float32))
        return obs, info

    def step(self, action):
        """Execute one step with opponent action generation.
        
        Before stepping, gets opponent's action using the opponent model.
        """
        # Get opponent action BEFORE step (so it's used in simulation)
        if self.opponent_ref[0] is not None:
            opp_obs = self._get_opponent_obs()
            opp_action, _ = self.opponent_ref[0].predict(opp_obs, deterministic=True)
            opp_states = [JointState(int(a)) for a in opp_action]
            self.env.match.set_actions(1, opp_states)

        # Step with agent's action (already set before this wrapper call)
        obs, reward, done, trunc, info = self.env.step(action)
        return obs, reward, done, trunc, info

    def _get_opponent_obs(self):
        """Get stacked observation for opponent (player 1 perspective)."""
        from env.observation import build_observation
        raw = build_observation(self.env.match, player=1,
                              prev_actions=self.env._prev_opp_actions)
        self._opp_frames.append(raw)
        # Stack frames: oldest first, matching VecFrameStack order
        stacked = np.concatenate(list(self._opp_frames))
        # Apply normalization if available
        if self.vec_normalize_ref and self.vec_normalize_ref[0] is not None:
            stacked = self.vec_normalize_ref[0].normalize_obs(stacked)
        return stacked


def make_ppo(env, config: EnvConfig, tensorboard_log: str) -> PPO:
    """Create a PPO model from config hyperparameters."""
    return PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=config.ppo_net_arch),
        learning_rate=config.ppo_learning_rate,
        n_steps=config.ppo_n_steps,
        batch_size=config.ppo_batch_size,
        n_epochs=config.ppo_n_epochs,
        target_kl=config.ppo_target_kl,
        gamma=config.ppo_gamma,
        gae_lambda=config.ppo_gae_lambda,
        clip_range=config.ppo_clip_range,
        ent_coef=config.ppo_ent_coef,
        vf_coef=config.ppo_vf_coef,
        max_grad_norm=config.ppo_max_grad_norm,
        verbose=1,
        tensorboard_log=tensorboard_log,
    )


def train_selfplay(
    total_timesteps: int = 500_000,
    max_turns: int = 20,
    save_path: str = "toribash_selfplay",
    update_opponent_every: int = 10000,
    resume: bool = False,
):
    """Train agent using self-play against copies of itself.
    
    Self-play Training:
    1. Train agent against current opponent model
    2. Periodically update opponent to best model seen
    3. Save best model separately
    
    Args:
        total_timesteps: Total training steps.
        max_turns: Max turns per episode.
        save_path: Path prefix for saving models.
        update_opponent_every: Steps between opponent updates.
        resume: Whether to resume from existing model.
    
    Returns:
        Trained PPO model.
    """
    config = EnvConfig(max_turns=max_turns)
    
    # Track the opponent model and VecNormalize reference
    opponent_ref = [None]
    vec_normalize_ref = [None]

    # Create base environment
    base_env = ToribashEnv(config)

    # Wrap with self-play wrapper and vectorize
    selfplay_env = SelfPlayWrapper(base_env, opponent_ref, vec_normalize_ref)
    env = DummyVecEnv([lambda: Monitor(selfplay_env)])
    env = VecFrameStack(env, n_stack=N_STACK)
    env = VecNormalize(env, norm_reward=True, gamma=0.99)
    vec_normalize_ref[0] = env  # Allow opponent to normalize observations
    
    # Model paths
    model_path = f"{save_path}.zip"
    best_path = f"{save_path}_best.zip"
    
    # Check for existing models to resume or continue
    if resume and os.path.exists(model_path):
        print(f"Resuming from {model_path}...")
        model = PPO.load(model_path, env=env)
        if os.path.exists(best_path):
            from stable_baselines3 import PPO as PPOClass
            opponent_ref[0] = PPOClass.load(best_path)
            print(f"Playing against best model: {best_path}")
    elif os.path.exists(best_path):
        # No current model but have best - start fresh with best as opponent
        from stable_baselines3 import PPO as PPOClass
        opponent_ref[0] = PPOClass.load(best_path)
        print(f"Starting fresh, playing against existing best: {best_path}")
        model = make_ppo(env, config, f"{save_path}_tensorboard")
    else:
        # Fresh start
        model = make_ppo(env, config, f"{save_path}_tensorboard")
    
    print(f"Training PPO with self-play...")
    print(f"  Max turns per episode: {max_turns}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Update opponent every: {update_opponent_every:,} steps")
    
    start_time = time.time()
    
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        callback=SelfPlayCallback(
            save_path=save_path,
            update_opponent_every=update_opponent_every,
            opponent_ref=opponent_ref,
            model=model,
        ),
    )
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f} seconds")
    
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")
    
    env.close()
    return model


class SelfPlayCallback(BaseCallback):
    """Callback to update opponent model and save best during self-play training."""
    
    def __init__(self, save_path, update_opponent_every, opponent_ref, model, verbose=0):
        """Initialize callback.
        
        Args:
            save_path: Path prefix for saved models.
            update_opponent_every: Steps between opponent updates.
            opponent_ref: Mutable list containing opponent model.
            model: The training PPO model.
        """
        super().__init__(verbose)
        self.save_path = save_path
        self.update_opponent_every = update_opponent_every
        self.opponent_ref = opponent_ref
        self.model = model
        self.last_opponent_update = 0
        self.best_reward = float('-inf')
        
    def _on_step(self) -> bool:
        """Called each training step."""
        # Update opponent to best model periodically
        if self.model.num_timesteps - self.last_opponent_update >= self.update_opponent_every:
            best_path = f"{self.save_path}_best.zip"
            if os.path.exists(best_path):
                from stable_baselines3 import PPO
                self.opponent_ref[0] = PPO.load(best_path)
                self.last_opponent_update = self.model.num_timesteps
                print(f"\n  Updated opponent to best model at step {self.model.num_timesteps:,}")
        
        # Save best model based on reward
        if len(self.model.ep_info_buffer) > 0:
            ep_rew = self.model.ep_info_buffer[-1]["r"]
            if ep_rew > self.best_reward:
                self.best_reward = ep_rew
                self.model.save(f"{self.save_path}_best")
        
        return True


def train(
    total_timesteps: int = 1_000_000,
    opponent_type: str = "hold",
    max_turns: int = 20,
    save_path: str = "toribash_ppo",
    eval_freq: int = 10000,
):
    """Train a PPO agent against a fixed opponent.
    
    Args:
        total_timesteps: Total training steps.
        opponent_type: Type of opponent ("hold", "random", "mirror").
        max_turns: Max turns per episode.
        save_path: Path prefix for saving models.
        eval_freq: Steps between evaluation episodes.
    
    Returns:
        Trained PPO model.
    """
    train_config = EnvConfig(max_turns=max_turns, opponent_type=opponent_type)
    train_env = DummyVecEnv([lambda: Monitor(ToribashEnv(train_config))])
    train_env = VecFrameStack(train_env, n_stack=N_STACK)
    train_env = VecNormalize(train_env, norm_reward=True, gamma=0.99)

    eval_config = EnvConfig(max_turns=max_turns, opponent_type=opponent_type)
    eval_env = DummyVecEnv([lambda: Monitor(ToribashEnv(eval_config))])
    eval_env = VecFrameStack(eval_env, n_stack=N_STACK)
    eval_env = VecNormalize(eval_env, norm_reward=False, training=False, gamma=0.99)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_path}_best",
        log_path=f"{save_path}_logs",
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )
    
    model = make_ppo(train_env, train_config, f"{save_path}_tensorboard")
    
    print(f"Training PPO agent...")
    print(f"  Opponent: {opponent_type}")
    print(f"  Max turns per episode: {max_turns}")
    print(f"  Total timesteps: {total_timesteps:,}")
    
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True,
    )
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f} seconds")
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")
    
    train_env.close()
    eval_env.close()
    return model


def watch_trained_agent(
    model_path: str,
    episodes: int = 5,
    opponent: str = "hold",
    opponent_model: str = None,
):
    """Watch a trained agent play with animated physics.
    
    Args:
        model_path: Path to player A's model.
        episodes: Number of episodes to watch.
        opponent: Opponent type ("hold", "random", "mirror").
        opponent_model: Path to opponent model (for AI vs AI).
    """
    import pygame
    from config.env_config import EnvConfig
    from config.body_config import JointState
    from env.toribash_env import ToribashEnv
    from stable_baselines3 import PPO
    from game.scoring import compute_turn_result, compute_reward, EXEMPT_GROUND_SEGMENTS, GROUND_PENALTIES
    from env.observation import build_observation
    
    pygame.init()
    model_a = PPO.load(model_path)
    model_b = PPO.load(opponent_model) if opponent_model else None

    config = EnvConfig(max_turns=20, opponent_type=opponent)
    env = ToribashEnv(config, render_mode="human")
    obs_dim = env.observation_space.shape[0]

    # Frame stack buffers for each player
    frames_a = collections.deque(maxlen=N_STACK)
    frames_b = collections.deque(maxlen=N_STACK)

    def stack_obs(frames, raw_obs):
        frames.append(raw_obs)
        return np.concatenate(list(frames))

    print(f"\nWatching AI vs AI: Player A={model_path} vs Player B={opponent_model or opponent}")

    for ep in range(episodes):
        obs, _ = env.reset()
        # Initialize frame buffers with zeros
        frames_a.clear()
        frames_b.clear()
        for _ in range(N_STACK):
            frames_a.append(np.zeros(obs_dim, dtype=np.float32))
            frames_b.append(np.zeros(obs_dim, dtype=np.float32))
        total_reward_a = 0

        print(f"\nEpisode {ep + 1}/{episodes}")

        while not env.match.is_done():
            # Player A action
            stacked_a = stack_obs(frames_a, obs)
            action_a, _ = model_a.predict(stacked_a, deterministic=False)
            joint_states_a = [JointState(int(a)) for a in action_a]
            env.match.set_actions(0, joint_states_a)

            # Player B action
            if model_b is not None:
                raw_b = build_observation(env.match, player=1,
                                          prev_actions=env._prev_opp_actions)
                stacked_b = stack_obs(frames_b, raw_b)
                action_b, _ = model_b.predict(stacked_b, deterministic=False)
                joint_states_b = [JointState(int(a)) for a in action_b]
                env.match.set_actions(1, joint_states_b)
            else:
                opp_action = env._get_opponent_action()
                env.match.set_actions(1, opp_action)

            env.match.world.collision_handler.clear_turn()

            # Animated playback
            for _ in range(config.steps_per_turn):
                env.render()
                env.match.world.step()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        pygame.quit()
                        return

                pygame.time.wait(16)

            # Score the turn (same logic as Match.simulate_turn)
            result = compute_turn_result(env.match.world.collision_handler, config)
            env.match.scores[0] += result.damage_a_to_b
            env.match.scores[1] += result.damage_b_to_a
            bad_a = result.ground_segments_a - EXEMPT_GROUND_SEGMENTS
            bad_b = result.ground_segments_b - EXEMPT_GROUND_SEGMENTS
            for seg in bad_a:
                env.match.scores[0] += GROUND_PENALTIES.get(seg, GROUND_PENALTIES["default"])
            for seg in bad_b:
                env.match.scores[1] += GROUND_PENALTIES.get(seg, GROUND_PENALTIES["default"])
            env.match.turn_results.append(result)
            env.match.turn += 1

            # Compute reward using the same function as training
            done = env.match.is_done()
            won = done and env.match.get_winner() == 0
            reward = compute_reward(result, player=0, config=config, won=won)
            total_reward_a += reward

            # Update action memory and build next observation
            env._prev_actions = joint_states_a
            if model_b is not None:
                env._prev_opp_actions = joint_states_b
            else:
                env._prev_opp_actions = opp_action
            obs = build_observation(env.match, player=0, prev_actions=env._prev_actions)

        print(f"  Player A reward: {total_reward_a:.2f}")
        print(f"  Final scores: A={env.match.scores[0]:.1f} B={env.match.scores[1]:.1f}")
        print(f"  Winner: {'A' if env.match.get_winner() == 0 else 'B' if env.match.get_winner() == 1 else 'Draw'}")

    env.close()
    pygame.quit()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train PPO agent for Toribash 2D")
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--opponent", type=str, default="hold",
                       choices=["hold", "random", "mirror", "selfplay"])
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument("--save", type=str, default="toribash_ppo")
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--update-opponent", type=int, default=10000)
    parser.add_argument("--watch", type=str, default=None,
                       help="Path to model to watch (skips training)")
    parser.add_argument("--opponent-model", type=str, default=None,
                       help="Path to opponent model for AI vs AI watching")
    parser.add_argument("--episodes", type=int, default=5)
    
    args = parser.parse_args()
    
    if args.watch:
        # Watch mode
        watch_trained_agent(args.watch, args.episodes, args.opponent, args.opponent_model)
    elif args.opponent == "selfplay":
        # Self-play training
        save_path = args.save if args.save != "toribash_ppo" else "toribash_selfplay"
        resume = os.path.exists(f"{save_path}.zip")
        train_selfplay(
            total_timesteps=args.timesteps,
            max_turns=args.turns,
            save_path=save_path,
            update_opponent_every=args.update_opponent,
            resume=resume,
        )
    else:
        # Standard training against fixed opponent
        train(
            total_timesteps=args.timesteps,
            opponent_type=args.opponent,
            max_turns=args.turns,
            save_path=args.save,
            eval_freq=args.eval_freq,
        )
