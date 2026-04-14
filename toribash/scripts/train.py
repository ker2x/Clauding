#!/usr/bin/env python3
"""PPO training script for Toribash 2D with self-play support."""

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

N_STACK = 3


class SelfPlayWrapper(gym.Wrapper):
    """Wrapper that uses opponent model for second player."""

    def __init__(self, env, opponent_ref, vec_normalize_ref=None):
        super().__init__(env)
        self.opponent_ref = opponent_ref
        self.vec_normalize_ref = vec_normalize_ref  # mutable list [VecNormalize | None]
        self._opp_obs_dim = env.observation_space.shape[0]
        self._opp_frames = collections.deque(maxlen=N_STACK)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        # Initialize opponent frame buffer with zeros
        self._opp_frames.clear()
        for _ in range(N_STACK):
            self._opp_frames.append(np.zeros(self._opp_obs_dim, dtype=np.float32))
        return obs, info

    def step(self, action):
        # Get opponent action BEFORE step (so it's used in simulation)
        if self.opponent_ref[0] is not None:
            opp_obs = self._get_opponent_obs()
            opp_action, _ = self.opponent_ref[0].predict(opp_obs, deterministic=True)
            opp_states = [JointState(int(a)) for a in opp_action]
            self.env.match.set_actions(1, opp_states)

        # Now step (agent's actions already set before this wrapper)
        obs, reward, done, trunc, info = self.env.step(action)
        return obs, reward, done, trunc, info

    def _get_opponent_obs(self):
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


def train_selfplay(
    total_timesteps: int = 500_000,
    max_turns: int = 20,
    save_path: str = "toribash_selfplay",
    update_opponent_every: int = 10000,
    resume: bool = False,
):
    """Train agent using self-play against copies of itself."""
    
    config = EnvConfig(max_turns=max_turns)
    
    # Track the opponent model and VecNormalize reference
    opponent_ref = [None]
    vec_normalize_ref = [None]

    # Create base environment with hold opponent initially
    base_env = ToribashEnv(config)

    # Wrap with self-play wrapper and vectorize
    selfplay_env = SelfPlayWrapper(base_env, opponent_ref, vec_normalize_ref)
    env = DummyVecEnv([lambda: Monitor(selfplay_env)])
    env = VecFrameStack(env, n_stack=N_STACK)
    env = VecNormalize(env, norm_reward=True, gamma=0.99)
    vec_normalize_ref[0] = env  # Now opponent can normalize its obs
    
    # Check for existing model to resume and set up opponent
    model_path = f"{save_path}.zip"
    best_path = f"{save_path}_best.zip"
    
    if resume and os.path.exists(model_path):
        print(f"Resuming from {model_path}...")
        model = PPO.load(model_path, env=env)
        # Load best model as opponent (if it exists)
        if os.path.exists(best_path):
            from stable_baselines3 import PPO as PPOClass
            opponent_ref[0] = PPOClass.load(best_path)
            print(f"Playing against best model: {best_path}")
    elif os.path.exists(best_path):
        # No current model but have best - start fresh with best as opponent
        from stable_baselines3 import PPO as PPOClass
        opponent_ref[0] = PPOClass.load(best_path)
        print(f"Starting fresh, playing against existing best: {best_path}")
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_rate=5e-5,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            target_kl=0.05,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=f"{save_path}_tensorboard",
        )
    else:
        # Create PPO model
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_rate=5e-5,  # Lower for stability with clip=0.2
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            target_kl=0.05,  # Guardrail that only fires on genuine instability
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=f"{save_path}_tensorboard",
        )
    
    print(f"Training PPO with self-play...")
    print(f"  Max turns per episode: {max_turns}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Update opponent every: {update_opponent_every:,} steps")
    
    start_time = time.time()
    best_reward = float('-inf')
    
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
    """Callback to update opponent model during self-play training."""
    
    def __init__(self, save_path, update_opponent_every, opponent_ref, model, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.update_opponent_every = update_opponent_every
        self.opponent_ref = opponent_ref
        self.model = model
        self.last_opponent_update = 0
        self.best_reward = float('-inf')
        
    def _on_step(self) -> bool:
        if self.model.num_timesteps - self.last_opponent_update >= self.update_opponent_every:
            # Update opponent to play against the BEST model, not the current one
            best_path = f"{self.save_path}_best.zip"
            from stable_baselines3 import PPO
            if os.path.exists(best_path):
                self.opponent_ref[0] = PPO.load(best_path)
                self.last_opponent_update = self.model.num_timesteps
                print(f"\n  Updated opponent to best model at step {self.model.num_timesteps:,}")
        
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
    """Train a PPO agent on Toribash 2D."""
    
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
    
    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=f"{save_path}_tensorboard",
    )
    
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


def watch_trained_agent(model_path: str, episodes: int = 5, opponent: str = "hold", opponent_model: str = None):
    """Watch a trained agent play with animated physics.
    
    Args:
        model_path: Path to player A's model
        episodes: Number of episodes to watch
        opponent: Opponent type ("hold", "random", "mirror")
        opponent_model: Path to opponent model (for selfplay mode)
    """
    import pygame
    from config.env_config import EnvConfig
    from config.body_config import JointState
    from env.toribash_env import ToribashEnv
    from stable_baselines3 import PPO
    from game.scoring import compute_turn_result
    from env.observation import build_observation
    
    pygame.init()
    model_a = PPO.load(model_path)
    model_b = PPO.load(opponent_model) if opponent_model else None

    config = EnvConfig(max_turns=20, opponent_type=opponent)
    env = ToribashEnv(config, render_mode="human")
    obs_dim = env.observation_space.shape[0]

    # Frame stack buffers for each player (no VecNormalize — approximate for visualization)
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
            # Player A action (stacked obs)
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

            for _ in range(config.steps_per_turn):
                env.render()
                env.match.world.step()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        pygame.quit()
                        return

                pygame.time.wait(16)

            result = compute_turn_result(env.match.world.collision_handler, config)
            env.match.scores[0] += result.damage_a_to_b
            env.match.scores[1] += result.damage_b_to_a
            env.match.turn_results.append(result)
            env.match.turn += 1

            obs, _, _, _, _ = env.step(action_a)
            total_reward_a += result.damage_a_to_b * config.reward_damage_dealt + \
                             result.damage_b_to_a * config.reward_damage_taken

        print(f"  Player A reward: {total_reward_a:.2f}")
        print(f"  Final scores: A={env.match.scores[0]:.1f} B={env.match.scores[1]:.1f}")
        print(f"  Winner: {'A' if env.match.get_winner() == 0 else 'B' if env.match.get_winner() == 1 else 'Draw'}")

    env.close()
    pygame.quit()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train PPO agent for Toribash 2D")
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--opponent", type=str, default="hold", choices=["hold", "random", "mirror", "selfplay"])
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument("--save", type=str, default="toribash_ppo")
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--update-opponent", type=int, default=10000)
    parser.add_argument("--watch", type=str, default=None)
    parser.add_argument("--opponent-model", type=str, default=None, help="Path to opponent model for AI vs AI watching")
    parser.add_argument("--episodes", type=int, default=5)
    
    args = parser.parse_args()
    
    if args.watch:
        watch_trained_agent(args.watch, args.episodes, args.opponent, args.opponent_model)
    elif args.opponent == "selfplay":
        save_path = args.save if args.save != "toribash_ppo" else "toribash_selfplay"
        # Check if model exists and auto-resume
        resume = os.path.exists(f"{save_path}.zip")
        train_selfplay(
            total_timesteps=args.timesteps,
            max_turns=args.turns,
            save_path=save_path,
            update_opponent_every=args.update_opponent,
            resume=resume,
        )
    else:
        train(
            total_timesteps=args.timesteps,
            opponent_type=args.opponent,
            max_turns=args.turns,
            save_path=args.save,
            eval_freq=args.eval_freq,
        )
