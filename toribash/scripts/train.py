#!/usr/bin/env python3
"""PPO training script for Toribash 2D with self-play support."""

import sys
sys.path.insert(0, sys.path[0] + '/..')

import os
import time
import copy
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from config.env_config import EnvConfig
from config.body_config import JointState
from env.toribash_env import ToribashEnv


class SelfPlayWrapper(gym.Wrapper):
    """Wrapper that uses opponent model for second player."""
    
    def __init__(self, env, opponent_ref):
        super().__init__(env)
        self.opponent_ref = opponent_ref
        
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
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
        return build_observation(self.env.match, player=1)


def train_selfplay(
    total_timesteps: int = 500_000,
    max_turns: int = 20,
    save_path: str = "toribash_selfplay",
    update_opponent_every: int = 10000,
    resume: bool = False,
):
    """Train agent using self-play against copies of itself."""
    
    config = EnvConfig(max_turns=max_turns)
    
    # Track the opponent model
    opponent_ref = [None]
    
    # Create base environment with hold opponent initially
    base_env = ToribashEnv(config)
    
    # Wrap with self-play wrapper and vectorize
    env = DummyVecEnv([lambda: Monitor(SelfPlayWrapper(base_env, opponent_ref))])
    
    # Normalize observations and rewards for stable training
    env = VecNormalize(env, norm_reward=True, gamma=0.99)
    
    # Check for existing model to resume
    model_path = f"{save_path}.zip"
    if resume and os.path.exists(model_path):
        print(f"Resuming from {model_path}...")
        model = PPO.load(model_path, env=env)
    else:
        # Create PPO model with stable hyperparameters
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=1e-4,       # Lower learning rate for stability
            n_steps=512,              # Larger rollout for more stable updates
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,            # Smaller clip range for stability
            ent_coef=0.005,            # Small entropy bonus to prevent policy collapse in self-play
            vf_coef=1.0,              # Higher value function coefficient
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
            # Save and reload to create/update opponent copy
            import tempfile
            import os
            tmp_path = tempfile.mktemp(suffix='.zip')
            self.model.save(tmp_path)
            from stable_baselines3 import PPO
            self.opponent_ref[0] = PPO.load(tmp_path)
            os.remove(tmp_path)
            self.last_opponent_update = self.model.num_timesteps
            print(f"\n  Updated opponent at step {self.model.num_timesteps:,}")
        
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
    train_env = Monitor(ToribashEnv(train_config))
    eval_config = EnvConfig(max_turns=max_turns, opponent_type=opponent_type)
    eval_env = Monitor(ToribashEnv(eval_config))
    
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
        learning_rate=3e-4,
        n_steps=256,
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
    
    print(f"\nWatching AI vs AI: Player A={model_path} vs Player B={opponent_model or opponent}")
    
    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward_a = 0
        total_reward_b = 0
        
        print(f"\nEpisode {ep + 1}/{episodes}")
        print(f"  Scores: A={env.match.scores[0]:.1f} B={env.match.scores[1]:.1f}")
        
        while not env.match.is_done():
            # Player A action
            action_a, _ = model_a.predict(obs, deterministic=False)  # Use stochastic for more varied play
            joint_states_a = [JointState(int(a)) for a in action_a]
            env.match.set_actions(0, joint_states_a)
            
            # Player B action
            if model_b is not None:
                obs_b = build_observation(env.match, player=1)
                action_b, _ = model_b.predict(obs_b, deterministic=False)  # Use stochastic for more varied play
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
