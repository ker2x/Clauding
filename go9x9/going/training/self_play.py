"""
Self-play game generation for 9x9 Go training.
"""

import numpy as np
import torch
import torch.multiprocessing as mp
import time
from typing import List, Tuple
from queue import Empty

try:
    from ..engine.game import GoGame
    from ..mcts.mcts import MCTS
    from ..network.resnet import GoNetwork
    from ..engine.action_encoder import NUM_ACTIONS, PASS_ACTION
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from going.engine.game import GoGame
    from going.mcts.mcts import MCTS
    from going.network.resnet import GoNetwork
    from going.engine.action_encoder import NUM_ACTIONS, PASS_ACTION


class SelfPlayGame:
    """Play a single game using MCTS for move selection."""

    def __init__(self, network: GoNetwork, config, device: torch.device):
        self.network = network
        self.config = config
        self.device = device

    def play_game(self, on_move=None) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        """
        Play one complete game using MCTS.

        Returns:
            (states, policies, values): Training examples from the game
        """
        game = GoGame()
        game.komi = self.config.KOMI

        mcts = MCTS(
            network=self.network,
            c_puct=self.config.C_PUCT,
            num_simulations=self.config.MCTS_SIMS_SELFPLAY,
            dirichlet_alpha=self.config.DIRICHLET_ALPHA,
            dirichlet_epsilon=self.config.DIRICHLET_EPSILON,
            device=self.device,
            batch_size=getattr(self.config, 'MCTS_BATCH_SIZE', 1)
        )

        states = []
        policies = []
        players = []

        move_count = 0
        max_moves = self.config.MAX_GAME_LENGTH

        while not game.is_terminal() and move_count < max_moves:
            state = game.to_neural_input()
            states.append(state)
            players.append(game.current_player)

            # Run MCTS
            policy = mcts.search(game, add_noise=True)
            policies.append(policy)

            # Temperature schedule
            if move_count < self.config.TEMPERATURE_THRESHOLD:
                temperature = self.config.TEMPERATURE
            else:
                temperature = 0.0

            action = self._sample_action(policy, temperature, game)

            if on_move:
                on_move(game, policy, move_count, game.current_player)

            if action != -1 and game.make_action(action):
                move_count += 1
            else:
                break

        # Assign values
        outcome = game.get_result()
        values = self._compute_values(players, outcome, game.current_player)

        return states, policies, values

    def _sample_action(self, policy: np.ndarray, temperature: float, game=None) -> int:
        if game is not None:
            actual_legal = set(game.get_legal_actions())
            policy_actions = np.where(policy > 0)[0]
            legal_actions = np.array([a for a in policy_actions if a in actual_legal])

            if len(legal_actions) == 0:
                if len(actual_legal) > 0:
                    return int(np.random.choice(list(actual_legal)))
                else:
                    return -1
        else:
            legal_actions = np.where(policy > 0)[0]
            if len(legal_actions) == 0:
                return -1

        if temperature == 0:
            return int(legal_actions[np.argmax(policy[legal_actions])])
        else:
            legal_probs = policy[legal_actions] ** (1.0 / temperature)
            legal_probs = legal_probs / legal_probs.sum()
            return int(np.random.choice(legal_actions, p=legal_probs))

    def _compute_values(self, players: List[int], outcome: float,
                        final_player: int) -> List[float]:
        values = []
        for player in players:
            if outcome == 0.0:
                value = 0.0
            elif player == final_player:
                value = outcome
            else:
                value = -outcome
            values.append(value)
        return values


def play_games_sequential(
    network: GoNetwork, config, device: torch.device,
    num_games: int, on_move=None
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    all_states = []
    all_policies = []
    all_values = []

    self_play = SelfPlayGame(network, config, device)
    bar_width = 40
    print("  ", end="", flush=True)

    for game_idx in range(num_games):
        states, policies, values = self_play.play_game(on_move=on_move)
        all_states.extend(states)
        all_policies.extend(policies)
        all_values.extend(values)

        progress = (game_idx + 1) / num_games
        filled = int(bar_width * progress)
        bar = '=' * filled + '-' * (bar_width - filled)
        print(f"\r  [{bar}] {game_idx + 1}/{num_games} games | "
              f"{len(all_states)} examples", end="", flush=True)

    print()
    return all_states, all_policies, all_values


def self_play_worker(rank, network, config, device, games_to_play, result_queue):
    self_play = SelfPlayGame(network, config, device)
    for _ in range(games_to_play):
        states, policies, values = self_play.play_game()
        if len(states) > 0:
            result_queue.put((states, policies, values))


def play_games_parallel(
    network: GoNetwork, config, device: torch.device,
    num_games: int
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    num_workers = config.NUM_WORKERS
    games_per_worker = num_games // num_workers
    remainder = num_games % num_workers

    result_queue = mp.Queue()
    processes = []

    print(f"  Starting {num_workers} workers...")
    for rank in range(num_workers):
        worker_games = games_per_worker + (1 if rank < remainder else 0)
        if worker_games == 0:
            continue
        p = mp.Process(
            target=self_play_worker,
            args=(rank, network, config, device, worker_games, result_queue)
        )
        p.start()
        processes.append(p)

    all_states = []
    all_policies = []
    all_values = []
    collected_games = 0
    bar_width = 40
    print("  ", end="", flush=True)

    while collected_games < num_games:
        try:
            data = result_queue.get(timeout=0.1)
            states, policies, values = data
            all_states.extend(states)
            all_policies.extend(policies)
            all_values.extend(values)
            collected_games += 1

            progress = collected_games / num_games
            filled = int(bar_width * progress)
            bar = '=' * filled + '-' * (bar_width - filled)
            print(f"\r  [{bar}] {collected_games}/{num_games} games | "
                  f"{len(all_states)} examples", end="", flush=True)
        except Empty:
            if not any(p.is_alive() for p in processes) and result_queue.empty():
                print("\n  Warning: All workers died prematurely.")
                break
            continue

    print()
    for p in processes:
        p.join()

    return all_states, all_policies, all_values
