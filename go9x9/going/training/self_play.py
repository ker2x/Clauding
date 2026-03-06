"""
Self-play game generation for 9x9 Go training.
"""

import copy
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
    from ..engine.board import BLACK
    from ..engine.scoring import compute_ownership_map
    from .inference_server import inference_server_worker, RemoteNetwork
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from going.engine.game import GoGame
    from going.mcts.mcts import MCTS
    from going.network.resnet import GoNetwork
    from going.engine.action_encoder import NUM_ACTIONS, PASS_ACTION
    from going.engine.board import BLACK
    from going.engine.scoring import compute_ownership_map
    from going.training.inference_server import inference_server_worker, RemoteNetwork


class SelfPlayGame:
    """Play a single game using MCTS for move selection."""

    def __init__(self, network: GoNetwork, config, device: torch.device):
        self.network = network
        self.config = config
        self.device = device

    def play_game(self, on_move=None) -> Tuple[
        List[np.ndarray], List[np.ndarray], List[float],
        List[np.ndarray], List[float]
    ]:
        """
        Play one complete game using MCTS.

        Uses playout cap randomization: P_FAST_MOVE fraction of moves use
        MCTS_SIMS_FAST sims and are not stored; the rest use full sims and
        are stored for training.

        Returns:
            (states, policies, values, ownerships, surprises)
        """
        game = GoGame()
        game.komi = self.config.KOMI

        mcts_common = dict(
            network=self.network,
            c_puct=self.config.C_PUCT,
            dirichlet_alpha=self.config.DIRICHLET_ALPHA,
            dirichlet_epsilon=self.config.DIRICHLET_EPSILON,
            device=self.device,
            batch_size=getattr(self.config, 'MCTS_BATCH_SIZE', 1)
        )
        mcts_slow = MCTS(num_simulations=self.config.MCTS_SIMS_SELFPLAY, **mcts_common)
        mcts_fast = MCTS(num_simulations=getattr(self.config, 'MCTS_SIMS_FAST', 20), **mcts_common)
        p_fast = getattr(self.config, 'P_FAST_MOVE', 0.0)

        states = []
        policies = []
        players = []
        surprises = []

        move_count = 0
        max_moves = self.config.MAX_GAME_LENGTH

        while not game.is_terminal() and move_count < max_moves:
            # Playout cap randomization: fast moves advance game but aren't stored
            is_fast = (move_count >= self.config.TEMPERATURE_THRESHOLD
                       and np.random.random() < p_fast)
            mcts = mcts_fast if is_fast else mcts_slow

            # Run MCTS (root_prior stored on mcts after this call)
            policy = mcts.search(game, add_noise=not is_fast)

            if not is_fast:
                state = game.to_neural_input()
                states.append(state)
                players.append(game.current_player)
                policies.append(policy)

                # Compute surprise: KL(mcts_policy || network_prior)
                root_prior = mcts.root_prior
                eps = 1e-8
                kl = 0.0
                for a in np.where(policy > eps)[0]:
                    kl += policy[a] * np.log(policy[a] / (root_prior[a] + eps))
                surprises.append(max(0.0, kl))

            # Temperature schedule
            if move_count < self.config.TEMPERATURE_THRESHOLD:
                temperature = self.config.TEMPERATURE
            else:
                temperature = 0.0

            action = self._sample_action(policy, temperature, game)

            if on_move and not is_fast:
                on_move(game, policy, move_count, game.current_player)

            if action != -1 and game.make_action(action):
                move_count += 1
            else:
                break

        # Assign values
        outcome = game.get_result()
        values = self._compute_values(players, outcome, game.current_player)

        # Compute ownership from final board, per-player perspective
        ownership_abs = compute_ownership_map(game.board)
        ownerships = []
        for player in players:
            if player == BLACK:
                ownerships.append(ownership_abs.copy())
            else:
                ownerships.append(1.0 - ownership_abs)

        return states, policies, values, ownerships, surprises

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
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[np.ndarray], List[float]]:
    all_states = []
    all_policies = []
    all_values = []
    all_ownerships = []
    all_surprises = []

    self_play = SelfPlayGame(network, config, device)
    bar_width = 40
    print("  ", end="", flush=True)

    for game_idx in range(num_games):
        states, policies, values, ownerships, surprises = self_play.play_game(on_move=on_move)
        all_states.extend(states)
        all_policies.extend(policies)
        all_values.extend(values)
        all_ownerships.extend(ownerships)
        all_surprises.extend(surprises)

        progress = (game_idx + 1) / num_games
        filled = int(bar_width * progress)
        bar = '=' * filled + '-' * (bar_width - filled)
        print(f"\r  [{bar}] {game_idx + 1}/{num_games} games | "
              f"{len(all_states)} examples", end="", flush=True)

    print()
    return all_states, all_policies, all_values, all_ownerships, all_surprises


def self_play_worker(rank, network, config, device, games_to_play, result_queue):
    self_play = SelfPlayGame(network, config, device)
    for _ in range(games_to_play):
        states, policies, values, ownerships, surprises = self_play.play_game()
        if len(states) > 0:
            result_queue.put((states, policies, values, ownerships, surprises))


def play_games_parallel(
    network: GoNetwork, config, device: torch.device,
    num_games: int
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[np.ndarray], List[float]]:
    num_workers = config.NUM_WORKERS
    games_per_worker = num_games // num_workers
    remainder = num_games % num_workers

    result_queue = mp.Queue()
    processes = []

    # MPS (or CUDA) mode: one inference server process owns the network on the
    # target device; workers use RemoteNetwork proxies and run on CPU.
    use_inference_server = device.type != "cpu"
    server_process = None
    worker_result_queues = None

    if use_inference_server:
        worker_result_queues = [mp.Queue() for _ in range(num_workers)]
        request_queue = mp.Queue()
        # Spawn can't pickle MPS tensors — give the server a CPU copy;
        # inference_server_worker moves it to the target device itself.
        server_network = copy.deepcopy(network).cpu()
        server_process = mp.Process(
            target=inference_server_worker,
            args=(server_network, device.type, request_queue, worker_result_queues),
            daemon=True,
        )
        server_process.start()
        print(f"  Inference server started on {device.type}.")

    print(f"  Starting {num_workers} workers...")
    for rank in range(num_workers):
        worker_games = games_per_worker + (1 if rank < remainder else 0)
        if worker_games == 0:
            continue

        if use_inference_server:
            worker_net = RemoteNetwork(rank, request_queue, worker_result_queues[rank])
            worker_device = torch.device("cpu")
        else:
            worker_net = network
            worker_device = device

        p = mp.Process(
            target=self_play_worker,
            args=(rank, worker_net, config, worker_device, worker_games, result_queue)
        )
        p.start()
        processes.append(p)

    all_states = []
    all_policies = []
    all_values = []
    all_ownerships = []
    all_surprises = []
    collected_games = 0
    bar_width = 40
    print("  ", end="", flush=True)

    while collected_games < num_games:
        try:
            data = result_queue.get(timeout=0.1)
            states, policies, values, ownerships, surprises = data
            all_states.extend(states)
            all_policies.extend(policies)
            all_values.extend(values)
            all_ownerships.extend(ownerships)
            all_surprises.extend(surprises)
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

    if server_process is not None:
        request_queue.put(None)  # shutdown sentinel
        server_process.join(timeout=5)

    return all_states, all_policies, all_values, all_ownerships, all_surprises
