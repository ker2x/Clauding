"""
Self-play game generation for training.
"""

import numpy as np
import torch
import torch.multiprocessing as mp
import time
from typing import List, Tuple
from queue import Empty

try:
    from ..engine.game import CheckersGame
    from ..mcts.mcts import MCTS
    from ..network.resnet import CheckersNetwork
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from checkers8x8.engine.game import CheckersGame
    from checkers8x8.mcts.mcts import MCTS
    from checkers8x8.network.resnet import CheckersNetwork


class SelfPlayGame:
    """
    Play a single game using MCTS for move selection.

    Collects training examples during play.
    """

    def __init__(
        self,
        network: CheckersNetwork,
        config,
        device: torch.device
    ):
        """
        Initialize self-play game generator.

        Args:
            network: Neural network for evaluation
            config: Configuration object with hyperparameters
            device: PyTorch device
        """
        self.network = network
        self.config = config
        self.device = device

    def play_game(self, on_move=None) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        """
        Play one complete game using MCTS.

        Args:
            on_move: Optional callback function called after each move
                    Signature: (game_state, policy, move_count, current_player)

        Returns:
            (states, policies, values): Training examples from the game
                - states: List of game states (8, 8, 8)
                - policies: List of MCTS policy distributions (128,)
                - values: List of game outcomes from each state's perspective
        """
        game = CheckersGame()

        # Create MCTS instance
        mcts = MCTS(
            network=self.network,
            c_puct=self.config.C_PUCT,
            num_simulations=self.config.MCTS_SIMS_SELFPLAY,
            dirichlet_alpha=self.config.DIRICHLET_ALPHA,
            dirichlet_epsilon=self.config.DIRICHLET_EPSILON,
            device=self.device
        )

        # Storage for training examples
        states = []
        policies = []
        players = []  # Track which player made each move

        move_count = 0
        max_moves = self.config.MAX_GAME_LENGTH

        # Play until terminal or max moves
        while not game.is_terminal() and move_count < max_moves:
            # Store current state
            state = game.to_neural_input()
            states.append(state)
            players.append(game.current_player)

            # Run MCTS to get policy
            policy = mcts.search(game, add_noise=True)
            policies.append(policy)

            # Select move based on temperature
            if move_count < self.config.TEMPERATURE_THRESHOLD:
                # Exploration phase: use temperature
                temperature = self.config.TEMPERATURE
            else:
                # Exploitation phase: greedy
                temperature = 0.0

            # Sample action from policy
            action = self._sample_action(policy, temperature)

            # Call visualization callback before applying move
            if on_move:
                on_move(game, policy, move_count, game.current_player)

            # Apply action
            if action != -1 and game.make_action(action):
                move_count += 1
            else:
                # Should not happen, but safety check
                break

        # Game finished - assign values to all positions
        outcome = game.get_result()
        values = self._compute_values(players, outcome, game.current_player)

        return states, policies, values

    def _sample_action(self, policy: np.ndarray, temperature: float) -> int:
        """
        Sample action from policy distribution.

        Args:
            policy: Policy distribution (128,)
            temperature: Temperature for sampling

        Returns:
            Selected action index
        """
        legal_actions = np.where(policy > 0)[0]

        if len(legal_actions) == 0:
            return -1

        if temperature == 0:
            # Greedy: select max probability
            return legal_actions[np.argmax(policy[legal_actions])]
        else:
            # Temperature scaling
            legal_probs = policy[legal_actions] ** (1.0 / temperature)
            legal_probs = legal_probs / legal_probs.sum()
            return np.random.choice(legal_actions, p=legal_probs)

    def _compute_values(
        self,
        players: List[int],
        outcome: float,
        final_player: int
    ) -> List[float]:
        """
        Compute game outcome for each position.

        Args:
            players: List of players who made each move
            outcome: Game result from final player's perspective
            final_player: Player whose turn it was at game end

        Returns:
            List of values for each position (±1 or 0)
        """
        values = []

        for player in players:
            if outcome == 0.0:
                # Draw
                value = 0.0
            elif player == final_player:
                # Same player as final position
                value = outcome
            else:
                # Opposite player
                value = -outcome

            values.append(value)

        return values


def play_games_sequential(
    network: CheckersNetwork,
    config,
    device: torch.device,
    num_games: int,
    on_move=None
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    """
    Play multiple games sequentially.

    Args:
        network: Neural network
        config: Configuration
        device: Device
        num_games: Number of games to play
        on_move: Optional callback for visualization

    Returns:
        (states, policies, values): All training examples
    """
    all_states = []
    all_policies = []
    all_values = []

    self_play = SelfPlayGame(network, config, device)

    # Print progress bar
    print("  ", end="", flush=True)
    bar_width = 40

    for game_idx in range(num_games):
        states, policies, values = self_play.play_game(on_move=on_move)

        all_states.extend(states)
        all_policies.extend(policies)
        all_values.extend(values)

        # Update progress bar
        progress = (game_idx + 1) / num_games
        filled = int(bar_width * progress)
        bar = '█' * filled + '░' * (bar_width - filled)
        print(f"\r  [{bar}] {game_idx + 1}/{num_games} games | "
              f"{len(all_states)} examples", end="", flush=True)

    print()  # New line after completion

    return all_states, all_policies, all_values


def self_play_worker(
    rank: int,
    network: CheckersNetwork,
    config,
    device: torch.device,
    games_to_play: int,
    result_queue: mp.Queue,
    vis_queue: mp.Queue = None
):
    """
    Worker process for parallel self-play.
    
    Args:
        rank: Worker ID
        network: Shared neural network
        config: Configuration object
        device: Device to run on (usually CPU)
        games_to_play: Number of games this worker should play
        result_queue: Queue to send game results [(states, policies, values)]
        vis_queue: Queue to send visualization updates (only used by rank 0)
    """
    # Initialize game generator
    # Note: Each worker gets its own MCTS instance but shares the network weights
    self_play = SelfPlayGame(network, config, device)
    
    for _ in range(games_to_play):
        # Setup visualization callback if we are the visualization worker (rank 0)
        on_move = None
        if rank == 0 and vis_queue is not None:
            def on_move_callback(game, policy, move_count, current_player):
                # Only send update if we should render (simple throttle check could be here too)
                # But actual throttle is on consumer side. We send data here.
                # To minimize IPC overhead, we can do a quick check or just send all 
                # and let main process throttle. For now, send all.
                try:
                    # Convert to absolute board for visualization
                    board = game.to_absolute_board_array()
                    vis_queue.put_nowait((board, policy, move_count, current_player))
                except Exception:
                    pass  # Ignore queue full errors
            on_move = on_move_callback

        # Play game
        states, policies, values = self_play.play_game(on_move=on_move)
        
        # specific check for empty game (should not happen but good for safety)
        if len(states) > 0:
            result_queue.put((states, policies, values))


def play_games_parallel(
    network: CheckersNetwork,
    config,
    device: torch.device,
    num_games: int,
    game_visualizer=None
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    """
    Play games in parallel using multiple processes.
    
    Args:
        network: Neural network (must be in shared memory)
        config: Configuration
        device: Device (should be CPU for parallel workers)
        num_games: Total games to play
        game_visualizer: Visualization instance (for main thread rendering)
        
    Returns:
        (states, policies, values): Aggregated training examples
    """
    num_workers = config.NUM_WORKERS
    games_per_worker = num_games // num_workers
    remainder = num_games % num_workers
    
    # Establish queues
    result_queue = mp.Queue()
    vis_queue = mp.Queue() if game_visualizer else None
    
    processes = []
    
    # Start workers
    print(f"  Starting {num_workers} workers...")
    for rank in range(num_workers):
        # Distribute remainder games
        worker_games = games_per_worker + (1 if rank < remainder else 0)
        
        if worker_games == 0:
            continue
            
        p = mp.Process(
            target=self_play_worker,
            args=(
                rank, 
                network, 
                config, 
                device, 
                worker_games, 
                result_queue, 
                vis_queue
            )
        )
        p.start()
        processes.append(p)
        
    # Collect results
    all_states = []
    all_policies = []
    all_values = []
    
    collected_games = 0
    
    # Progress bar setup
    bar_width = 40
    print("  ", end="", flush=True)
    
    while collected_games < num_games:
        # 1. Process visualization updates (Worker 0 -> Main Thread)
        if vis_queue and game_visualizer:
            try:
                # Drain queue up to latest frame
                latest_vis = None
                while True:
                    latest_vis = vis_queue.get_nowait()
            except Empty:
                pass
            
            if latest_vis and game_visualizer.should_render():
                 game_visualizer.render(*latest_vis)
        
        # 2. Process Game Results
        try:
            # Non-blocking check for results
            data = result_queue.get(timeout=0.05)
            states, policies, values = data
            
            all_states.extend(states)
            all_policies.extend(policies)
            all_values.extend(values)
            
            collected_games += 1
            
            # Update progress
            progress = collected_games / num_games
            filled = int(bar_width * progress)
            bar = '█' * filled + '░' * (bar_width - filled)
            print(f"\r  [{bar}] {collected_games}/{num_games} games | "
                  f"{len(all_states)} examples", end="", flush=True)
                  
        except Empty:
            # If no results yet, check if processes are alive
            if not any(p.is_alive() for p in processes) and result_queue.empty():
                print("\n  Warning: All workers died prematurely.")
                break
            continue

    print() # Newline
    
    # Clean up
    for p in processes:
        p.join()
        
    return all_states, all_policies, all_values


# Testing
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from config8x8 import Config

    print("Testing self-play...")

    # Setup
    device = Config.get_selfplay_device()
    network = CheckersNetwork(
        num_filters=Config.NUM_FILTERS,
        num_res_blocks=Config.NUM_RES_BLOCKS,
        policy_size=Config.POLICY_SIZE
    )
    network.to(device)
    network.eval()

    # Play a single game
    print("\nPlaying single game with MCTS...")
    self_play = SelfPlayGame(network, Config, device)

    states, policies, values = self_play.play_game()

    print(f"\nGame statistics:")
    print(f"  Moves: {len(states)}")
    print(f"  Outcome: {values[0] if values else 'N/A'}")
    print(f"  State shape: {states[0].shape if states else 'N/A'}")
    print(f"  Policy shape: {policies[0].shape if policies else 'N/A'}")
    print(f"  Value distribution: {np.bincount([int(v + 1) for v in values])}")

    # Play multiple games
    print("\nPlaying 3 games...")
    all_states, all_policies, all_values = play_games_sequential(
        network, Config, device, num_games=3
    )

    print(f"\nTotal training examples: {len(all_states)}")
    print(f"Average game length: {len(all_states) / 3:.1f} moves")

    print("\n✓ Self-play tests passed!")
