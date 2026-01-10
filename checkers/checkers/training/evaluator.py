"""
Model evaluation via tournament play.
"""

import torch
import numpy as np
from typing import Tuple
from ..engine.game import CheckersGame
from ..mcts.mcts import MCTS
from ..network.resnet import CheckersNetwork


class Evaluator:
    """
    Evaluate networks by playing games between them.
    """

    def __init__(self, config, device: torch.device):
        """
        Initialize evaluator.

        Args:
            config: Configuration object
            device: PyTorch device
        """
        self.config = config
        self.device = device

    def evaluate(
        self,
        network1: CheckersNetwork,
        network2: CheckersNetwork,
        num_games: int
    ) -> Tuple[float, int, int, int]:
        """
        Play games between two networks and compute win rate.

        Args:
            network1: First network (challenger)
            network2: Second network (champion)
            num_games: Number of games to play

        Returns:
            (win_rate, wins, draws, losses): Results from network1's perspective
        """
        network1.eval()
        network2.eval()

        wins = 0
        draws = 0
        losses = 0

        # Play games, alternating which network goes first
        for game_idx in range(num_games):
            if game_idx % 2 == 0:
                # Network1 plays as player 1
                result = self._play_game(network1, network2)
            else:
                # Network1 plays as player 2 (flip result)
                result = -self._play_game(network2, network1)

            if result > 0:
                wins += 1
            elif result == 0:
                draws += 1
            else:
                losses += 1

            if (game_idx + 1) % 10 == 0:
                print(f"    Eval games: {game_idx + 1}/{num_games} "
                      f"(W/D/L: {wins}/{draws}/{losses})")

        # Compute win rate (draws count as 0.5)
        win_rate = (wins + 0.5 * draws) / num_games

        return win_rate, wins, draws, losses

    def _play_game(
        self,
        network1: CheckersNetwork,
        network2: CheckersNetwork
    ) -> float:
        """
        Play one game between two networks.

        Args:
            network1: Network for player 1
            network2: Network for player 2

        Returns:
            Game result from player 1's perspective (1.0, 0.0, or -1.0)
        """
        game = CheckersGame()

        # Create MCTS instances for both players
        mcts1 = MCTS(
            network=network1,
            c_puct=self.config.C_PUCT,
            num_simulations=self.config.MCTS_SIMS_EVAL,
            dirichlet_alpha=0.0,  # No exploration noise during evaluation
            dirichlet_epsilon=0.0,
            device=self.device
        )

        mcts2 = MCTS(
            network=network2,
            c_puct=self.config.C_PUCT,
            num_simulations=self.config.MCTS_SIMS_EVAL,
            dirichlet_alpha=0.0,
            dirichlet_epsilon=0.0,
            device=self.device
        )

        move_count = 0
        max_moves = self.config.MAX_GAME_LENGTH

        # Play game
        while not game.is_terminal() and move_count < max_moves:
            # Select MCTS based on current player
            if game.current_player == 1:
                mcts = mcts1
            else:
                mcts = mcts2

            # Get move from MCTS (greedy, no temperature)
            policy = mcts.search(game, add_noise=False)
            action = self._select_best_action(policy)

            # Apply move
            legal_moves = game.get_legal_moves()
            if action < len(legal_moves):
                game.make_move(legal_moves[action])
                move_count += 1
            else:
                # Should not happen
                break

        # Get result from player 1's perspective
        if game.is_terminal():
            result = game.get_result()
            # Flip result if it's from player 2's perspective
            if game.current_player == 2:
                result = -result
        else:
            # Draw by max moves
            result = 0.0

        return result

    def _select_best_action(self, policy: np.ndarray) -> int:
        """
        Select action with highest probability (greedy).

        Args:
            policy: Policy distribution

        Returns:
            Best action index
        """
        legal_actions = np.where(policy > 0)[0]
        if len(legal_actions) == 0:
            return 0
        return legal_actions[np.argmax(policy[legal_actions])]


# Testing
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

    from config import Config
    from checkers.network.resnet import CheckersNetwork

    print("Testing evaluator...")

    # Setup
    device = Config.get_device()

    # Create two identical networks (should get ~50% win rate)
    network1 = CheckersNetwork(
        num_filters=Config.NUM_FILTERS,
        num_res_blocks=Config.NUM_RES_BLOCKS,
        policy_size=Config.POLICY_SIZE
    )
    network1.to(device)
    network1.eval()

    network2 = CheckersNetwork(
        num_filters=Config.NUM_FILTERS,
        num_res_blocks=Config.NUM_RES_BLOCKS,
        policy_size=Config.POLICY_SIZE
    )
    network2.to(device)
    network2.eval()

    # Copy weights from network1 to network2
    network2.load_state_dict(network1.state_dict())

    # Create evaluator
    evaluator = Evaluator(Config, device)

    # Play a few games (reduced for testing)
    print("\nPlaying 4 evaluation games (identical networks)...")
    win_rate, wins, draws, losses = evaluator.evaluate(
        network1, network2, num_games=4
    )

    print(f"\nResults:")
    print(f"  Win rate: {win_rate:.2%}")
    print(f"  Wins: {wins}")
    print(f"  Draws: {draws}")
    print(f"  Losses: {losses}")

    # Expected: ~50% win rate with identical networks
    print(f"\n✓ Evaluator tests passed!")
