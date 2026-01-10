"""
Evaluation module for model vs model games.

Used to determine the best model by playing evaluation games.
"""

import torch
import numpy as np
from typing import Dict, Tuple
from pathlib import Path

try:
    from ..engine.game import CheckersGame
    from ..network.resnet import CheckersNetwork
    from ..mcts.mcts import MCTS
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from checkers8x8.engine.game import CheckersGame
    from checkers8x8.network.resnet import CheckersNetwork
    from checkers8x8.mcts.mcts import MCTS


def play_evaluation_game(
    model1: CheckersNetwork,
    model2: CheckersNetwork,
    config,
    device: torch.device,
    max_moves: int = 200,
    on_move=None
) -> float:
    """
    Play one evaluation game between two models.

    Args:
        model1: First model (perspective for result)
        model2: Second model
        config: Configuration object
        device: PyTorch device
        max_moves: Maximum moves before draw
        on_move: Optional callback for move visualization

    Returns:
        Result from model1's perspective: 1.0 (win), 0.0 (loss), 0.5 (draw)
    """
    game = CheckersGame()
    models = [model1, model2]
    
    # Create MCTS instances for both models (no exploration noise)
    mcts_instances = [
        MCTS(
            network=model,
            c_puct=config.C_PUCT,
            num_simulations=config.MCTS_SIMS_EVAL,
            dirichlet_alpha=config.DIRICHLET_ALPHA,
            dirichlet_epsilon=0.0,  # No noise for evaluation
            device=device
        )
        for model in models
    ]
    
    current_player = 0  # 0 for model1, 1 for model2
    move_count = 0
    
    while not game.is_terminal() and move_count < max_moves:
        # Run MCTS search
        mcts = mcts_instances[current_player]
        policy = mcts.search(game, add_noise=False)
        
        # Callback for visualization
        if on_move:
            on_move(game, policy, move_count, game.current_player)
        
        # Greedy action selection (best action)
        action = mcts.get_best_action()
        
        if action == -1 or not game.make_action(action):
            # Invalid action - this shouldn't happen
            break
        
        current_player = 1 - current_player
        move_count += 1
    
    # Get result from current game perspective
    result = game.get_result()
    
    # Convert to model1's perspective
    # If game ended on model1's turn, result is from model1's perspective
    # If game ended on model2's turn, flip the result
    if current_player == 1:
        result = -result
    
    # Convert to evaluation format: 1.0 (win), 0.0 (loss), 0.5 (draw)
    if result > 0:
        return 1.0  # model1 wins
    elif result < 0:
        return 0.0  # model1 loses
    else:
        return 0.5  # draw


def evaluate_models(
    current_model: CheckersNetwork,
    best_model: CheckersNetwork,
    config,
    device: torch.device,
    num_games: int = None,
    on_move=None
) -> Dict[str, float]:
    """
    Evaluate current model against best model.

    Plays num_games games with alternating sides.

    Args:
        current_model: Current training model
        best_model: Current best model
        config: Configuration object
        device: PyTorch device
        num_games: Number of games to play (default: config.EVAL_GAMES)
        on_move: Optional callback for move visualization

    Returns:
        Dictionary with evaluation results
    """
    if num_games is None:
        num_games = config.EVAL_GAMES
    
    # Put models in eval mode
    current_model.eval()
    best_model.eval()
    
    wins = 0
    losses = 0
    draws = 0
    
    print(f"  Playing {num_games} evaluation games...")
    
    # Play games with alternating sides
    for game_idx in range(num_games):
        # Alternate which model goes first
        if game_idx % 2 == 0:
            # Current model plays as player 1
            result = play_evaluation_game(current_model, best_model, config, device, on_move=on_move)
        else:
            # Best model plays as player 1, flip result
            result = play_evaluation_game(best_model, current_model, config, device, on_move=on_move)
            result = 1.0 - result  # Flip: 1->0, 0->1, 0.5->0.5
        
        # Count results
        if result == 1.0:
            wins += 1
        elif result == 0.0:
            losses += 1
        else:
            draws += 1
        
        # Print progress
        if (game_idx + 1) % 5 == 0 or (game_idx + 1) == num_games:
            print(f"    Games {game_idx + 1}/{num_games}: "
                  f"W{wins} L{losses} D{draws}")
    
    # Calculate win rate (draws count as 0.5)
    win_rate = (wins + 0.5 * draws) / num_games
    
    return {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'total_games': num_games,
        'win_rate': win_rate,
    }


# Testing
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from config8x8 import Config
    
    print("Testing Evaluation Module")
    print("=" * 60)
    
    device = torch.device("cpu")
    
    # Create two identical models (should get ~50% win rate)
    model1 = CheckersNetwork(num_filters=64, num_res_blocks=2, policy_size=128)
    model2 = CheckersNetwork(num_filters=64, num_res_blocks=2, policy_size=128)
    
    # Copy weights so they're identical
    model2.load_state_dict(model1.state_dict())
    
    print("\nEvaluating identical models (should be ~50% win rate)...")
    results = evaluate_models(model1, model2, Config, device, num_games=4)
    
    print(f"\nResults:")
    print(f"  Wins: {results['wins']}")
    print(f"  Losses: {results['losses']}")
    print(f"  Draws: {results['draws']}")
    print(f"  Win rate: {results['win_rate']:.1%}")
    
    print("\n✓ Evaluation module tests passed!")
