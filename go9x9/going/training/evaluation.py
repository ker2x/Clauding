"""
Evaluation module for model vs model games.
"""

import torch
import numpy as np
from typing import Dict

try:
    from ..engine.game import GoGame
    from ..network.resnet import GoNetwork
    from ..mcts.mcts import MCTS
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from going.engine.game import GoGame
    from going.network.resnet import GoNetwork
    from going.mcts.mcts import MCTS


def play_evaluation_game(
    model1: GoNetwork, model2: GoNetwork,
    config, device: torch.device, max_moves: int = 200,
    on_move=None
) -> float:
    """
    Play one evaluation game between two models.

    Returns:
        Result from model1's perspective: 1.0 (win), 0.0 (loss), 0.5 (draw)
    """
    game = GoGame()
    game.komi = config.KOMI

    mcts_instances = [
        MCTS(
            network=model,
            c_puct=config.C_PUCT,
            num_simulations=config.MCTS_SIMS_EVAL,
            dirichlet_alpha=config.DIRICHLET_ALPHA,
            dirichlet_epsilon=0.0,  # No noise for evaluation
            device=device,
            batch_size=getattr(config, 'MCTS_BATCH_SIZE', 1)
        )
        for model in [model1, model2]
    ]

    current_model_idx = 0  # 0 = model1 (Black), 1 = model2 (White)
    move_count = 0

    while not game.is_terminal() and move_count < max_moves:
        mcts = mcts_instances[current_model_idx]
        policy = mcts.search(game, add_noise=False)

        if on_move:
            on_move(game, policy, move_count, game.current_player)

        action = mcts.get_best_action()
        if action == -1:
            break

        if not game.make_action(action):
            break

        current_model_idx = 1 - current_model_idx
        move_count += 1

    # Get result from current player's perspective
    result = game.get_result()

    # Convert to model1's perspective
    # model1 = Black (player 1). If current_model_idx == 1, result is from model2's turn
    if current_model_idx == 1:
        result = -result

    if result > 0:
        return 1.0
    elif result < 0:
        return 0.0
    else:
        return 0.5


def evaluate_models(
    current_model: GoNetwork, best_model: GoNetwork,
    config, device: torch.device, num_games: int = None,
    on_move=None
) -> Dict[str, float]:
    if num_games is None:
        num_games = config.EVAL_GAMES

    current_model.eval()
    best_model.eval()

    wins = 0
    losses = 0
    draws = 0

    print(f"  Playing {num_games} evaluation games...")

    for game_idx in range(num_games):
        if game_idx % 2 == 0:
            result = play_evaluation_game(
                current_model, best_model, config, device, on_move=on_move
            )
        else:
            result = play_evaluation_game(
                best_model, current_model, config, device, on_move=on_move
            )
            result = 1.0 - result

        if result == 1.0:
            wins += 1
        elif result == 0.0:
            losses += 1
        else:
            draws += 1

        if (game_idx + 1) % 5 == 0 or (game_idx + 1) == num_games:
            print(f"    Games {game_idx + 1}/{num_games}: W{wins} L{losses} D{draws}")

    win_rate = (wins + 0.5 * draws) / num_games

    return {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'total_games': num_games,
        'win_rate': win_rate,
        'win_loss_diff': wins - losses,
    }
