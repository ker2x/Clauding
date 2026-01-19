#!/usr/bin/env python3
"""
Benchmark script for evaluating AlphaZero agent against baseline opponents.

Measures absolute playing strength by playing against:
- Random: Picks random legal moves
- Greedy: Prefers captures (picks longest chain)
- Minimax-D2, D4, D6: Alpha-beta with material evaluation

Usage:
    python scripts8x8/benchmark.py
    python scripts8x8/benchmark.py --model checkpoints8x8/best_model.pt
    python scripts8x8/benchmark.py --games 50 --skip-minimax-6
"""

import os
import sys
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np

from config8x8 import Config
from checkers8x8.engine.game import CheckersGame
from checkers8x8.network.resnet import CheckersNetwork
from checkers8x8.mcts.mcts import MCTS
from checkers8x8.benchmark.opponents import RandomPlayer, GreedyPlayer, MinimaxPlayer


class MCTSPlayer:
    """Wrapper for MCTS-based neural network player."""

    def __init__(
        self,
        network: CheckersNetwork,
        config,
        device: torch.device,
        num_simulations: Optional[int] = None
    ):
        self.network = network
        self.config = config
        self.device = device
        self.num_simulations = num_simulations or config.MCTS_SIMS_EVAL

        self.mcts = MCTS(
            network=network,
            c_puct=config.C_PUCT,
            num_simulations=self.num_simulations,
            dirichlet_alpha=config.DIRICHLET_ALPHA,
            dirichlet_epsilon=0.0,  # No exploration noise for evaluation
            device=device
        )

    @property
    def name(self) -> str:
        return f"MCTS-{self.num_simulations}"

    def select_action(self, game: CheckersGame) -> int:
        """Select best action using MCTS."""
        self.mcts.search(game, add_noise=False)
        return self.mcts.get_best_action()


def play_game(
    mcts_player: MCTSPlayer,
    opponent,
    mcts_plays_first: bool,
    max_moves: int = 200,
    verbose: bool = False
) -> Tuple[float, int]:
    """
    Play one game between MCTS player and opponent.

    Args:
        mcts_player: The neural network MCTS player
        opponent: Baseline opponent (Random, Greedy, or Minimax)
        mcts_plays_first: If True, MCTS plays as Player 1
        max_moves: Maximum moves before draw
        verbose: Print game progress

    Returns:
        (result, num_moves): result is 1.0 (MCTS wins), 0.0 (loss), 0.5 (draw)
    """
    game = CheckersGame()
    move_count = 0

    while not game.is_terminal() and move_count < max_moves:
        # Determine whose turn it is
        is_mcts_turn = (move_count % 2 == 0) == mcts_plays_first

        if is_mcts_turn:
            # MCTS player's turn
            action = mcts_player.select_action(game)
            if action == -1:
                break
            if not game.make_action(action):
                if verbose:
                    print(f"MCTS invalid action {action}")
                break
        else:
            # Opponent's turn
            move = opponent.select_move(game)
            if move is None:
                break
            game.make_move(move)

        move_count += 1

        if verbose and move_count % 20 == 0:
            print(f"  Move {move_count}...")

    # Get result
    result = game.get_result()

    # Result is from current player's perspective
    # We need to convert to MCTS player's perspective
    mcts_to_move = (move_count % 2 == 0) == mcts_plays_first

    if mcts_to_move:
        # MCTS is to move, result is from MCTS's perspective
        mcts_result = result
    else:
        # Opponent is to move, flip result
        mcts_result = -result

    # Convert to evaluation format
    if mcts_result > 0:
        return 1.0, move_count  # MCTS wins
    elif mcts_result < 0:
        return 0.0, move_count  # MCTS loses
    else:
        return 0.5, move_count  # Draw


def benchmark_opponent(
    mcts_player: MCTSPlayer,
    opponent,
    num_games: int,
    verbose: bool = False
) -> Dict:
    """
    Run benchmark games against one opponent.

    Args:
        mcts_player: Neural network player
        opponent: Baseline opponent
        num_games: Number of games to play
        verbose: Print progress

    Returns:
        Dictionary with wins, losses, draws, win_rate
    """
    wins = 0
    losses = 0
    draws = 0
    total_moves = 0

    for game_idx in range(num_games):
        # Alternate who plays first
        mcts_plays_first = (game_idx % 2 == 0)

        result, num_moves = play_game(
            mcts_player, opponent, mcts_plays_first, verbose=verbose
        )
        total_moves += num_moves

        if result == 1.0:
            wins += 1
        elif result == 0.0:
            losses += 1
        else:
            draws += 1

        # Progress indicator
        if (game_idx + 1) % 10 == 0 or (game_idx + 1) == num_games:
            print(f"    {game_idx + 1}/{num_games} games | "
                  f"W{wins} L{losses} D{draws}", end='\r')

    print()  # New line after progress

    win_rate = (wins + 0.5 * draws) / num_games

    return {
        'opponent': opponent.name,
        'games': num_games,
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'win_rate': win_rate,
        'avg_moves': total_moves / num_games
    }


def estimate_elo(results: List[Dict]) -> int:
    """
    Rough ELO estimate based on benchmark results.

    This is a very rough heuristic:
    - Random ~800 ELO
    - Greedy ~1000 ELO
    - Minimax-D2 ~1100 ELO
    - Minimax-D4 ~1300 ELO
    - Minimax-D6 ~1500 ELO

    Uses win rate against each to interpolate.
    """
    elo_benchmarks = {
        'Random': 800,
        'Greedy': 1000,
        'Minimax-D2': 1100,
        'Minimax-D4': 1300,
        'Minimax-D6': 1500,
    }

    total_weight = 0
    weighted_elo = 0

    for r in results:
        opponent = r['opponent']
        if opponent not in elo_benchmarks:
            continue

        base_elo = elo_benchmarks[opponent]
        win_rate = r['win_rate']

        # ELO formula: expected score = 1 / (1 + 10^((Rb - Ra)/400))
        # If we know expected score (win_rate), we can solve for Ra - Rb
        if win_rate >= 0.99:
            elo_diff = 400  # Cap at ~400 points advantage
        elif win_rate <= 0.01:
            elo_diff = -400
        else:
            # Solve: win_rate = 1 / (1 + 10^(-diff/400))
            # 1/win_rate - 1 = 10^(-diff/400)
            # -diff/400 = log10(1/win_rate - 1)
            # diff = -400 * log10(1/win_rate - 1)
            import math
            elo_diff = -400 * math.log10(1 / win_rate - 1)

        estimated_elo = base_elo + elo_diff
        weight = r['games']  # Weight by number of games

        weighted_elo += estimated_elo * weight
        total_weight += weight

    if total_weight == 0:
        return 1000  # Default

    return int(weighted_elo / total_weight)


def get_strength_description(elo: int) -> str:
    """Get human-readable description of playing strength."""
    if elo < 900:
        return "beginner"
    elif elo < 1100:
        return "novice"
    elif elo < 1300:
        return "intermediate"
    elif elo < 1500:
        return "strong amateur"
    elif elo < 1700:
        return "advanced"
    elif elo < 1900:
        return "expert"
    else:
        return "master-level"


def load_model(model_path: str, config, device: torch.device) -> Tuple[CheckersNetwork, int]:
    """
    Load model from checkpoint.

    Returns:
        (network, iteration): Loaded network and training iteration
    """
    print(f"Loading model from {model_path}...")

    network = CheckersNetwork(
        num_filters=config.NUM_FILTERS,
        num_res_blocks=config.NUM_RES_BLOCKS,
        policy_size=config.POLICY_SIZE
    )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    network.load_state_dict(checkpoint['network_state_dict'])
    network.to(device)
    network.eval()

    iteration = checkpoint.get('iteration', 0)

    print(f"  Loaded model from iteration {iteration}")
    return network, iteration


def save_results(results: List[Dict], iteration: int, elo: int, log_path: str):
    """Save benchmark results to CSV log."""
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Check if file exists to write header
    write_header = not log_file.exists()

    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow([
                'timestamp', 'iteration', 'estimated_elo',
                'opponent', 'games', 'wins', 'losses', 'draws', 'win_rate'
            ])

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        for r in results:
            writer.writerow([
                timestamp,
                iteration,
                elo,
                r['opponent'],
                r['games'],
                r['wins'],
                r['losses'],
                r['draws'],
                f"{r['win_rate']:.3f}"
            ])

    print(f"\nResults saved to {log_path}")


def print_results_table(results: List[Dict], model_path: str, elo: int):
    """Print results in nice table format."""
    print("\n" + "=" * 70)
    print(f"Benchmark Results: {model_path}")
    print("=" * 70)
    print()

    # Table header
    print(f"{'Opponent':<16} | {'Games':>5} | {'Wins':>4} | {'Losses':>6} | {'Draws':>5} | {'Win Rate':>8}")
    print("-" * 70)

    for r in results:
        print(f"{r['opponent']:<16} | {r['games']:>5} | {r['wins']:>4} | {r['losses']:>6} | "
              f"{r['draws']:>5} | {r['win_rate']*100:>7.1f}%")

    print("-" * 70)
    print()
    print(f"Estimated strength: ~{elo} ELO ({get_strength_description(elo)})")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark AlphaZero agent against baseline opponents'
    )
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model checkpoint (default: checkpoints8x8/best_model.pt)')
    parser.add_argument('--games', type=int, default=10,
                       help='Number of games per opponent (default: 10)')
    parser.add_argument('--mcts-sims', type=int, default=None,
                       help='MCTS simulations per move (default: from config)')
    parser.add_argument('--skip-minimax-6', action='store_true',
                       help='Skip Minimax depth-6 (slow)')
    parser.add_argument('--skip-minimax-4', action='store_true',
                       help='Skip Minimax depth-4')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: fewer games, skip slow opponents')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--log', type=str, default='logs8x8/benchmark_results.csv',
                       help='Path to save results CSV')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='mps',
                       choices=['cpu', 'mps', 'cuda'],
                       help='Device for neural network inference (default: mps)')

    args = parser.parse_args()

    # Set seed
    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Quick mode settings
    if args.quick:
        args.games = min(args.games, 10)
        args.skip_minimax_6 = True
        args.skip_minimax_4 = True

    # Find model
    if args.model is None:
        # Try to find best model
        best_path = Path('checkpoints8x8/best_model.pt')
        latest_path = Path('checkpoints8x8/latest.pt')

        if best_path.exists():
            args.model = str(best_path)
        elif latest_path.exists():
            args.model = str(latest_path)
        else:
            print("Error: No model found. Please specify --model or train first.")
            sys.exit(1)

    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)

    # Setup device
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load model
    network, iteration = load_model(args.model, Config, device)

    # Create MCTS player
    mcts_player = MCTSPlayer(
        network=network,
        config=Config,
        device=device,
        num_simulations=args.mcts_sims
    )
    print(f"MCTS player: {mcts_player.num_simulations} simulations/move")

    # Create opponents
    opponents = [
        (RandomPlayer(seed=args.seed), args.games),
        (GreedyPlayer(seed=args.seed), args.games),
        (MinimaxPlayer(depth=2, seed=args.seed), args.games),
    ]

    if not args.skip_minimax_4:
        opponents.append((MinimaxPlayer(depth=4, seed=args.seed), max(10, args.games // 2)))

    if not args.skip_minimax_6:
        opponents.append((MinimaxPlayer(depth=6, seed=args.seed), max(5, args.games // 4)))

    # Run benchmarks
    print("\n" + "=" * 70)
    print("Starting Benchmark")
    print("=" * 70)

    results = []
    total_start = time.time()

    for opponent, num_games in opponents:
        print(f"\nPlaying {num_games} games vs {opponent.name}...")
        start = time.time()

        result = benchmark_opponent(
            mcts_player, opponent, num_games, verbose=args.verbose
        )
        results.append(result)

        elapsed = time.time() - start
        print(f"  Completed in {elapsed:.1f}s ({elapsed/num_games:.1f}s/game)")

    total_time = time.time() - total_start
    print(f"\nTotal benchmark time: {total_time:.1f}s")

    # Calculate estimated ELO
    elo = estimate_elo(results)

    # Print results
    print_results_table(results, args.model, elo)

    # Save results
    save_results(results, iteration, elo, args.log)


if __name__ == "__main__":
    main()
