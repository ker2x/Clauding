#!/usr/bin/env python3
"""
Benchmark script for evaluating AlphaZero agent against baseline opponents.

Measures absolute playing strength by playing against:
- Minimax-D2, D4, D6, D8: Alpha-beta with positional evaluation (default)
- Minimax-D10: Optional deeper search (slow, use --include-d10)
- Random: Picks random legal moves (disabled by default)
- Greedy: Prefers captures (disabled by default)

Performance optimizations:
- Uses batched NN evaluation across multiple games (requires C++ extension)
- Default 50 MCTS sims (vs 200 for training) - faster, still accurate
- Use --sequential to disable batching

Usage:
    python scripts8x8/benchmark.py
    python scripts8x8/benchmark.py --model checkpoints8x8/best_model.pt
    python scripts8x8/benchmark.py --games 50 --skip-minimax-6
    python scripts8x8/benchmark.py --include-random --include-greedy
    python scripts8x8/benchmark.py --mcts-sims 100  # More accurate
    python scripts8x8/benchmark.py --sequential     # Non-batched mode
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

try:
    import checkers_cpp
    HAS_CPP = True
except ImportError:
    HAS_CPP = False


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
    max_moves: int = 400,
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
    moves_wins = []
    moves_losses = []
    moves_draws = []

    for game_idx in range(num_games):
        # Alternate who plays first
        mcts_plays_first = (game_idx % 2 == 0)

        result, num_moves = play_game(
            mcts_player, opponent, mcts_plays_first, verbose=verbose
        )
        total_moves += num_moves

        if result == 1.0:
            wins += 1
            moves_wins.append(num_moves)
        elif result == 0.0:
            losses += 1
            moves_losses.append(num_moves)
        else:
            draws += 1
            moves_draws.append(num_moves)

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
        'avg_moves': total_moves / num_games,
        'avg_moves_win': sum(moves_wins) / len(moves_wins) if moves_wins else None,
        'avg_moves_loss': sum(moves_losses) / len(moves_losses) if moves_losses else None,
        'avg_moves_draw': sum(moves_draws) / len(moves_draws) if moves_draws else None,
    }


def benchmark_opponent_batched(
    network: CheckersNetwork,
    config,
    device: torch.device,
    opponent_factory,
    num_games: int,
    num_simulations: int,
    verbose: bool = False
) -> Dict:
    """
    Run batched benchmark games against one opponent type.

    Runs multiple games in parallel with batched NN evaluation.
    Much faster on MPS/GPU than sequential games.

    Args:
        network: Neural network
        config: Config object
        device: PyTorch device
        opponent_factory: Callable that creates opponent instances
        num_games: Number of games to play
        num_simulations: MCTS simulations per move
        verbose: Print progress

    Returns:
        Dictionary with wins, losses, draws, win_rate
    """
    if not HAS_CPP:
        # Fall back to sequential
        mcts_player = MCTSPlayer(network, config, device, num_simulations)
        return benchmark_opponent(mcts_player, opponent_factory(), num_games, verbose)

    # Initialize games and MCTS instances
    games = []
    mcts_instances = []
    opponents = []
    mcts_plays_first = []
    move_counts = []
    finished = []

    for i in range(num_games):
        game = checkers_cpp.Game()
        games.append(game)

        mcts = checkers_cpp.MCTS(
            config.C_PUCT,
            num_simulations,
            config.DIRICHLET_ALPHA,
            0.0  # No noise for evaluation
        )
        mcts_instances.append(mcts)
        opponents.append(opponent_factory())
        mcts_plays_first.append(i % 2 == 0)
        move_counts.append(0)
        finished.append(False)

    max_moves = 400
    opponent_name = opponents[0].name
    prev_finished = 0

    # Play all games simultaneously
    with torch.no_grad():
        while not all(finished):
            # Collect games where it's MCTS's turn
            mcts_turn_games = []
            for i in range(num_games):
                if finished[i]:
                    continue

                game = games[i]
                if game.is_terminal() or move_counts[i] >= max_moves:
                    finished[i] = True
                    continue

                is_mcts_turn = (move_counts[i] % 2 == 0) == mcts_plays_first[i]

                if is_mcts_turn:
                    mcts_turn_games.append(i)
                else:
                    # Opponent's turn - make move immediately
                    # Convert C++ game to Python for opponent
                    py_game = CheckersGame()
                    py_game.player_men = game.player_men
                    py_game.player_kings = game.player_kings
                    py_game.opponent_men = game.opponent_men
                    py_game.opponent_kings = game.opponent_kings
                    py_game.current_player = game.current_player
                    py_game.move_count = game.move_count

                    move = opponents[i].select_move(py_game)
                    if move is None:
                        finished[i] = True
                        continue

                    # Apply to C++ game via action
                    action = move.to_actions()[0]
                    if not game.make_action(action):
                        finished[i] = True
                        continue

                    move_counts[i] += 1

            if not mcts_turn_games:
                continue

            # Start MCTS for all MCTS-turn games
            for i in mcts_turn_games:
                if not finished[i]:
                    mcts_instances[i].start_search(games[i])

            # Batched MCTS loop
            active_mcts = [(i, mcts_instances[i]) for i in mcts_turn_games
                          if not finished[i] and not mcts_instances[i].is_finished()]

            while active_mcts:
                all_inputs = []
                batch_map = []  # (game_idx, batch_id)

                for game_idx, mcts in active_mcts:
                    batch_id, inputs = mcts.find_leaves(16)
                    if batch_id == -1:
                        continue

                    for inp in inputs:
                        all_inputs.append(inp)
                        batch_map.append((game_idx, batch_id))

                if not all_inputs:
                    break

                # Single batched NN evaluation
                input_tensor = torch.tensor(all_inputs, dtype=torch.float32, device=device)
                input_tensor = input_tensor.view(-1, 8, 8, 8)

                policy_logits, values = network(input_tensor)
                policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()
                values_np = values.cpu().numpy().flatten()

                # Group results by batch
                results_by_batch = {}
                for idx, (game_idx, batch_id) in enumerate(batch_map):
                    key = (game_idx, batch_id)
                    if key not in results_by_batch:
                        results_by_batch[key] = ([], [])
                    results_by_batch[key][0].append(policy_probs[idx].tolist())
                    results_by_batch[key][1].append(values_np[idx])

                # Process results
                for (game_idx, batch_id), (policies, vals) in results_by_batch.items():
                    mcts_instances[game_idx].process_results(batch_id, policies, vals)

                # Update active list
                active_mcts = [(i, mcts_instances[i]) for i in mcts_turn_games
                              if not finished[i] and not mcts_instances[i].is_finished()]

            # Make MCTS moves
            for i in mcts_turn_games:
                if finished[i]:
                    continue

                policy = mcts_instances[i].get_policy(0.0)  # Greedy
                action = int(np.argmax(policy))

                if action == -1 or not games[i].make_action(action):
                    finished[i] = True
                    continue

                move_counts[i] += 1

            # Progress (only print when something finishes)
            cur_finished = sum(finished)
            if cur_finished > prev_finished:
                avg_moves = sum(move_counts) / max(1, cur_finished)
                print(f"\r    {cur_finished}/{num_games} games (avg {avg_moves:.0f} moves)     ", end='', flush=True)
                prev_finished = cur_finished

    print()

    # Count results
    wins = 0
    losses = 0
    draws = 0
    total_moves = 0
    moves_wins = []
    moves_losses = []
    moves_draws = []

    for i in range(num_games):
        game = games[i]
        result = game.get_result()

        # Convert to MCTS perspective
        mcts_to_move = (move_counts[i] % 2 == 0) == mcts_plays_first[i]
        if not mcts_to_move:
            result = -result

        total_moves += move_counts[i]

        if result > 0:
            wins += 1
            moves_wins.append(move_counts[i])
        elif result < 0:
            losses += 1
            moves_losses.append(move_counts[i])
        else:
            draws += 1
            moves_draws.append(move_counts[i])

    win_rate = (wins + 0.5 * draws) / num_games

    return {
        'opponent': opponent_name,
        'games': num_games,
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'win_rate': win_rate,
        'avg_moves': total_moves / num_games,
        'avg_moves_win': sum(moves_wins) / len(moves_wins) if moves_wins else None,
        'avg_moves_loss': sum(moves_losses) / len(moves_losses) if moves_losses else None,
        'avg_moves_draw': sum(moves_draws) / len(moves_draws) if moves_draws else None,
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
        'Minimax-D8': 1700,
        'Minimax-D10': 1900,
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
    elif elo < 2100:
        return "master-level"
    else:
        return "grandmaster-level"


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
                'opponent', 'games', 'wins', 'losses', 'draws', 'win_rate',
                'avg_moves', 'avg_moves_win', 'avg_moves_loss', 'avg_moves_draw'
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
                f"{r['win_rate']:.3f}",
                f"{r['avg_moves']:.1f}",
                f"{r['avg_moves_win']:.1f}" if r['avg_moves_win'] is not None else '',
                f"{r['avg_moves_loss']:.1f}" if r['avg_moves_loss'] is not None else '',
                f"{r['avg_moves_draw']:.1f}" if r['avg_moves_draw'] is not None else '',
            ])

    print(f"\nResults saved to {log_path}")


def format_avg_moves(value) -> str:
    """Format average moves value, handling None."""
    return f"{value:>5.1f}" if value is not None else "    -"


def print_results_table(results: List[Dict], model_path: str, elo: int):
    """Print results in nice table format."""
    print("\n" + "=" * 100)
    print(f"Benchmark Results: {model_path}")
    print("=" * 100)
    print()

    # Table header
    print(f"{'Opponent':<12} | {'Games':>5} | {'W':>3} | {'L':>3} | {'D':>3} | "
          f"{'Win%':>6} | {'AvgLen':>6} | {'W Len':>5} | {'L Len':>5} | {'D Len':>5}")
    print("-" * 100)

    for r in results:
        print(f"{r['opponent']:<12} | {r['games']:>5} | {r['wins']:>3} | {r['losses']:>3} | "
              f"{r['draws']:>3} | {r['win_rate']*100:>5.1f}% | {r['avg_moves']:>6.1f} | "
              f"{format_avg_moves(r['avg_moves_win'])} | {format_avg_moves(r['avg_moves_loss'])} | "
              f"{format_avg_moves(r['avg_moves_draw'])}")

    print("-" * 100)
    print()
    print(f"Estimated strength: ~{elo} ELO ({get_strength_description(elo)})")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark AlphaZero agent against baseline opponents'
    )
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model checkpoint (default: checkpoints8x8/best_model.pt)')
    parser.add_argument('--games', type=int, default=10,
                       help='Number of games per opponent (default: 10)')
    parser.add_argument('--mcts-sims', type=int, default=50,
                       help='MCTS simulations per move (default: 50)')
    parser.add_argument('--skip-minimax-6', action='store_true',
                       help='Skip Minimax depth-6')
    parser.add_argument('--skip-minimax-4', action='store_true',
                       help='Skip Minimax depth-4')
    parser.add_argument('--skip-minimax-8', action='store_true',
                       help='Skip Minimax depth-8')
    parser.add_argument('--include-d10', action='store_true',
                       help='Include Minimax depth-10 (very slow)')
    parser.add_argument('--include-random', action='store_true',
                       help='Include Random opponent (disabled by default)')
    parser.add_argument('--include-greedy', action='store_true',
                       help='Include Greedy opponent (disabled by default)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: fewer games, skip slow opponents')
    parser.add_argument('--sequential', action='store_true',
                       help='Use sequential (non-batched) benchmark (slower)')
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
        args.skip_minimax_8 = True

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
        # CPU: Set thread count from config
        torch.set_num_threads(Config.NUM_THREADS)
        device = torch.device('cpu')
    print(f"Using device: {device}")
    if device.type == 'cpu':
        print(f"PyTorch threads: {torch.get_num_threads()}")

    # Load model
    network, iteration = load_model(args.model, Config, device)

    print(f"MCTS simulations: {args.mcts_sims}")
    use_batched = HAS_CPP and not args.sequential
    print(f"Mode: {'batched' if use_batched else 'sequential'}")

    # Create MCTS player (for sequential mode)
    mcts_player = None
    if args.sequential or not HAS_CPP:
        mcts_player = MCTSPlayer(
            network=network,
            config=Config,
            device=device,
            num_simulations=args.mcts_sims
        )

    # Create opponent factories (Random and Greedy disabled by default)
    # Factory pattern needed for batched mode (each game needs its own instance)
    opponent_configs = []

    if args.include_random:
        opponent_configs.append((
            lambda s=args.seed: RandomPlayer(seed=s),
            "Random",
            args.games
        ))

    if args.include_greedy:
        opponent_configs.append((
            lambda s=args.seed: GreedyPlayer(seed=s),
            "Greedy",
            args.games
        ))

    opponent_configs.append((
        lambda s=args.seed: MinimaxPlayer(depth=2, seed=s),
        "Minimax-D2",
        args.games
    ))

    if not args.skip_minimax_4:
        opponent_configs.append((
            lambda s=args.seed: MinimaxPlayer(depth=4, seed=s),
            "Minimax-D4",
            max(10, args.games // 2)
        ))

    if not args.skip_minimax_6:
        opponent_configs.append((
            lambda s=args.seed: MinimaxPlayer(depth=6, seed=s),
            "Minimax-D6",
            max(10, args.games // 2)
        ))

    if not args.skip_minimax_8:
        opponent_configs.append((
            lambda s=args.seed: MinimaxPlayer(depth=8, seed=s),
            "Minimax-D8",
            max(5, args.games // 4)
        ))

    if args.include_d10:
        opponent_configs.append((
            lambda s=args.seed: MinimaxPlayer(depth=10, seed=s),
            "Minimax-D10",
            max(3, args.games // 5)
        ))

    # Run benchmarks
    print("\n" + "=" * 70)
    print("Starting Benchmark")
    print("=" * 70)

    results = []
    total_start = time.time()

    for opponent_factory, opponent_name, num_games in opponent_configs:
        print(f"\nPlaying {num_games} games vs {opponent_name}...")
        start = time.time()

        if use_batched:
            result = benchmark_opponent_batched(
                network, Config, device, opponent_factory,
                num_games, args.mcts_sims, verbose=args.verbose
            )
        else:
            result = benchmark_opponent(
                mcts_player, opponent_factory(), num_games, verbose=args.verbose
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
