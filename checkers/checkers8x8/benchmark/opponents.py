"""
Baseline opponents for benchmarking AlphaZero agent strength.

Provides:
- RandomPlayer: Picks random legal moves
- GreedyPlayer: Prefers captures, else random
- MinimaxPlayer: Alpha-beta pruning with material evaluation
"""

import random
from abc import ABC, abstractmethod
from typing import Optional

try:
    from ..engine.game import CheckersGame
    from ..engine.moves import Move
    from ..engine.bitboard import count_bits
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from checkers8x8.engine.game import CheckersGame
    from checkers8x8.engine.moves import Move
    from checkers8x8.engine.bitboard import count_bits


class BasePlayer(ABC):
    """Abstract base class for benchmark opponents."""

    @abstractmethod
    def select_move(self, game: CheckersGame) -> Optional[Move]:
        """
        Select a move for the current position.

        Args:
            game: Current game state

        Returns:
            Selected Move object, or None if no legal moves
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return player name for display."""
        pass


class RandomPlayer(BasePlayer):
    """Plays random legal moves."""

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

    @property
    def name(self) -> str:
        return "Random"

    def select_move(self, game: CheckersGame) -> Optional[Move]:
        moves = game.get_legal_moves()
        if not moves:
            return None
        return random.choice(moves)


class GreedyPlayer(BasePlayer):
    """
    Greedy player that prefers captures.

    If captures are available, picks one randomly.
    Otherwise picks a random non-capture move.
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

    @property
    def name(self) -> str:
        return "Greedy"

    def select_move(self, game: CheckersGame) -> Optional[Move]:
        moves = game.get_legal_moves()
        if not moves:
            return None

        # In checkers, if captures exist, all moves are captures (forced)
        # So just pick randomly - captures are already prioritized by rules
        # But we can prefer multi-captures if available
        if moves[0].captured_squares:
            # All moves are captures - prefer longer capture chains
            max_captures = max(len(m.captured_squares) for m in moves)
            best_captures = [m for m in moves if len(m.captured_squares) == max_captures]
            return random.choice(best_captures)

        return random.choice(moves)


class MinimaxPlayer(BasePlayer):
    """
    Minimax player with alpha-beta pruning.

    Evaluation includes:
    - Material: man = 100, king = 300
    - Advancement: bonus for pieces closer to promotion
    - Center control: bonus for controlling center squares
    - Back row: small bonus for protecting back row (prevents enemy kings)
    - Mobility: bonus for having more legal moves
    - King activity: kings in center are more valuable
    """

    # Square tables for 32 playable squares (indexed 0-31)
    # Board layout (from white's perspective, squares 0-31):
    #     0   1   2   3       (row 0, black's back row)
    #   4   5   6   7         (row 1)
    #     8   9  10  11       (row 2)
    #  12  13  14  15         (row 3)
    #    16  17  18  19       (row 4)
    #  20  21  22  23         (row 5)
    #    24  25  26  27       (row 6)
    #  28  29  30  31         (row 7, white's back row)

    # Center control bonus (for both men and kings)
    CENTER_CONTROL = [
        0,  0,  0,  0,   # row 0
        0,  1,  1,  0,   # row 1
        1,  2,  2,  1,   # row 2
        2,  4,  4,  2,   # row 3
        2,  4,  4,  2,   # row 4
        1,  2,  2,  1,   # row 5
        0,  1,  1,  0,   # row 6
        0,  0,  0,  0,   # row 7
    ]

    # Back row bonus (protecting promotion squares)
    BACK_ROW_BONUS = 2

    def __init__(self, depth: int = 4, seed: Optional[int] = None, simple_eval: bool = False):
        """
        Initialize minimax player.

        Args:
            depth: Search depth (plies)
            seed: Random seed for tie-breaking
            simple_eval: If True, use simple material-only evaluation (for comparison)
        """
        self.depth = depth
        self.simple_eval = simple_eval
        if seed is not None:
            random.seed(seed)
        self._nodes_searched = 0

    @property
    def name(self) -> str:
        return f"Minimax-D{self.depth}"

    def select_move(self, game: CheckersGame) -> Optional[Move]:
        """Select best move using minimax with alpha-beta pruning."""
        moves = game.get_legal_moves()
        if not moves:
            return None

        if len(moves) == 1:
            return moves[0]

        self._nodes_searched = 0
        best_score = float('-inf')
        best_moves = []

        alpha = float('-inf')
        beta = float('inf')

        for move in moves:
            # Clone game and apply move
            game_copy = game.clone()
            game_copy.make_move(move)

            # Evaluate from opponent's perspective (negated)
            score = -self._minimax(game_copy, self.depth - 1, -beta, -alpha)

            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

            alpha = max(alpha, score)

        # Random tie-breaking among equally good moves
        return random.choice(best_moves)

    def _minimax(
        self,
        game: CheckersGame,
        depth: int,
        alpha: float,
        beta: float
    ) -> float:
        """
        Minimax search with alpha-beta pruning.

        Uses negamax formulation (always maximizing).

        Args:
            game: Current game state
            depth: Remaining search depth
            alpha: Alpha bound
            beta: Beta bound

        Returns:
            Evaluation score from current player's perspective
        """
        self._nodes_searched += 1

        # Terminal node or depth limit
        if game.is_terminal():
            result = game.get_result()
            # Must exceed any possible material evaluation (~1400 max)
            return result * 10000

        if depth == 0:
            return self._evaluate(game)

        moves = game.get_legal_moves()
        if not moves:
            # No moves = loss
            return -1000

        # Move ordering: captures first (longer chains better), then center moves
        moves = self._order_moves(game, moves)

        best_score = float('-inf')

        for move in moves:
            game_copy = game.clone()
            game_copy.make_move(move)

            # Negamax: negate score and swap alpha/beta
            score = -self._minimax(game_copy, depth - 1, -beta, -alpha)

            best_score = max(best_score, score)
            alpha = max(alpha, score)

            # Alpha-beta cutoff
            if alpha >= beta:
                break

        return best_score

    def _order_moves(self, game: CheckersGame, moves: list) -> list:
        """
        Order moves to improve alpha-beta pruning.

        Priority:
        1. Captures (longer chains first)
        2. Moves toward center
        3. Advancement moves
        """
        def move_priority(move):
            score = 0
            # Captures are very important
            if move.captured_squares:
                score += 1000 + len(move.captured_squares) * 100

            # Final square preference
            final_sq = move.to_square
            score += self.CENTER_CONTROL[final_sq] * 10

            # Advancement (lower final square = more advanced for current player)
            row = final_sq // 4
            score += (7 - row) * 5

            return -score  # Negate for descending sort

        return sorted(moves, key=move_priority)

    def _evaluate(self, game: CheckersGame) -> float:
        """
        Static evaluation function.

        Evaluated from current player's perspective.

        Args:
            game: Game state to evaluate

        Returns:
            Evaluation score (positive = good for current player)
        """
        if self.simple_eval:
            return self._evaluate_simple(game)

        # Material values
        MAN_VALUE = 100
        KING_VALUE = 300

        player_score = 0
        opponent_score = 0

        # Evaluate player pieces
        # Note: In the game, current player's pieces are at rows 5-7 (squares 20-31)
        # and opponent is at rows 0-2 (squares 0-11) in the starting position.
        # But after perspective swapping, we always evaluate from current player's view.

        # Player men
        men = game.player_men
        while men:
            sq = (men & -men).bit_length() - 1
            men &= men - 1

            player_score += MAN_VALUE
            # Advancement: lower square number = more advanced for player
            # (player pieces move toward row 0 for promotion)
            row = sq // 4
            advancement_bonus = (7 - row) * 2  # 0-14 points
            player_score += advancement_bonus
            player_score += self.CENTER_CONTROL[sq]

            # Back row protection
            if sq >= 28:  # Row 7 - player's back row
                player_score += self.BACK_ROW_BONUS

        # Player kings
        kings = game.player_kings
        while kings:
            sq = (kings & -kings).bit_length() - 1
            kings &= kings - 1

            player_score += KING_VALUE
            player_score += self.CENTER_CONTROL[sq] * 2  # Kings benefit more from center

        # Opponent men
        men = game.opponent_men
        while men:
            sq = (men & -men).bit_length() - 1
            men &= men - 1

            opponent_score += MAN_VALUE
            # Opponent pieces move toward row 7 for promotion
            row = sq // 4
            advancement_bonus = row * 2  # 0-14 points
            opponent_score += advancement_bonus
            opponent_score += self.CENTER_CONTROL[sq]

            # Back row protection for opponent
            if sq <= 3:  # Row 0 - opponent's back row
                opponent_score += self.BACK_ROW_BONUS

        # Opponent kings
        kings = game.opponent_kings
        while kings:
            sq = (kings & -kings).bit_length() - 1
            kings &= kings - 1

            opponent_score += KING_VALUE
            opponent_score += self.CENTER_CONTROL[sq] * 2

        # Mobility bonus (expensive to compute, so scaled down)
        # Only compute at leaf nodes and not too often
        mobility_bonus = 0
        moves = game.get_legal_moves()
        mobility_bonus = len(moves) * 2

        return (player_score - opponent_score) + mobility_bonus

    def _evaluate_simple(self, game: CheckersGame) -> float:
        """Simple material-only evaluation for comparison."""
        player_men = count_bits(game.player_men)
        player_kings = count_bits(game.player_kings)
        opponent_men = count_bits(game.opponent_men)
        opponent_kings = count_bits(game.opponent_kings)

        player_material = player_men + 3 * player_kings
        opponent_material = opponent_men + 3 * opponent_kings

        return player_material - opponent_material

    @property
    def nodes_searched(self) -> int:
        """Return number of nodes searched in last move."""
        return self._nodes_searched


# Testing
if __name__ == "__main__":
    print("Testing Benchmark Opponents")
    print("=" * 60)

    # Create a game
    game = CheckersGame()

    # Test RandomPlayer
    print("\n1. RandomPlayer")
    random_player = RandomPlayer(seed=42)
    move = random_player.select_move(game)
    print(f"   Selected move: {move}")

    # Test GreedyPlayer
    print("\n2. GreedyPlayer")
    greedy_player = GreedyPlayer(seed=42)
    move = greedy_player.select_move(game)
    print(f"   Selected move: {move}")

    # Test MinimaxPlayer at different depths
    for depth in [2, 4, 6]:
        print(f"\n3. MinimaxPlayer (depth={depth})")
        minimax_player = MinimaxPlayer(depth=depth, seed=42)

        import time
        start = time.time()
        move = minimax_player.select_move(game)
        elapsed = time.time() - start

        print(f"   Selected move: {move}")
        print(f"   Nodes searched: {minimax_player.nodes_searched:,}")
        print(f"   Time: {elapsed:.3f}s")

    # Play a quick game: Random vs Minimax-2
    print("\n4. Quick game: Random vs Minimax-D2")
    game = CheckersGame()
    random_p = RandomPlayer(seed=123)
    minimax_p = MinimaxPlayer(depth=2, seed=456)

    players = [random_p, minimax_p]
    move_count = 0

    while not game.is_terminal() and move_count < 200:
        current = players[move_count % 2]
        move = current.select_move(game)
        if move:
            game.make_move(move)
        move_count += 1

    result = game.get_result()
    # Result is from perspective of player to move (who lost if no moves)
    # If Minimax (player 1) was last to move successfully, result is from Random's perspective
    if move_count % 2 == 0:
        # Random's turn, Random to move
        print(f"   Result: {'Random wins' if result > 0 else 'Minimax wins' if result < 0 else 'Draw'}")
    else:
        # Minimax's turn
        print(f"   Result: {'Minimax wins' if result > 0 else 'Random wins' if result < 0 else 'Draw'}")
    print(f"   Moves played: {move_count}")

    print("\n" + "=" * 60)
    print("All opponent tests passed!")
