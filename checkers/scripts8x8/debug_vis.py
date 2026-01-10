
import numpy as np
import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import checkers_cpp
except ImportError:
    print("Failed to import checkers_cpp")
    sys.exit(1)

from checkers8x8.engine.bitboard import flip_bitboard, get_set_squares, square_to_row_col

def get_board_array_from_game(game) -> np.ndarray:
    """
    Convert game state to 8x8 array for visualization.
    Handles both Python CheckersGame and C++ Game objects.
    """
    if hasattr(game, 'to_absolute_board_array'):
        return game.to_absolute_board_array()
        
    # Reimplement logic for C++ object using bitboards
    
    board = np.zeros((8, 8), dtype=np.int8)
    
    if game.current_player == 1:
        p1_men, p1_kings = game.player_men, game.player_kings
        p2_men, p2_kings = game.opponent_men, game.opponent_kings
        p1_men = int(p1_men) if isinstance(p1_men, float) else p1_men # cast just in case
        needs_flip = False
    else:
        p2_men, p2_kings = game.player_men, game.player_kings
        p1_men, p1_kings = game.opponent_men, game.opponent_kings
        needs_flip = True
    
    # Cast all to int to be safe
    p1_men = int(p1_men)
    p1_kings = int(p1_kings)
    p2_men = int(p2_men)
    p2_kings = int(p2_kings)
    
    if needs_flip:
        p1_men = flip_bitboard(p1_men)
        p1_kings = flip_bitboard(p1_kings)
        p2_men = flip_bitboard(p2_men)
        p2_kings = flip_bitboard(p2_kings)
        
    for square in get_set_squares(p1_men):
        row, col = square_to_row_col(square)
        board[row, col] = 1
    for square in get_set_squares(p1_kings):
        row, col = square_to_row_col(square)
        board[row, col] = 2
    for square in get_set_squares(p2_men):
        row, col = square_to_row_col(square)
        board[row, col] = -1
    for square in get_set_squares(p2_kings):
        row, col = square_to_row_col(square)
        board[row, col] = -2
        
    return board

def render_board(board):
    scale = {0: '.', 1: 'r', 2: 'R', -1: 'w', -2: 'W'}
    print("  0 1 2 3 4 5 6 7")
    for r in range(8):
        line = f"{r} "
        for c in range(8):
            line += f"{scale[board[r,c]]} "
        print(line)
    print("")

def test():
    print("Initializing C++ game...")
    game = checkers_cpp.Game()
    
    print("Initial State (Player 1 turn):")
    board = get_board_array_from_game(game)
    render_board(board)
    
    # Verify P1 is at bottom (rows 5,6,7) - expected 'r'
    # Verify P2 is at top (rows 0,1,2) - expected 'w'
    
    # Make a move for P1
    # Move 20 -> 16 (indices). 
    # 20 is row 5, col 0. 16 is row 4, col 1.
    # Dir (5,0)->(4,1) is UR (NE).
    # Action = 20*4 + 0 = 80
    
    print("Making action 80 (20->16, NE)...")
    success = game.make_action(80)
    print(f"Success: {success}")
    
    print("State after P1 move (Player 2 turn):")
    board = get_board_array_from_game(game)
    render_board(board)
    
    # Should still show P1 (r) at bottom, but one piece moved.
    # P2 (w) at top.
    
    # Make a move for P2
    # P2 is at top. Moves 'Forward' (relative to self) is DOWN the board.
    # Piece at 9 (Row 2, Col 2). Move to 13 (Row 3, Col 1)?
    # Relative to P2 (at bottom): Piece at 22. Move to 17 (UR) or 18 (UL)?
    # Wait, we need to find a valid P2 move.
    
    legal_actions = game.get_legal_actions()
    print(f"P2 Legal Actions: {legal_actions}")
    
    if legal_actions:
        action = legal_actions[0]
        print(f"Making P2 action {action}...")
        game.make_action(action)
        
        print("State after P2 move (Player 1 turn):")
        board = get_board_array_from_game(game)
        render_board(board)

if __name__ == "__main__":
    test()
