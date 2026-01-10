"""
Pygame-based real-time game visualization for 8x8 checkers.

Shows the board state and policy heatmap during self-play.
"""

import pygame
import numpy as np
from typing import Optional

try:
    from ..engine.bitboard import square_to_row_col, NUM_SQUARES
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from checkers8x8.engine.bitboard import square_to_row_col, NUM_SQUARES


# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_SQUARE = (139, 69, 19)
LIGHT_SQUARE = (222, 184, 135)
RED_PIECE = (220, 20, 60)
BLACK_PIECE = (40, 40, 40)
GOLD_CROWN = (255, 215, 0)
POLICY_COLOR = (100, 200, 255)
TEXT_COLOR = (255, 255, 255)


class GameVisualizer:
    """
    Real-time pygame visualization for 8x8 checkers.

    Displays:
    - Board with pieces (men and kings)
    - Policy heatmap (where AI is considering moves)
    - Move count and game info
    """

    def __init__(self, square_size: int = 80):
        """
        Initialize visualizer.

        Args:
            square_size: Size of each square in pixels
        """
        pygame.init()

        self.square_size = square_size
        self.board_size = 8
        self.board_width = self.board_size * square_size
        self.info_width = 300
        self.window_width = self.board_width + self.info_width
        self.window_height = self.board_width

        # Create window
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("8x8 Checkers - Self-Play")

        # Font
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        self.clock = pygame.time.Clock()

    def render(
        self,
        game_array: np.ndarray,
        policy: Optional[np.ndarray] = None,
        move_count: int = 0,
        player: int = 1
    ):
        """
        Render the current game state.

        Args:
            game_array: 8x8 board array (pieces encoded as integers)
            policy: Policy distribution over 128 actions (optional)
            move_count: Current move number
            player: Current player (1 or 2)
        """
        # Clear screen
        self.screen.fill(BLACK)

        # Draw board and pieces
        self._draw_board(game_array, policy)

        # Draw info panel
        self._draw_info_panel(move_count, player)

        # Update display
        pygame.display.flip()
        self.clock.tick(30)  # 30 FPS

        # Process events (allow window to close)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False

        return True

    def _draw_board(self, game_array: np.ndarray, policy: Optional[np.ndarray]):
        """Draw the checkerboard with pieces and policy overlay."""
        # Draw squares
        for row in range(self.board_size):
            for col in range(self.board_size):
                x = col * self.square_size
                y = row * self.square_size

                # Checkerboard pattern
                if (row + col) % 2 == 0:
                    color = LIGHT_SQUARE
                else:
                    color = DARK_SQUARE

                pygame.draw.rect(
                    self.screen,
                    color,
                    (x, y, self.square_size, self.square_size)
                )

        # Draw policy heatmap (if provided)
        if policy is not None:
            self._draw_policy_heatmap(policy)

        # Draw pieces
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = game_array[row, col]

                if piece != 0:
                    self._draw_piece(row, col, piece)

    def _draw_policy_heatmap(self, policy: np.ndarray):
        """
        Draw policy probability heatmap.

        Shows which squares the AI is considering moving to.
        """
        # Convert action-based policy to square-based visualization
        square_probs = np.zeros(64)  # 8x8 board

        # Aggregate policy probabilities by destination square
        for action_idx in range(128):
            if policy[action_idx] > 0.01:  # Only show significant probabilities
                # Decode action: from_square * 4 + direction
                from_square = action_idx // 4
                direction = action_idx % 4

                # Map to board position
                from_row, from_col = square_to_row_col(from_square)

                # Direction offsets
                dir_offsets = {
                    0: (-1, -1),  # NW
                    1: (-1, +1),  # NE
                    2: (+1, -1),  # SW
                    3: (+1, +1),  # SE
                }

                dr, dc = dir_offsets[direction]
                to_row, to_col = from_row + dr, from_col + dc

                # Check bounds
                if 0 <= to_row < 8 and 0 <= to_col < 8:
                    board_idx = to_row * 8 + to_col
                    square_probs[board_idx] += policy[action_idx]

        # Normalize
        if square_probs.max() > 0:
            square_probs = square_probs / square_probs.max()

        # Draw heatmap
        for idx in range(64):
            if square_probs[idx] > 0.01:
                row = idx // 8
                col = idx % 8

                x = col * self.square_size
                y = row * self.square_size

                # Overlay semi-transparent blue
                alpha = int(255 * square_probs[idx])
                overlay = pygame.Surface((self.square_size, self.square_size))
                overlay.set_alpha(alpha // 2)  # Semi-transparent
                overlay.fill(POLICY_COLOR)
                self.screen.blit(overlay, (x, y))

    def _draw_piece(self, row: int, col: int, piece: int):
        """
        Draw a piece on the board.

        Args:
            row, col: Board position
            piece: Piece type
                1: Player man
                2: Player king
                -1: Opponent man
                -2: Opponent king
        """
        x = col * self.square_size + self.square_size // 2
        y = row * self.square_size + self.square_size // 2
        radius = self.square_size // 3

        # Determine color
        if piece > 0:
            color = RED_PIECE  # Player
        else:
            color = BLACK_PIECE  # Opponent

        # Draw piece
        pygame.draw.circle(self.screen, color, (x, y), radius)
        pygame.draw.circle(self.screen, BLACK, (x, y), radius, 2)  # Outline

        # Draw crown for kings
        if abs(piece) == 2:
            crown_size = radius // 2
            pygame.draw.circle(self.screen, GOLD_CROWN, (x, y), crown_size)

    def _draw_info_panel(self, move_count: int, player: int):
        """Draw information panel on the right."""
        panel_x = self.board_width
        padding = 20

        # Background
        pygame.draw.rect(
            self.screen,
            (30, 30, 30),
            (panel_x, 0, self.info_width, self.window_height)
        )

        # Title
        title = self.font_large.render("8x8 Checkers", True, WHITE)
        self.screen.blit(title, (panel_x + padding, padding))

        # Move count
        y_offset = 80
        move_text = self.font_small.render(f"Move: {move_count}", True, WHITE)
        self.screen.blit(move_text, (panel_x + padding, y_offset))

        # Current player
        y_offset += 40
        player_color = RED_PIECE if player == 1 else BLACK_PIECE
        player_text = self.font_small.render(f"Player: {player}", True, player_color)
        self.screen.blit(player_text, (panel_x + padding, y_offset))

        # Legend
        y_offset += 80
        legend_title = self.font_small.render("Legend:", True, WHITE)
        self.screen.blit(legend_title, (panel_x + padding, y_offset))

        y_offset += 30
        # Red piece
        pygame.draw.circle(self.screen, RED_PIECE, (panel_x + padding + 15, y_offset + 10), 12)
        legend_text = self.font_small.render("Player 1", True, WHITE)
        self.screen.blit(legend_text, (panel_x + padding + 40, y_offset))

        y_offset += 30
        # Black piece
        pygame.draw.circle(self.screen, BLACK_PIECE, (panel_x + padding + 15, y_offset + 10), 12)
        legend_text = self.font_small.render("Player 2", True, WHITE)
        self.screen.blit(legend_text, (panel_x + padding + 40, y_offset))

        y_offset += 30
        # King
        pygame.draw.circle(self.screen, RED_PIECE, (panel_x + padding + 15, y_offset + 10), 12)
        pygame.draw.circle(self.screen, GOLD_CROWN, (panel_x + padding + 15, y_offset + 10), 6)
        legend_text = self.font_small.render("King", True, WHITE)
        self.screen.blit(legend_text, (panel_x + padding + 40, y_offset))

        y_offset += 30
        # Policy
        pygame.draw.rect(self.screen, POLICY_COLOR, (panel_x + padding, y_offset, 25, 20))
        legend_text = self.font_small.render("AI Thinking", True, WHITE)
        self.screen.blit(legend_text, (panel_x + padding + 40, y_offset))

    def close(self):
        """Close the visualization window."""
        pygame.quit()


# Testing
if __name__ == "__main__":
    import time

    print("Testing Game Visualizer...")

    viz = GameVisualizer(square_size=80)

    # Create a test board
    board = np.zeros((8, 8), dtype=np.int8)

    # Add some pieces
    board[0, 1] = -1  # Opponent man
    board[0, 3] = -1
    board[1, 0] = -1
    board[2, 1] = -2  # Opponent king

    board[5, 2] = 1   # Player man
    board[6, 3] = 1
    board[7, 0] = 1
    board[4, 3] = 2   # Player king

    # Create a simple policy
    policy = np.zeros(128)
    policy[85] = 0.6  # High probability for action 85
    policy[89] = 0.4  # Medium probability for action 89

    # Render for 5 seconds
    print("Rendering test board for 5 seconds...")
    for i in range(150):  # ~5 seconds at 30 FPS
        if not viz.render(board, policy, move_count=i, player=1):
            break
        time.sleep(1/30)

    viz.close()
    print("✓ Visualizer test complete!")
