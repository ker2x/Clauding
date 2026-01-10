"""
Pygame-based real-time game visualization for 8x8 checkers.

Shows the board state and policy heatmap during self-play.
"""

import pygame
import numpy as np
from typing import Optional, Dict
from collections import deque

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
GRAPH_COLOR_TOTAL = (0, 255, 255)  # Cyan
GRAPH_COLOR_POLICY = (255, 100, 100)  # Reddish
GRAPH_COLOR_VALUE = (100, 255, 100)  # Greenish
GRAPH_COLOR_BUFFER = (255, 255, 0)  # Yellow


class GameVisualizer:
    """
    Real-time pygame visualization for 8x8 checkers with training metrics.

    Displays:
    - Board with pieces (men and kings)
    - Policy heatmap (where AI is considering moves)
    - Move count and game info
    - Real-time training metrics (loss and buffer size)
    """

    def __init__(self, square_size: int = 70, max_metrics: int = 100):
        """
        Initialize visualizer.

        Args:
            square_size: Size of each square in pixels
            max_metrics: Maximum number of metric points to store for graphs
        """
        pygame.init()

        self.square_size = square_size
        self.board_size = 8
        self.board_width = self.board_size * square_size
        self.info_width = 350  # Slightly wider for graphs
        self.window_width = self.board_width + self.info_width
        self.window_height = self.board_width

        # Metric storage
        self.max_metrics = max_metrics
        self.metrics_history = {
            'total_loss': deque(maxlen=max_metrics),
            'policy_loss': deque(maxlen=max_metrics),
            'value_loss': deque(maxlen=max_metrics),
            'buffer_size': deque(maxlen=max_metrics),
            'iterations': deque(maxlen=max_metrics)
        }

        # Create window
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("8x8 Checkers - Training Visualizer")

        # Font
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)

        self.clock = pygame.time.Clock()

    def update_metrics(self, metrics: Dict):
        """
        Update training metrics for graph rendering.

        Args:
            metrics: Dictionary with training stats
        """
        if 'total_loss' in metrics:
            self.metrics_history['total_loss'].append(metrics['total_loss'])
            self.metrics_history['policy_loss'].append(metrics.get('policy_loss', 0))
            self.metrics_history['value_loss'].append(metrics.get('value_loss', 0))
            self.metrics_history['buffer_size'].append(metrics.get('buffer_size', 0))
            self.metrics_history['iterations'].append(metrics.get('iteration', 0))

    def render(
        self,
        game_array: np.ndarray,
        policy: Optional[np.ndarray] = None,
        move_count: int = 0,
        player: int = 1
    ):
        """
        Render the current game state and metrics.

        Args:
            game_array: 8x8 board array
            policy: Policy distribution over 128 actions
            move_count: Current move number
            player: Current player (1 or 2)
        """
        # Clear screen
        self.screen.fill(BLACK)

        # Draw board and pieces
        self._draw_board(game_array, policy)

        # Draw info panel (text and graphs)
        self._draw_info_panel(move_count, player)

        # Update display
        pygame.display.flip()
        self.clock.tick(30)  # 30 FPS

        # Process events
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
        """Draw information panel on the right with graphs."""
        panel_x = self.board_width
        padding = 20
        y_offset = padding

        # Background
        pygame.draw.rect(self.screen, (30, 30, 30), (panel_x, 0, self.info_width, self.window_height))

        # Title
        title = self.font_large.render("8x8 Checkers Training", True, WHITE)
        self.screen.blit(title, (panel_x + padding, y_offset))
        y_offset += 40

        # Stats
        move_text = self.font_medium.render(f"Move: {move_count}", True, WHITE)
        self.screen.blit(move_text, (panel_x + padding, y_offset))
        y_offset += 30

        p_color = RED_PIECE if player == 1 else WHITE
        player_text = self.font_medium.render(f"Player: {player}", True, p_color)
        self.screen.blit(player_text, (panel_x + padding, y_offset))
        y_offset += 40

        # Draw Loss Graph
        y_offset = self._draw_graph_box(
            "Losses (Total, Policy, Value)", 
            [self.metrics_history['total_loss'], self.metrics_history['policy_loss'], self.metrics_history['value_loss']],
            [GRAPH_COLOR_TOTAL, GRAPH_COLOR_POLICY, GRAPH_COLOR_VALUE],
            panel_x + padding, y_offset, 
            self.info_width - 2 * padding, 120
        )
        
        y_offset += 20

        # Draw Buffer Graph
        y_offset = self._draw_graph_box(
            "Replay Buffer Size", 
            [self.metrics_history['buffer_size']],
            [GRAPH_COLOR_BUFFER],
            panel_x + padding, y_offset, 
            self.info_width - 2 * padding, 100
        )

        # Legend (simplified)
        y_offset += 20
        legend_y = self.window_height - 100
        legend_text = self.font_small.render("Legend: Red=P1, Black=P2, Blue=AI Plan", True, WHITE)
        self.screen.blit(legend_text, (panel_x + padding, legend_y))

    def _draw_graph_box(self, title, data_lists, colors, x, y, w, h):
        """Draw a box with one or more lines of data."""
        # Box background
        pygame.draw.rect(self.screen, (10, 10, 10), (x, y, w, h))
        pygame.draw.rect(self.screen, (100, 100, 100), (x, y, w, h), 1)

        # Title
        title_surf = self.font_small.render(title, True, WHITE)
        self.screen.blit(title_surf, (x, y - 18))

        # Plot data
        max_val = 0.001
        for data in data_lists:
            if len(data) > 0:
                max_val = max(max_val, max(data))
        
        # Add some margin to top of graph
        max_val *= 1.1

        for data, color in zip(data_lists, colors):
            if len(data) < 2:
                continue
            
            points = []
            for i, val in enumerate(data):
                px = x + (i / (self.max_metrics - 1)) * w
                py = y + h - (val / max_val) * h
                points.append((px, py))
            
            pygame.draw.lines(self.screen, color, False, points, 2)

        # Show current value of first data list if available
        if len(data_lists[0]) > 0:
            val_text = self.font_small.render(f"{data_lists[0][-1]:.4f}", True, colors[0])
            self.screen.blit(val_text, (x + w - 50, y + 5))

        return y + h

    def close(self):
        """Close the visualization window."""
        pygame.quit()


# Testing
if __name__ == "__main__":
    import time

    print("Testing Game Visualizer...")
    viz = GameVisualizer(square_size=70)

    # Create a test board
    board = np.zeros((8, 8), dtype=np.int8)
    board[3, 3] = 1
    board[4, 4] = -1

    # Simulate metrics
    for i in range(1, 51):
        viz.update_metrics({
            'iteration': i,
            'total_loss': 2.0 / i + np.random.rand() * 0.1,
            'policy_loss': 1.5 / i + np.random.rand() * 0.05,
            'value_loss': 0.5 / i + np.random.rand() * 0.05,
            'buffer_size': i * 1000
        })
        
        if not viz.render(board, move_count=i):
            break
        time.sleep(0.05)

    print("✓ Visualizer test complete!")
    time.sleep(1)
    viz.close()
