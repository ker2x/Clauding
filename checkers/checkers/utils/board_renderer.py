"""
Pygame-based checkers board renderer.
"""

import pygame
import numpy as np
from typing import Optional, Tuple


class BoardRenderer:
    """
    Render 10x10 checkers board with pieces and overlays.
    """
    
    def __init__(self, square_size: int = 50):
        """
        Initialize board renderer.
        
        Args:
            square_size: Size of each square in pixels
        """
        self.square_size = square_size
        self.board_size = 10
        self.total_size = self.square_size * self.board_size
        
        # Colors
        self.dark_square = (139, 90, 43)    # Brown
        self.light_square = (222, 184, 135)  # Tan
        self.player1_color = (220, 20, 60)   # Crimson (Player 1)
        self.player2_color = (248, 248, 255) # Ghost white (Player 2)
        self.king_crown_color = (255, 215, 0)  # Gold
        self.highlight_color = (255, 255, 0, 100)  # Yellow transparent
        
        # Fonts
        try:
            self.coord_font = pygame.font.SysFont('Arial', 12)
        except:
            self.coord_font = pygame.font.Font(None, 16)
    
    def render_board(self, surface: pygame.Surface, offset: Tuple[int, int] = (0, 0)):
        """
        Render the checkerboard pattern.
        
        Args:
            surface: Surface to draw on
            offset: (x, y) offset for board position
        """
        for row in range(self.board_size):
            for col in range(self.board_size):
                x = offset[0] + col * self.square_size
                y = offset[1] + row * self.square_size
                
                # Checkerboard pattern: dark squares on (row + col) % 2 == 1
                if (row + col) % 2 == 1:
                    color = self.dark_square
                else:
                    color = self.light_square
                
                pygame.draw.rect(surface, color, (x, y, self.square_size, self.square_size))
        
        # Draw border
        border_rect = pygame.Rect(offset[0], offset[1], self.total_size, self.total_size)
        pygame.draw.rect(surface, (50, 50, 50), border_rect, 3)
    
    def render_coordinates(self, surface: pygame.Surface, offset: Tuple[int, int] = (0, 0)):
        """
        Render row and column coordinates.
        
        Args:
            surface: Surface to draw on
            offset: (x, y) offset for board position
        """
        # Column labels (a-j) at bottom
        for col in range(self.board_size):
            label = chr(ord('a') + col)
            text_surf = self.coord_font.render(label, True, (0, 0, 0))
            text_rect = text_surf.get_rect()
            text_rect.centerx = offset[0] + col * self.square_size + self.square_size // 2
            text_rect.top = offset[1] + self.total_size + 5
            surface.blit(text_surf, text_rect)
        
        # Row labels (1-10) on left
        for row in range(self.board_size):
            label = str(self.board_size - row)
            text_surf = self.coord_font.render(label, True, (0, 0, 0))
            text_rect = text_surf.get_rect()
            text_rect.right = offset[0] - 5
            text_rect.centery = offset[1] + row * self.square_size + self.square_size // 2
            surface.blit(text_surf, text_rect)
    
    def render_pieces(
        self,
        surface: pygame.Surface,
        game_array: np.ndarray,
        offset: Tuple[int, int] = (0, 0)
    ):
        """
        Render pieces on the board.
        
        Game array encoding:
        - 0: Empty
        - 1: Player 1 man
        - 2: Player 1 king
        - -1: Player 2 man  
        - -2: Player 2 king
        
        Args:
            surface: Surface to draw on
            game_array: 10x10 numpy array with piece positions
            offset: (x, y) offset for board position
        """
        piece_radius = self.square_size // 3
        
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = game_array[row, col]
                
                if piece == 0:
                    continue
                
                # Calculate center of square
                center_x = offset[0] + col * self.square_size + self.square_size // 2
                center_y = offset[1] + row * self.square_size + self.square_size // 2
                
                # Determine color based on player
                if piece > 0:
                    color = self.player1_color
                else:
                    color = self.player2_color
                
                # Draw piece (circle)
                pygame.draw.circle(surface, color, (center_x, center_y), piece_radius)
                pygame.draw.circle(surface, (0, 0, 0), (center_x, center_y), piece_radius, 2)
                
                # Draw king indicator (crown or double circle)
                if abs(piece) == 2:
                    # Draw smaller inner circle for king
                    pygame.draw.circle(
                        surface,
                        self.king_crown_color,
                        (center_x, center_y),
                        piece_radius // 2,
                        3
                    )
    
    def render_heatmap(
        self,
        surface: pygame.Surface,
        policy: np.ndarray,
        legal_moves: Optional[list] = None,
        offset: Tuple[int, int] = (0, 0),
        alpha: int = 128
    ):
        """
        Render policy heatmap overlay on board.
        
        Args:
            surface: Surface to draw on
            policy: Policy probabilities (150-dim vector)
            legal_moves: List of legal moves (optional, for filtering)
            offset: (x, y) offset for board position
            alpha: Transparency (0-255)
        """
        if policy is None or len(policy) == 0:
            return
        
        # Create temporary surface for transparency
        heatmap_surface = pygame.Surface((self.total_size, self.total_size), pygame.SRCALPHA)
        
        # Normalize policy values
        max_prob = np.max(policy) if np.max(policy) > 0 else 1.0
        
        # Map policy to board squares (simplified visualization)
        # In reality, policy maps to moves, not squares
        # For visualization, we'll show top moves' destination squares
        top_k = 10
        top_indices = np.argsort(policy)[-top_k:][::-1]
        
        for idx in top_indices:
            prob = policy[idx]
            if prob < 0.01:
                continue
            
            # Simple mapping: idx to board position (this is approximate)
            # In a real implementation, you'd map move indices to actual moves
            row = (idx // 10) % 10
            col = idx % 10
            
            # Calculate color intensity
            intensity = int(255 * (prob / max_prob))
            color = (255, 0, 0, min(alpha, intensity))
            
            x = col * self.square_size
            y = row * self.square_size
            
            pygame.draw.rect(heatmap_surface, color, (x, y, self.square_size, self.square_size))
        
        surface.blit(heatmap_surface, offset)
    
    def render_highlight(
        self,
        surface: pygame.Surface,
        row: int,
        col: int,
        offset: Tuple[int, int] = (0, 0),
        color: Optional[Tuple[int, int, int, int]] = None
    ):
        """
        Highlight a specific square.
        
        Args:
            surface: Surface to draw on
            row: Row index (0-9)
            col: Column index (0-9)
            offset: (x, y) offset for board position
            color: RGBA color (default yellow transparent)
        """
        if color is None:
            color = self.highlight_color
        
        highlight_surface = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
        highlight_surface.fill(color)
        
        x = offset[0] + col * self.square_size
        y = offset[1] + row * self.square_size
        
        surface.blit(highlight_surface, (x, y))


if __name__ == "__main__":
    # Test the board renderer
    pygame.init()
    
    screen_width = 700
    screen_height = 700
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Board Renderer Test")
    
    clock = pygame.time.Clock()
    renderer = BoardRenderer(square_size=60)
    
    # Create test board with some pieces
    board = np.zeros((10, 10), dtype=np.int8)
    
    # Initial setup (simplified)
    # Player 1 pieces (bottom)
    for row in range(4):
        for col in range(10):
            if (row + col) % 2 == 1:
                board[row, col] = 1
    
    # Player 2 pieces (top)
    for row in range(6, 10):
        for col in range(10):
            if (row + col) % 2 == 1:
                board[row, col] = -1
    
    # Add some kings
    board[2, 3] = 2
    board[7, 6] = -2
    
    # Board offset for centering
    offset_x = (screen_width - renderer.total_size) // 2
    offset_y = 30
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
        
        screen.fill((240, 240, 240))
        
        # Render board
        renderer.render_board(screen, offset=(offset_x, offset_y))
        renderer.render_pieces(screen, board, offset=(offset_x, offset_y))
        renderer.render_coordinates(screen, offset=(offset_x, offset_y))
        
        # Test highlight
        renderer.render_highlight(screen, 2, 3, offset=(offset_x, offset_y))
        
        # Title
        try:
            font = pygame.font.SysFont('Arial', 20, bold=True)
        except:
            font = pygame.font.Font(None, 24)
        title = font.render("Board Renderer Test - Press Q to quit", True, (0, 0, 0))
        screen.blit(title, (screen_width // 2 - title.get_width() // 2, 5))
        
        pygame.display.flip()
        clock.tick(30)
    
    pygame.quit()
    print("✓ Board renderer test passed!")
