"""Pygame rendering of ragdolls, ground, joint controls, and UI.

This module provides the PygameRenderer class for visualizing the game state.
It renders:
- Arena with ground and grid
- Two ragdolls with colored segments
- Joint state indicators (colored dots per joint)
- Bottom panel with turn info and joint controls
- Clickable joint controls for human play

Screen Layout:
    +------------------------------------------+
    |                                          |
    |              Game Viewport                |
    |           (Ragdolls + Ground)            |
    |                                          |
    +------------------------------------------+
    |  Turn: 1/20   Score: A=0.0 B=0.0       |
    |  Player A joints | Player B joints      |
    +------------------------------------------+

Usage:
    >>> from rendering.pygame_renderer import PygameRenderer
    >>> from game.match import Match
    >>> match = Match()
    >>> renderer = PygameRenderer(match, mode="human")
    >>> while True:
    ...     renderer.render(match)
    ...     # Handle pygame events
    ...     pygame.time.wait(16)
"""

import math
import pygame
import pymunk
from config.body_config import BodyConfig, JointState, DEFAULT_BODY
from config.constants import ARENA_WIDTH, ARENA_HEIGHT, GROUND_Y, VIEWPORT_Y_OFFSET, GRID_SPACING
from physics.ragdoll import Ragdoll
from game.match import Match


# =============================================================================
# Color Constants
# =============================================================================

# Background and ground colors
BG_COLOR = (30, 30, 35)
GROUND_COLOR = (60, 60, 65)
GRID_COLOR = (40, 40, 45)

# Player tint colors (applied to segment colors)
# Blue tint for Player A, red tint for Player B
PLAYER_A_TINT = (0.7, 0.85, 1.0)
PLAYER_B_TINT = (1.0, 0.75, 0.7)

# Joint state colors (for indicators and controls)
JOINT_STATE_COLORS = {
    JointState.CONTRACT: (255, 80, 80),    # Red
    JointState.EXTEND: (80, 160, 255),     # Blue
    JointState.HOLD: (80, 220, 80),         # Green
    JointState.RELAX: (140, 140, 140),      # Gray
}

# Joint state abbreviations
JOINT_STATE_NAMES = {
    JointState.CONTRACT: "CON",
    JointState.EXTEND: "EXT",
    JointState.HOLD: "HLD",
    JointState.RELAX: "RLX",
}


# =============================================================================
# Screen Layout Constants
# =============================================================================

# Total screen dimensions
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700

# Viewport (top portion) for game rendering
VIEWPORT_HEIGHT = 500

# Panel (bottom portion) for controls and info
PANEL_HEIGHT = 200

# Scale factor: pixels per centimeter
# Arena is 600cm wide, screen is 1000px wide
SCALE = SCREEN_WIDTH / ARENA_WIDTH


def world_to_screen(x: float, y: float) -> tuple[int, int]:
    """Convert physics coordinates to screen coordinates.
    
    Physics coordinates: origin at ground level, Y increases upward
    Screen coordinates: origin at top-left, Y increases downward
    
    Args:
        x: Physics X coordinate (cm)
        y: Physics Y coordinate (cm)
    
    Returns:
        (screen_x, screen_y) tuple in pixels.
    """
    sx = int(x * SCALE)
    # Flip Y and offset by viewport height
    sy = int(VIEWPORT_HEIGHT - (y - GROUND_Y + VIEWPORT_Y_OFFSET) * SCALE)
    return (sx, sy)


class PygameRenderer:
    """Renders the Toribash match state using pygame.
    
    This class handles all visualization of the game:
    - Ragdolls with colored segments
    - Ground and grid
    - Joint state indicators
    - Bottom panel with controls and info
    
    Attributes:
        mode: Rendering mode ("human" for display, else returns surfaces).
        screen: pygame.Surface for rendering.
        clock: pygame.time.Clock for frame rate control.
        font: Font for text rendering.
        small_font: Smaller font for panel text.
    
    Usage:
        renderer = PygameRenderer(match, mode="human")
        while running:
            renderer.render(match)
            pygame.time.wait(16)
    """
    
    def __init__(self, match: Match | None = None, mode: str = "human"):
        """Initialize the renderer.
        
        Args:
            match: Match to render (can be None, set later in render()).
            mode: "human" for interactive display, other for offscreen.
        """
        self.mode = mode
        self._initialized = False
        self.screen = None
        self.clock = None
        self.font = None
        self.small_font = None

    def _init_pygame(self) -> None:
        """Initialize pygame and create screen/surfaces.
        
        Called lazily on first render to avoid pygame initialization
        when not needed (e.g., headless training).
        """
        if self._initialized:
            return
        
        pygame.init()
        
        if self.mode == "human":
            # Interactive display mode
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Toribash 2D")
        else:
            # Offscreen mode (for recording, etc.)
            self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        
        # Create clock for frame rate control
        self.clock = pygame.time.Clock()
        
        # Create fonts
        self.font = pygame.font.SysFont("monospace", 18)
        self.small_font = pygame.font.SysFont("monospace", 13)
        
        self._initialized = True

    def render(self, match: Match) -> pygame.Surface | None:
        """Render the current match state.
        
        Args:
            match: Match instance to render.
        
        Returns:
            None in "human" mode (updates display directly).
            pygame.Surface in other modes.
        """
        self._init_pygame()

        # Fill background
        self.screen.fill(BG_COLOR)

        # Draw ground line
        gy = world_to_screen(0, GROUND_Y)[1]
        pygame.draw.line(self.screen, GROUND_COLOR, (0, gy), (SCREEN_WIDTH, gy), 3)
        
        # Draw ground fill
        pygame.draw.rect(
            self.screen, (25, 25, 28),
            (0, gy, SCREEN_WIDTH, VIEWPORT_HEIGHT - gy)
        )

        # Draw grid lines
        for x in range(0, int(ARENA_WIDTH), int(GRID_SPACING)):
            sx = int(x * SCALE)
            pygame.draw.line(self.screen, GRID_COLOR, (sx, 0), (sx, gy), 1)
        for y in range(int(GROUND_Y), int(GROUND_Y + ARENA_HEIGHT), int(GRID_SPACING)):
            sy = world_to_screen(0, y)[1]
            if 0 <= sy <= gy:
                pygame.draw.line(self.screen, GRID_COLOR, (0, sy), (SCREEN_WIDTH, sy), 1)

        # Draw both ragdolls
        self._draw_ragdoll(match.world.ragdoll_a, PLAYER_A_TINT)
        self._draw_ragdoll(match.world.ragdoll_b, PLAYER_B_TINT)

        # Draw separator line between viewport and panel
        pygame.draw.line(
            self.screen, (50, 50, 55),
            (0, VIEWPORT_HEIGHT), (SCREEN_WIDTH, VIEWPORT_HEIGHT), 2
        )

        # Draw bottom panel with info and controls
        self._draw_panel(match)

        # Update display or return surface
        if self.mode == "human":
            pygame.display.flip()
            return None
        else:
            return self.screen.copy()

    def _draw_ragdoll(self, ragdoll: Ragdoll, tint: tuple[float, float, float]) -> None:
        """Draw all segments and joints of a ragdoll.
        
        Args:
            ragdoll: Ragdoll instance to draw.
            tint: RGB multiplier (0-1) applied to segment colors.
        """
        config = ragdoll.body_config

        # Draw each segment as a rotated rectangle
        for seg_def in config.segments:
            body, shape = ragdoll.segments[seg_def.name]
            vertices = shape.get_vertices()
            
            # Transform vertices from body-local to world coordinates
            world_verts = [body.local_to_world(v) for v in vertices]
            screen_verts = [world_to_screen(v.x, v.y) for v in world_verts]

            # Apply tint to segment color
            color = tuple(int(c * t) for c, t in zip(seg_def.color, tint))
            
            # Draw filled polygon
            pygame.draw.polygon(self.screen, color, screen_verts)
            
            # Draw outline
            pygame.draw.polygon(self.screen, (255, 255, 255), screen_verts, 1)

        # Draw joint indicators
        for jdef in config.joints:
            parent_body = ragdoll.segments[jdef.parent][0]
            anchor_world = parent_body.local_to_world(jdef.anchor_parent)
            sx, sy = world_to_screen(anchor_world.x, anchor_world.y)

            state = ragdoll.joint_states.get(jdef.name, JointState.HOLD)
            color = JOINT_STATE_COLORS[state]

            if jdef.name in ragdoll.dismembered:
                # Draw X for dismembered joints
                pygame.draw.line(self.screen, (255, 0, 0), (sx - 4, sy - 4), (sx + 4, sy + 4), 2)
                pygame.draw.line(self.screen, (255, 0, 0), (sx - 4, sy + 4), (sx + 4, sy - 4), 2)
            else:
                # Draw circle for active joints
                pygame.draw.circle(self.screen, color, (sx, sy), 4)

    def _draw_panel(self, match: Match) -> None:
        """Draw the bottom panel with turn info and joint controls.
        
        Args:
            match: Match instance to read info from.
        """
        panel_y = VIEWPORT_HEIGHT + 5

        # Turn counter
        info_text = f"Turn: {match.turn}/{match.config.max_turns}"
        self.screen.blit(
            self.font.render(info_text, True, (200, 200, 200)),
            (10, panel_y)
        )

        # Score display
        score_text = f"Score: A={match.scores[0]:.1f}  B={match.scores[1]:.1f}"
        self.screen.blit(
            self.font.render(score_text, True, (200, 200, 200)),
            (250, panel_y)
        )

        # Instructions
        instr = "SPACE=simulate  R=reset  Q=quit"
        self.screen.blit(
            self.font.render(instr, True, (140, 140, 140)),
            (550, panel_y)
        )

        # Joint controls for both players (side by side)
        self._draw_joint_controls(
            match.world.ragdoll_a, "Player A", panel_y + 30, 0
        )
        self._draw_joint_controls(
            match.world.ragdoll_b, "Player B", panel_y + 30, SCREEN_WIDTH // 2
        )

    def _draw_joint_controls(
        self, ragdoll: Ragdoll, label: str, y: int, x_offset: int
    ) -> None:
        """Draw joint state controls for one player.
        
        Displays all 14 joints with their current state (color-coded).
        
        Args:
            ragdoll: Ragdoll to draw controls for.
            label: Player label (e.g., "Player A").
            y: Y position to start drawing.
            x_offset: X offset for left edge.
        """
        half_w = SCREEN_WIDTH // 2
        self.screen.blit(
            self.small_font.render(label, True, (180, 180, 180)),
            (x_offset + 5, y)
        )

        col_width = half_w // 7
        row_height = 18
        y += 18

        for i, jdef in enumerate(ragdoll.body_config.joints):
            col = i % 7
            row = i // 7
            x = x_offset + 5 + col * col_width
            ry = y + row * row_height

            state = ragdoll.joint_states.get(jdef.name, JointState.HOLD)
            color = JOINT_STATE_COLORS[state]
            state_name = JOINT_STATE_NAMES[state]

            # Draw joint name + state
            short_name = jdef.name[:6]
            text = f"{short_name}:{state_name}"
            surf = self.small_font.render(text, True, color)
            self.screen.blit(surf, (x, ry))

    def get_joint_at_pos(self, pos: tuple[int, int], ragdoll_idx: int) -> str | None:
        """Get joint name from a click position in the panel.
        
        Used for human play - maps mouse clicks to joint controls.
        
        Args:
            pos: (x, y) click position in screen coordinates.
            ragdoll_idx: Which player's panel (0 for A, 1 for B).
        
        Returns:
            Joint name if click is on a joint, None otherwise.
        """
        panel_y = VIEWPORT_HEIGHT + 5 + 30 + 18  # Below header
        half_w = SCREEN_WIDTH // 2
        x_offset = 0 if ragdoll_idx == 0 else half_w

        x, y = pos
        
        # Check bounds
        if x < x_offset or x >= x_offset + half_w:
            return None
        if y < panel_y:
            return None

        col_width = half_w // 7
        row_height = 18

        # Calculate column and row from position
        col = (x - x_offset - 5) // col_width
        row = (y - panel_y) // row_height

        idx = row * 7 + col
        joints = DEFAULT_BODY.joints
        
        if 0 <= idx < len(joints):
            return joints[idx].name
        return None

    def close(self) -> None:
        """Clean up pygame resources."""
        if self._initialized:
            pygame.quit()
            self._initialized = False
