"""Pygame rendering of ragdolls, ground, joint controls, and UI."""

import math
import pygame
import pymunk
from config.body_config import BodyConfig, JointState, DEFAULT_BODY
from config.constants import ARENA_WIDTH, ARENA_HEIGHT, GROUND_Y
from physics.ragdoll import Ragdoll
from game.match import Match


# Colors
BG_COLOR = (30, 30, 35)
GROUND_COLOR = (60, 60, 65)
GRID_COLOR = (40, 40, 45)

PLAYER_A_TINT = (0.7, 0.85, 1.0)  # blue-ish
PLAYER_B_TINT = (1.0, 0.75, 0.7)  # red-ish

JOINT_STATE_COLORS = {
    JointState.CONTRACT: (255, 80, 80),    # red
    JointState.EXTEND: (80, 160, 255),     # blue
    JointState.HOLD: (80, 220, 80),        # green
    JointState.RELAX: (140, 140, 140),     # gray
}

JOINT_STATE_NAMES = {
    JointState.CONTRACT: "CON",
    JointState.EXTEND: "EXT",
    JointState.HOLD: "HLD",
    JointState.RELAX: "RLX",
}

# Screen layout
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
VIEWPORT_HEIGHT = 500  # physics viewport
PANEL_HEIGHT = 200     # bottom panel for joint controls
SCALE = SCREEN_WIDTH / ARENA_WIDTH  # pixels per cm


def world_to_screen(x: float, y: float) -> tuple[int, int]:
    """Convert physics coords to screen coords (y-flip)."""
    sx = int(x * SCALE)
    sy = int(VIEWPORT_HEIGHT - (y - GROUND_Y + 30) * SCALE)
    return (sx, sy)


class PygameRenderer:
    """Renders the match state using pygame."""

    def __init__(self, match: Match | None = None, mode: str = "human"):
        self.mode = mode
        self._initialized = False
        self.screen = None
        self.clock = None
        self.font = None
        self.small_font = None

    def _init_pygame(self):
        if self._initialized:
            return
        pygame.init()
        if self.mode == "human":
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Toribash 2D")
        else:
            self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 18)
        self.small_font = pygame.font.SysFont("monospace", 13)
        self._initialized = True

    def render(self, match: Match) -> pygame.Surface | None:
        """Render current match state."""
        self._init_pygame()

        self.screen.fill(BG_COLOR)

        # Draw ground
        gy = world_to_screen(0, GROUND_Y)[1]
        pygame.draw.line(self.screen, GROUND_COLOR, (0, gy), (SCREEN_WIDTH, gy), 3)
        pygame.draw.rect(self.screen, (25, 25, 28),
                         (0, gy, SCREEN_WIDTH, VIEWPORT_HEIGHT - gy))

        # Draw grid
        for x in range(0, int(ARENA_WIDTH), 50):
            sx = int(x * SCALE)
            pygame.draw.line(self.screen, GRID_COLOR, (sx, 0), (sx, gy), 1)
        for y in range(int(GROUND_Y), int(GROUND_Y + ARENA_HEIGHT), 50):
            sy = world_to_screen(0, y)[1]
            if 0 <= sy <= gy:
                pygame.draw.line(self.screen, GRID_COLOR, (0, sy), (SCREEN_WIDTH, sy), 1)

        # Draw ragdolls
        self._draw_ragdoll(match.world.ragdoll_a, PLAYER_A_TINT)
        self._draw_ragdoll(match.world.ragdoll_b, PLAYER_B_TINT)

        # Draw separator line
        pygame.draw.line(self.screen, (50, 50, 55),
                         (0, VIEWPORT_HEIGHT), (SCREEN_WIDTH, VIEWPORT_HEIGHT), 2)

        # Draw bottom panel
        self._draw_panel(match)

        if self.mode == "human":
            pygame.display.flip()
            return None
        else:
            return self.screen.copy()

    def _draw_ragdoll(self, ragdoll: Ragdoll, tint: tuple[float, float, float]):
        """Draw all segments and joints of a ragdoll."""
        config = ragdoll.body_config

        # Draw segments as rotated rectangles
        for seg_def in config.segments:
            body, shape = ragdoll.segments[seg_def.name]
            vertices = shape.get_vertices()
            # Transform to world coords
            world_verts = [body.local_to_world(v) for v in vertices]
            screen_verts = [world_to_screen(v.x, v.y) for v in world_verts]

            # Apply tint to segment color
            color = tuple(int(c * t) for c, t in zip(seg_def.color, tint))
            pygame.draw.polygon(self.screen, color, screen_verts)
            pygame.draw.polygon(self.screen, (255, 255, 255), screen_verts, 1)

        # Draw joints as colored dots
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
                pygame.draw.circle(self.screen, color, (sx, sy), 4)

    def _draw_panel(self, match: Match):
        """Draw the bottom panel with joint controls and info."""
        panel_y = VIEWPORT_HEIGHT + 5

        # Turn info
        info_text = f"Turn: {match.turn}/{match.config.max_turns}"
        self.screen.blit(self.font.render(info_text, True, (200, 200, 200)),
                         (10, panel_y))

        # Scores
        score_text = f"Score: A={match.scores[0]:.1f}  B={match.scores[1]:.1f}"
        self.screen.blit(self.font.render(score_text, True, (200, 200, 200)),
                         (250, panel_y))

        # Instructions
        instr = "SPACE=simulate  R=reset  Q=quit"
        self.screen.blit(self.font.render(instr, True, (140, 140, 140)),
                         (550, panel_y))

        # Joint controls for both players
        self._draw_joint_controls(match.world.ragdoll_a, "Player A", panel_y + 30, 0)
        self._draw_joint_controls(match.world.ragdoll_b, "Player B", panel_y + 30, SCREEN_WIDTH // 2)

    def _draw_joint_controls(self, ragdoll: Ragdoll, label: str, y: int, x_offset: int):
        """Draw clickable joint state controls."""
        half_w = SCREEN_WIDTH // 2
        self.screen.blit(self.small_font.render(label, True, (180, 180, 180)),
                         (x_offset + 5, y))

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
        """Get joint name from a click position in the panel. Returns None if not on a joint."""
        panel_y = VIEWPORT_HEIGHT + 5 + 30 + 18  # below header
        half_w = SCREEN_WIDTH // 2
        x_offset = 0 if ragdoll_idx == 0 else half_w

        x, y = pos
        if x < x_offset or x >= x_offset + half_w:
            return None
        if y < panel_y:
            return None

        col_width = half_w // 7
        row_height = 18

        col = (x - x_offset - 5) // col_width
        row = (y - panel_y) // row_height

        idx = row * 7 + col
        joints = DEFAULT_BODY.joints
        if 0 <= idx < len(joints):
            return joints[idx].name
        return None

    def close(self):
        if self._initialized:
            pygame.quit()
            self._initialized = False
