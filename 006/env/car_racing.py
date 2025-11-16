__credits__ = ["Laurent Laborde"]

import math
import time

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from .car_dynamics import Car
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import EzPickle

# Import reward configuration from constants
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from constants import (
    PROGRESS_REWARD_SCALE,
    LAP_COMPLETION_REWARD,
    STEP_PENALTY,
    STATIONARY_PENALTY,
    STATIONARY_SPEED_THRESHOLD,
    ONTRACK_REWARD,
    OFFTRACK_PENALTY,
    OFFTRACK_THRESHOLD,
    OFFTRACK_TERMINATION_PENALTY,
    FORWARD_SPEED_REWARD_SCALE,
)

# Import domain randomization
from config.domain_randomization import DomainRandomizationConfig
from utils.domain_randomizer import DomainRandomizer


# Box2D no longer needed - using custom 2D physics engine
# Removed Box2D dependency for cleaner, more interpretable physics

try:
    # As pygame is necessary for using the environment (reset and step) even without a render mode
    #   therefore, pygame is a necessary import for the environment.
    import pygame
    from pygame import gfxdraw
except ImportError as e:
    raise DependencyNotInstalled(
        'pygame is not installed, run `pip install pygame`'
    ) from e

try:
    import cv2
except ImportError as e:
    raise DependencyNotInstalled(
        'opencv is not installed, run `pip install opencv-python`'
    ) from e

# rendering resolution
VIDEO_W = 600
VIDEO_H = 400

# window size (rendering + gui)
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0                 # Track scale
TRACK_RAD = 900 / SCALE     # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE    # Game over boundary

FPS = 50  # Frames per second
ZOOM = 2.7  # Camera zoom

TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4
GRASS_DIM = PLAYFIELD / 20.0
MAX_SHAPE_DIM = (
    max(GRASS_DIM, TRACK_WIDTH, TRACK_DETAIL_STEP) * math.sqrt(2) * ZOOM * SCALE
)


class FrictionDetector:
    """
    Detects wheel-track collisions using accurate polygon-based geometry.

    Replaces Box2D contact listener with spatial geometry queries.
    Uses spatial partitioning for performance: only checks tiles near the car
    (~61 tiles instead of all 300), reducing computational cost by 80%.

    Performance Optimization:
    - Two-stage search: coarse (every 10th tile) then fine refinement
    - Spatial range: ±30 tiles around car position (wraps around for circular track)
    - Per-step cost: ~244 polygon checks (4 wheels × 61 tiles)
    - Tolerance: 0.3 units outside polygon edge for wheel-on-track detection

    Methods:
    - update_contacts(): Main entry point, called each physics step
    - _point_in_polygon(): Ray casting algorithm for point-in-polygon test
    - _distance_to_polygon_edge(): Minimum distance from point to polygon edges
    """
    def __init__(self, env, lap_complete_percent):
        self.env = env
        self.lap_complete_percent = lap_complete_percent

    def _point_in_polygon(self, px, py, vertices):
        """
        Check if point (px, py) is inside polygon using ray casting algorithm.

        Args:
            px, py: Point coordinates
            vertices: List of (x, y) tuples defining polygon vertices

        Returns:
            True if point is inside polygon, False otherwise
        """
        n = len(vertices)
        inside = False

        x1, y1 = vertices[0]
        for i in range(1, n + 1):
            x2, y2 = vertices[i % n]
            # Ray casting: count intersections with edges
            if ((y1 > py) != (y2 > py)) and (px < (x2 - x1) * (py - y1) / (y2 - y1) + x1):
                inside = not inside
            x1, y1 = x2, y2

        return inside

    def _distance_to_polygon_edge(self, px, py, vertices):
        """
        Calculate minimum distance from point to polygon edges.

        Args:
            px, py: Point coordinates
            vertices: List of (x, y) tuples defining polygon vertices

        Returns:
            Minimum distance from point to any polygon edge
        """
        min_dist = float('inf')

        n = len(vertices)
        for i in range(n):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % n]

            # Vector from edge start to end
            dx = x2 - x1
            dy = y2 - y1
            len_sq = dx * dx + dy * dy

            if len_sq < 1e-10:
                # Degenerate edge (two vertices at same position)
                dist = np.sqrt((px - x1)**2 + (py - y1)**2)
            else:
                # Project point onto edge line segment
                t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / len_sq))
                proj_x = x1 + t * dx
                proj_y = y1 + t * dy
                dist = np.sqrt((px - proj_x)**2 + (py - proj_y)**2)

            min_dist = min(min_dist, dist)

        return min_dist

    def update_contacts(self, car, road_tiles):
        """
        Update wheel-tile contacts based on accurate polygon geometry.
        Uses spatial partitioning to only check nearby tiles (~61 instead of 300).
        Called each step to determine which tiles each wheel is touching.

        Performance: ~2000 geometric operations per step (4 wheels × 61 tiles × 8 ops)
        Still very fast due to simple arithmetic and spatial partitioning.
        """
        # Small tolerance for wheels just barely off the track edge
        # This accounts for wheel radius and numerical precision
        NEAR_TRACK_THRESHOLD = 0.3  # Allow 0.3 units outside polygon edge
        SPATIAL_CHECK_RANGE = 30  # Only check tiles within ±30 indices of car position

        # Clear old contacts
        for wheel in car.wheels:
            wheel.tiles.clear()

        # Get car position to determine nearby tiles
        car_x, car_y = car.hull.position

        # Find closest tile to car using cached tile centers
        # Two-stage coarse-then-fine search for efficiency
        min_dist_sq = float('inf')
        closest_tile_idx = 0

        # Coarse search: check every 10th tile
        for i in range(0, len(road_tiles), 10):
            tile = road_tiles[i]
            dx = car_x - tile.center_x
            dy = car_y - tile.center_y
            dist_sq = dx * dx + dy * dy
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_tile_idx = i

        # Fine search: refine around coarse result
        search_start = max(0, closest_tile_idx - 10)
        search_end = min(len(road_tiles), closest_tile_idx + 11)
        for i in range(search_start, search_end):
            tile = road_tiles[i]
            dx = car_x - tile.center_x
            dy = car_y - tile.center_y
            dist_sq = dx * dx + dy * dy
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_tile_idx = i

        # Build list of nearby tiles to check (wrap around for circular track)
        num_tiles = len(road_tiles)
        tile_indices_to_check = []
        for offset in range(-SPATIAL_CHECK_RANGE, SPATIAL_CHECK_RANGE + 1):
            idx = (closest_tile_idx + offset) % num_tiles
            tile_indices_to_check.append(idx)

        # Check each wheel against nearby track tiles using polygon geometry
        for wheel_idx, wheel in enumerate(car.wheels):
            # Use actual wheel position (set in Car._update_hull())
            wheel_world_x = wheel.position[0]
            wheel_world_y = wheel.position[1]

            for tile_idx in tile_indices_to_check:
                tile = road_tiles[tile_idx]

                # Check if wheel center is inside tile polygon
                inside = self._point_in_polygon(wheel_world_x, wheel_world_y, tile.vertices)

                if not inside:
                    # Wheel is outside polygon - check if it's close to the edge
                    dist_to_edge = self._distance_to_polygon_edge(
                        wheel_world_x, wheel_world_y, tile.vertices
                    )
                    if dist_to_edge > NEAR_TRACK_THRESHOLD:
                        continue  # Wheel is too far from this tile

                # Wheel is on track (inside polygon or very close to edge)
                wheel.tiles.add(tile)

                # Get car ID (multi-car support)
                car_id = car.car_id if hasattr(car, 'car_id') else 0

                # Initialize per-car tracking on first use (multi-car support)
                if not hasattr(tile, 'visited_by_cars'):
                    tile.visited_by_cars = set()
                    tile.road_visited = False  # Keep for backward compatibility

                # Handle tile visitation (per-car tracking)
                if car_id not in tile.visited_by_cars:
                    tile.visited_by_cars.add(car_id)
                    tile.road_visited = True  # Backward compatibility

                    # Update per-car tile count
                    if self.env.num_cars > 1:
                        self.env.car_tile_visited_counts[car_id] += 1
                    else:
                        self.env.tile_visited_count += 1

                    # Update furthest tile reached (for progress tracking)
                    # Anti-exploit: Only update if car is moving forward (prevents backward driving exploits)
                    car_forward_velocity = car.vx if hasattr(car, 'vx') else 0.0
                    is_moving_forward = car_forward_velocity > 0.1  # Must be moving forward at > 0.1 m/s

                    if is_moving_forward and tile.idx > self.env.furthest_tile_idx:
                        self.env.furthest_tile_idx = tile.idx
                    elif not is_moving_forward and self.env.verbose and tile.idx > self.env.furthest_tile_idx:
                        # Debug: car reached new tile while moving backward
                        car_str = f"Car {car_id} " if self.env.num_cars > 1 else ""
                        print(f"  ⚠ {car_str}Tile {tile.idx} reached while moving BACKWARD "
                              f"(vx={car_forward_velocity:.2f} m/s) - NO PROGRESS UPDATE")

                    # Lap completion check (per-car)
                    if tile.idx == 0:
                        if self.env.num_cars > 1:
                            progress = self.env.car_tile_visited_counts[car_id] / len(self.env.track)
                        else:
                            progress = self.env.tile_visited_count / len(self.env.track)

                        if progress > self.lap_complete_percent:
                            self.env.new_lap = True


class CarRacing(gym.Env, EzPickle):
    """
    ## Description
    A top-down racing environment using vector state representation.
    The generated track is random every episode.

    This implementation uses a 71-dimensional vector state containing track geometry,
    car dynamics, and lookahead waypoints - optimized for efficient SAC training.

    Some indicators are shown at the bottom of the window when rendering.
    From left to right: true speed, four ABS sensors, steering wheel position, and gyroscope.

    Remember: it's a powerful rear-wheel drive car - don't press the accelerator
    and turn at the same time.

    ## Action Space
    Continuous action space with 2 actions:
    - 0: steering, -1 is full left, +1 is full right
    - 1: acceleration, -1 is full brake, +1 is full gas

    ## Observation Space

    Vector mode only:
    - 71-dimensional state vector with track geometry, car dynamics, and lookahead waypoints
    - Fast and informative representation optimized for training

    ## Rewards
    The reward structure uses continuous progress tracking with safe driving incentives.
    All reward parameters are configurable at the top of this file (lines 64-74).

    **Dense rewards (main objective and safe driving):**
    - Progress reward: +progress_delta × PROGRESS_REWARD_SCALE points for forward movement
      (default: PROGRESS_REWARD_SCALE = 2000, so full lap = +2000 points)
    - Progress measured as furthest tile reached / total tiles (0.0 to 1.0)
    - Only forward progress counts (backward movement = 0 reward)
    - Dense signal: reward every frame the car moves to a new furthest tile
    - Lap completion: +LAP_COMPLETION_REWARD bonus for completing full lap (default: 1000 points)
    - On-track reward: +ONTRACK_REWARD per frame when all wheels on track (default: +0.5)

    **Dense penalties (constraints and safety):**
    - Per-step penalty: -STEP_PENALTY every frame (default: -0.5, mild time pressure)
    - Stationary penalty: -STATIONARY_PENALTY every frame when speed < STATIONARY_SPEED_THRESHOLD
      (default: -1.0 when speed < 0.5 m/s, discourages staying still)
    - Off-track penalty: -OFFTRACK_PENALTY per wheel off-track per frame when ANY wheels off
      (default: -5.0 per wheel, strongly discourages off-track driving)
    - Off-track termination: -OFFTRACK_TERMINATION_PENALTY × (1.0 - progress) when all wheels go off track
      (default: -100 × progress_multiplier, proportional to how far the car progressed)
      Progress-based: crash at 0% = -100, crash at 50% = -50, crash at 90% = -10 (min 10%)

    **Optimal racing line:**
    The agent learns the optimal racing line (which uses the full width of the track) through
    progress rewards alone. No center-line reward is used, as this would penalize proper racing lines.

    Example with defaults: Reaching 50% progress in 500 frames while moving:
    - Progress reward: 0.5 × 2000 = +1000
    - Step penalty: -0.5 × 500 = -250
    - On-track reward: +0.5 × 500 = +250
    - Total: ~+1000 points (rewards safe, progressive driving!)

    Example with stationary car for 100 frames:
    - Progress reward: 0
    - Step penalty: -0.5 × 100 = -50
    - Stationary penalty: -1.0 × 100 = -100
    - Total: -150 points (strongly discourages staying still!)

    ## Starting State
    The car starts at rest in the center of the road.

    ## Episode Termination
    The episode finishes when all the tiles are visited. The car can also go outside the playfield -
     that is, far off the track (all 4 wheels off), in which case it will receive -OFFTRACK_TERMINATION_PENALTY × (1.0 - progress)
     reward and die (default: -100 × progress_multiplier, proportionally lower penalty for crashes further into the track).
     This encourages the agent to learn to go farther before crashing.

    Additionally, if `terminate_stationary=True`, episodes will be truncated early if the car makes
     no progress (no new tiles visited) for `stationary_patience` frames (default: 100), after a
     minimum of `stationary_min_steps` steps (default: 50). This prevents agents from learning to
     sit still and waste compute time.
"""
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
        ],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        verbose: bool = False,
        lap_complete_percent: float = 0.95,
        terminate_stationary: bool = True,
        stationary_patience: int = 50,
        stationary_min_steps: int = 50,
        state_mode: str = "vector",
        max_episode_steps: int | None = 2500,
        reward_shaping: bool = True,
        min_episode_steps: int = 150,
        short_episode_penalty: float = -50.0,
        num_cars: int = 1,
        domain_randomization_config: DomainRandomizationConfig | None = None,
    ):
        """
        Args:
            state_mode: Must be "vector" (71D state vector with track geometry and car dynamics).
                        This is the only supported mode for optimal performance and training.
            max_episode_steps: Maximum steps per episode (default: 1500). None for unlimited.
            reward_shaping: Apply penalty for short episodes (default: True)
            min_episode_steps: Minimum episode length before penalty (default: 150)
            short_episode_penalty: Penalty for episodes shorter than min_episode_steps (default: -50.0)
            num_cars: Number of cars racing simultaneously (default: 1). When >1, enables multi-car mode
                      where all cars race on the same track as ghost cars (no collision with each other).
            domain_randomization_config: Configuration for domain randomization (default: None, disabled).
                      Use config.domain_randomization presets or create custom DomainRandomizationConfig.
        """
        EzPickle.__init__(
            self,
            render_mode,
            verbose,
            lap_complete_percent,
            terminate_stationary,
            stationary_patience,
            stationary_min_steps,
            state_mode,
            max_episode_steps,
            reward_shaping,
            min_episode_steps,
            short_episode_penalty,
            num_cars,
            domain_randomization_config,
        )
        self.lap_complete_percent = lap_complete_percent
        self.terminate_stationary = terminate_stationary
        self.stationary_patience = stationary_patience
        self.stationary_min_steps = stationary_min_steps
        self.state_mode = state_mode
        self.max_episode_steps = max_episode_steps
        self.reward_shaping = reward_shaping
        self.min_episode_steps = min_episode_steps
        self.short_episode_penalty = short_episode_penalty
        self.num_cars = num_cars

        # Domain randomization
        if domain_randomization_config is None:
            domain_randomization_config = DomainRandomizationConfig(enabled=False)
        self.domain_randomization_config = domain_randomization_config
        self.domain_randomizer = DomainRandomizer(domain_randomization_config)

        self._init_colors()

        self.friction_detector = FrictionDetector(self, self.lap_complete_percent)
        # No Box2D world needed - using custom physics engine
        self.world = None  # Kept for compatibility with Car.__init__
        self.screen: pygame.Surface | None = None
        self.surf = None
        self.clock = None
        self.isopen = True
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None

        # Multi-car support: cars list + backward compatibility
        self.cars = []  # List of Car instances (multi-car mode)
        self.car: Car | None = None  # Single car reference (backward compatibility)

        # Per-car tracking (multi-car mode)
        self.car_rewards = []
        self.car_prev_rewards = []
        self.car_tile_visited_counts = []
        self.car_last_checkpoints = []
        self.car_frames_since_progress = []
        self.car_total_steps = []

        # Single-car tracking (backward compatibility)
        self.reward = 0.0
        self.prev_reward = 0.0

        self.verbose = verbose
        self.debug_step_counter = 0
        self.new_lap = False

        # Vector mode: waypoint lookahead count
        # Increased from 10 to 20 to allow braking at high speed
        # At 108 km/h (30 m/s), 20 waypoints = 70m = 2.33 seconds lookahead
        # This allows enough time to brake for corners (braking from 108→36 km/h needs ~41m)
        self.vector_lookahead = 20

        # Progress tracking for continuous rewards (configured at top of file)
        self.progress_reward_scale = PROGRESS_REWARD_SCALE

        # Continuous action space: [steering, acceleration]
        # steering: [-1, +1], acceleration: [-1 (brake), +1 (gas)]
        self.action_space = spaces.Box(
            np.array([-1, -1]).astype(np.float32),
            np.array([+1, +1]).astype(np.float32),
        )  # steer, acceleration

        # Observation space: vector mode only
        if self.state_mode != "vector":
            raise ValueError(f"Only state_mode='vector' is supported, got '{self.state_mode}'")

        # Vector state: car state (11) + track segment info (5) + lookahead waypoints (40)
        # + speed (1) + longitudinal accel (1) + lateral accel (1)
        # + slip angles (4) + slip ratios (4)
        # = 71 values total (increased from 47 to support 20 waypoint lookahead)
        if self.num_cars > 1:
            # Multi-car: return stacked observations for all cars
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.num_cars, 71), dtype=np.float32
            )
        else:
            # Single car (backward compatible)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(71,), dtype=np.float32
            )

        self.render_mode = render_mode

    def _destroy(self):
        if not self.road:
            return
        # Tiles are just objects now, no Box2D bodies to destroy
        self.road = []
        # Destroy all cars (multi-car support)
        for car in self.cars:
            if car is not None:
                car.destroy()
        self.cars = []
        self.car = None

    def _init_colors(self):
        # Default track colors
        self.road_color = np.array([102, 102, 102])
        self.bg_color = np.array([102, 204, 102])
        self.grass_color = np.array([102, 230, 102])

    def _get_car_color(self, car_idx):
        """Return distinct color for each car (for rendering)."""
        colors = [
            (0.8, 0.0, 0.0),  # Red
            (0.0, 0.0, 0.8),  # Blue
            (0.0, 0.8, 0.0),  # Green
            (0.8, 0.8, 0.0),  # Yellow
            (0.8, 0.0, 0.8),  # Magenta
            (0.0, 0.8, 0.8),  # Cyan
            (0.8, 0.4, 0.0),  # Orange
            (0.4, 0.0, 0.8),  # Purple
        ]
        return colors[car_idx % len(colors)]

    def _get_all_observations(self):
        """Get observations for all cars."""
        if self.num_cars == 1:
            return self._create_vector_state()
        else:
            observations = []
            for car in self.cars:
                # Temporarily set self.car for _create_vector_state()
                original_car = self.car
                self.car = car
                obs = self._create_vector_state()
                self.car = original_car
                observations.append(obs)
            return np.array(observations, dtype=np.float32)

    def _create_track(self):
        CHECKPOINTS = 12

        # Create checkpoints
        checkpoints = []
        for c in range(CHECKPOINTS):
            noise = self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
            alpha = 2 * math.pi * c / CHECKPOINTS + noise
            rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)

            if c == 0:
                alpha = 0
                rad = 1.5 * TRACK_RAD
            if c == CHECKPOINTS - 1:
                alpha = 2 * math.pi * c / CHECKPOINTS
                self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                rad = 1.5 * TRACK_RAD

            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5 * TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi

            while True:  # Find destination from checkpoints
                failed = True

                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break

                if not failed:
                    break

                alpha -= 2 * math.pi
                continue

            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            # destination vector projected on rad:
            proj = r1x * dest_dx + r1y * dest_dy
            while beta - alpha > 1.5 * math.pi:
                beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi:
                beta += 2 * math.pi
            prev_beta = beta
            proj *= SCALE
            if proj > 0.3:
                beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
            if proj < -0.3:
                beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
            x += p1x * TRACK_DETAIL_STEP
            y += p1y * TRACK_DETAIL_STEP
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze == 0:
                break

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False  # Failed
            pass_through_start = (
                track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha
            )
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        if self.verbose:
            print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2 - i1))
        assert i1 != -1
        assert i2 != -1

        track = track[i1 : i2 - 1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (track[0][2] - track[-1][2]))
            + np.square(first_perp_y * (track[0][3] - track[-1][3]))
        )
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        # Red-white border on hard turns
        border = [False] * len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i - neg - 0][1]
                beta2 = track[i - neg - 1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i - neg] |= border[i]

        # Create tiles (no Box2D, just geometry)
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            road1_l = (
                x1 - TRACK_WIDTH * math.cos(beta1),
                y1 - TRACK_WIDTH * math.sin(beta1),
            )
            road1_r = (
                x1 + TRACK_WIDTH * math.cos(beta1),
                y1 + TRACK_WIDTH * math.sin(beta1),
            )
            road2_l = (
                x2 - TRACK_WIDTH * math.cos(beta2),
                y2 - TRACK_WIDTH * math.sin(beta2),
            )
            road2_r = (
                x2 + TRACK_WIDTH * math.cos(beta2),
                y2 + TRACK_WIDTH * math.sin(beta2),
            )
            vertices = [road1_l, road1_r, road2_r, road2_l]

            # Create simple tile object (no Box2D)
            t = type('Tile', (), {})()
            t.userData = t
            c = 0.01 * (i % 3) * 255
            t.color = self.road_color + c
            t.road_visited = False
            t.road_friction = 1.0
            t.idx = i
            t.vertices = vertices
            # Cache tile center for fast collision detection
            t.center_x = (road1_l[0] + road1_r[0] + road2_r[0] + road2_l[0]) / 4.0
            t.center_y = (road1_l[1] + road1_r[1] + road2_r[1] + road2_l[1]) / 4.0

            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (
                    x1 + side * TRACK_WIDTH * math.cos(beta1),
                    y1 + side * TRACK_WIDTH * math.sin(beta1),
                )
                b1_r = (
                    x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1),
                    y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1),
                )
                b2_l = (
                    x2 + side * TRACK_WIDTH * math.cos(beta2),
                    y2 + side * TRACK_WIDTH * math.sin(beta2),
                )
                b2_r = (
                    x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2),
                    y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2),
                )
                self.road_poly.append(
                    (
                        [b1_l, b1_r, b2_r, b2_l],
                        (255, 255, 255) if i % 2 == 0 else (255, 0, 0),
                    )
                )
        self.track = track
        return True

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)
        self._destroy()
        # Recreate friction detector for new episode
        self.friction_detector = FrictionDetector(self, self.lap_complete_percent)

        # Initialize per-car tracking arrays
        self.car_rewards = [0.0] * self.num_cars
        self.car_prev_rewards = [0.0] * self.num_cars
        self.car_tile_visited_counts = [0] * self.num_cars
        self.car_last_checkpoints = [-1] * self.num_cars
        self.car_frames_since_progress = [0] * self.num_cars
        self.car_total_steps = [0] * self.num_cars

        # Single-car tracking (backward compatibility)
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.frames_since_progress = 0
        self.total_steps = 0
        self.last_checkpoint_reached = -1

        self.t = 0.0
        self.new_lap = False
        self.road_poly = []
        self.debug_step_counter = 0

        # Episode step counter for timeout and reward shaping
        self.episode_steps = 0

        # Previous velocity for acceleration computation (per-car)
        if self.num_cars > 1:
            self.car_prev_vx = [0.0] * self.num_cars
            self.car_prev_vy = [0.0] * self.num_cars
        else:
            self.prev_vx = 0.0
            self.prev_vy = 0.0

        # Progress tracking for continuous rewards
        self.furthest_tile_idx = 0  # Furthest tile index reached (for progress calculation)
        self.last_progress = 0.0    # Previous progress (0.0 to 1.0)

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose:
                print(
                    "retry to generate track (normal if there are not many"
                    "instances of this message)"
                )

        # Progress tracking initialized (no pre-computation needed)
        if self.verbose:
            print(f"Track has {len(self.track)} tiles for continuous progress tracking")

        init_beta, init_x, init_y = self.track[0][1:4]
        # The car's "front" is its +X axis in physics.
        # The track's "forward" direction is 90 degrees (pi/2) from its
        # normal (beta). We set the car's initial yaw to align these.
        init_yaw = init_beta + (math.pi / 2.0)

        # Create cars (multi-car support)
        self.cars = []
        for car_idx in range(self.num_cars):
            car = Car(self.world, init_yaw, init_x, init_y)
            car.car_id = car_idx  # Track which car this is
            # Assign different colors for rendering
            car.hull.color = self._get_car_color(car_idx)
            self.cars.append(car)

        # Maintain backward compatibility
        self.car = self.cars[0] if self.num_cars > 0 else None

        # Apply domain randomization
        if self.domain_randomization_config.enabled:
            randomized_params = self.domain_randomizer.randomize()

            # Apply to all cars
            for car in self.cars:
                self.domain_randomizer.apply_to_car(car, randomized_params)

            # Apply to track
            self.domain_randomizer.apply_to_track(self, randomized_params)

            if self.verbose:
                info = self.domain_randomizer.get_info_dict()
                print(f"\n=== Domain Randomization Applied ===")
                print(f"  Mass: {info.get('mass', 'N/A'):.1f} kg")
                print(f"  Lateral Grip (D_lat): {info.get('pacejka_d_lat', 'N/A'):.3f}")
                print(f"  Longitudinal Grip (D_lon): {info.get('pacejka_d_lon', 'N/A'):.3f}")
                print(f"  Surface Friction: {info.get('surface_friction', 'N/A'):.3f}")
                print(f"  Engine Power: {info.get('engine_power', 'N/A'):.0f} W")
                print(f"====================================\n")

        if self.render_mode == "human":
            self.render()

        # Return vectorized observations for multi-car, single for backward compatibility
        if self.num_cars > 1:
            return self._get_all_observations(), {}
        else:
            return self.step(None)[0], {}

    def step(self, action: np.ndarray | int | None):
        """
        Step environment with 1 or N cars.

        Args:
            action:
                - Single car: shape (2,) or int or None
                - Multi car: shape (num_cars, 2) or array of ints

        Returns:
            observations: shape (num_cars, 71) if multi-car, else (71,)
            rewards: shape (num_cars,) if multi-car, else scalar
            terminated: shape (num_cars,) if multi-car, else bool
            truncated: shape (num_cars,) if multi-car, else bool
            infos: list of dicts if multi-car, else dict
        """
        if self.num_cars == 1:
            # Single car mode (backward compatible)
            return self._step_single_car(action)
        else:
            # Multi-car mode
            return self._step_multi_car(action)

    def _step_single_car(self, action: np.ndarray | int | None):
        """Original single-car step logic (backward compatible)."""
        assert self.car is not None

        # Start timing for verbose mode
        step_start_time = time.perf_counter() if self.verbose else None

        # Initialize action vars for debug print
        gas, brake = 0.0, 0.0
        steer_action = 0.0
        accel = 0.0

        if action is not None:
            action = action.astype(np.float64)
            # Actions: steering [-1, 1], acceleration [-1 (brake), +1 (gas)]
            steer_action = -action[0]
            accel = np.clip(action[1], -1.0, 1.0)

            # Convert acceleration to gas/brake
            if accel > 0:
                gas = accel
                brake = 0.0
            else:
                gas = 0.0
                brake = -accel

            self.car.steer(steer_action)
            self.car.gas(gas)
            self.car.brake(brake)

        # Step custom physics engine and get debug info
        physics_start = time.perf_counter() if self.verbose else None
        debug_info = self.car.step(1.0 / FPS)
        physics_time = (time.perf_counter() - physics_start) * 1000 if self.verbose else None

        # Update wheel-tile contacts for friction computation
        collision_start = time.perf_counter() if self.verbose else None
        self.friction_detector.update_contacts(self.car, self.road)
        collision_time = (time.perf_counter() - collision_start) * 1000 if self.verbose else None
        self.t += 1.0 / FPS

        # Create vector state
        state_start = time.perf_counter() if self.verbose else None
        self.state = self._create_vector_state()
        state_time = (time.perf_counter() - state_start) * 1000 if self.verbose else None

        step_reward = 0
        terminated = False
        truncated = False
        info = {}
        speed = 0.0
        forward_velocity = 0.0
        wheels_off_track = 0

        if action is not None:  # First step without action, called from reset()
            self.reward -= STEP_PENALTY  # Time penalty (encourages speed)

            # Calculate speed magnitude (for stationary detection and debug output)
            speed = np.sqrt(
                self.car.hull.linearVelocity[0] ** 2 +
                self.car.hull.linearVelocity[1] ** 2
            )

            # Calculate forward velocity (project velocity onto car's forward direction)
            # Car's forward direction is its heading angle
            car_forward_x = np.cos(self.car.hull.angle)
            car_forward_y = np.sin(self.car.hull.angle)

            # Dot product of velocity with forward direction (kept for debug output)
            forward_velocity = (
                self.car.hull.linearVelocity[0] * car_forward_x +
                self.car.hull.linearVelocity[1] * car_forward_y
            )

            # Continuous penalty for wheels off track (no sharp boundaries to exploit)
            wheels_off_track = sum(1 for wheel in self.car.wheels if len(wheel.tiles) == 0)
            if wheels_off_track > OFFTRACK_THRESHOLD:
                self.reward -= OFFTRACK_PENALTY * wheels_off_track

            # Stationary penalty (discourages staying still)
            if speed < STATIONARY_SPEED_THRESHOLD:
                self.reward -= STATIONARY_PENALTY

            # Continuous progress reward (dense signal for forward movement)
            current_progress = self.furthest_tile_idx / len(self.track)
            progress_delta = max(0, current_progress - self.last_progress)  # Only forward progress
            if progress_delta > 0:
                progress_reward = progress_delta * self.progress_reward_scale
                self.reward += progress_reward
                self.last_progress = current_progress

                if self.verbose and progress_delta > 0.01:  # Print every ~1% progress
                    progress_pct = current_progress * 100
                    print(f"  → Progress: {progress_pct:.1f}% (+{progress_reward:.1f} reward)")

            # Reward for staying on track (all wheels on track)
            if wheels_off_track <= OFFTRACK_THRESHOLD:
                self.reward += ONTRACK_REWARD

            # Speed reward removed - progress is the main reward
            # This prevents the agent from learning to just go fast without caring about staying on track
            # The agent should learn the optimal racing line (using full track width) through progress rewards
            if FORWARD_SPEED_REWARD_SCALE > 0:
                car_forward_velocity = self.car.hull.linearVelocity[0]
                if car_forward_velocity > 0:
                    speed_reward = min(car_forward_velocity * FORWARD_SPEED_REWARD_SCALE, 2.0)
                    self.reward += speed_reward

            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward

            # Track stationary car (for early termination)
            if self.terminate_stationary:
                self.total_steps += 1

                # Check if car made progress:
                # 1. Visited new tile (step_reward > 0), OR
                # 2. Moving with meaningful velocity (speed > 0.5 m/s in any direction)
                is_making_progress = (step_reward > 0) or (speed > 0.5)

                if is_making_progress:
                    self.frames_since_progress = 0
                else:
                    self.frames_since_progress += 1

                # Terminate early if:
                # 1. We've taken enough steps (min_steps)
                # 2. No progress for 'patience' frames
                if (self.total_steps >= self.stationary_min_steps and
                        self.frames_since_progress >= self.stationary_patience):
                    truncated = True
                    info['stationary_termination'] = True

            if self.tile_visited_count == len(self.track) or self.new_lap:
                # Termination due to finishing lap
                # Award generous bonus for lap completion (encourages fast lap times)
                self.reward += LAP_COMPLETION_REWARD
                terminated = True
                info["lap_finished"] = True
                info["lap_completion_bonus"] = LAP_COMPLETION_REWARD

            # Check if all 4 wheels are off track
            all_wheels_off_track = all(len(wheel.tiles) == 0 for wheel in self.car.wheels)
            if all_wheels_off_track:
                terminated = True
                info["lap_finished"] = False
                info["off_track"] = True
                # Proportional penalty: lower penalty if more progress was made
                # Farther crashes are rewarded with smaller penalties
                current_progress = self.furthest_tile_idx / len(self.track) if len(self.track) > 0 else 0.0
                progress_multiplier = max(0.1, 1.0 - current_progress)  # Min 10% penalty even at 100% progress
                step_reward = -OFFTRACK_TERMINATION_PENALTY * progress_multiplier
                info["off_track_penalty"] = -OFFTRACK_TERMINATION_PENALTY * progress_multiplier
                info["progress_at_crash"] = current_progress

            # Built-in timeout logic (replaces TimeLimit wrapper for vector mode)
            self.episode_steps += 1
            if self.max_episode_steps is not None and self.episode_steps >= self.max_episode_steps:
                truncated = True
                info['TimeLimit.truncated'] = True

            # Built-in reward shaping (replaces RewardShaper wrapper for vector mode)
            if self.reward_shaping and (terminated or truncated):
                if self.episode_steps < self.min_episode_steps:
                    step_reward += self.short_episode_penalty
                    info['reward_shaping'] = self.short_episode_penalty
                    info['original_step_reward'] = step_reward - self.short_episode_penalty

        # Debug output with timing
        if self.verbose and action is not None and self.debug_step_counter % 10 == 0:
            # Calculate total step time so far (before rendering)
            step_time_so_far = (time.perf_counter() - step_start_time) * 1000

            print(f"\n{'='*70}")
            print(f"--- STEP {self.debug_step_counter} ---")
            print(f"  ACTION:  Gas={gas:0.2f}, Brake={brake:0.2f}, Steer={steer_action:0.2f}")
            print(f"  CAR:     vx={self.car.vx: >6.2f} (long), vy={self.car.vy: >6.2f} (lat), yaw_rate={self.car.yaw_rate: >6.2f}")
            print(f"  WORLD:   Speed={speed: >6.2f} m/s, ForwardVel={forward_velocity: >6.2f} m/s")
            print(f"  PHYSICS: Fx={debug_info['fx_total']: >8.2f}, Fy={debug_info['fy_total']: >8.2f}, Torque={debug_info['torque']: >8.2f}")
            print(f"  REWARD:  WheelsOff={wheels_off_track}, StepRwd={step_reward: >6.2f}, TotalRwd={self.reward: >8.2f}")
            print(f"  TIRES (SlipRatio, SlipAngle):")
            f_fl = debug_info['tire_forces'][0]
            f_fr = debug_info['tire_forces'][1]
            f_rl = debug_info['tire_forces'][2]
            f_rr = debug_info['tire_forces'][3]
            print(f"    FL: SR={f_fl['slip_ratio']: >5.2f}, SA={f_fl['slip_angle']: >5.2f} | FR: SR={f_fr['slip_ratio']: >5.2f}, SA={f_fr['slip_angle']: >5.2f}")
            print(f"    RL: SR={f_rl['slip_ratio']: >5.2f}, SA={f_rl['slip_angle']: >5.2f} | RR: SR={f_rr['slip_ratio']: >5.2f}, SA={f_rr['slip_angle']: >5.2f}")
            print(f"\n  TIMING:")
            print(f"    Physics step:      {physics_time:>7.2f} ms")
            print(f"    Collision detect:  {collision_time:>7.2f} ms")
            print(f"    State creation:    {state_time:>7.2f} ms")
            print(f"    Other logic:       {step_time_so_far - physics_time - collision_time - state_time:>7.2f} ms")
            print(f"    TOTAL (so far):    {step_time_so_far:>7.2f} ms")
            print(f"{'='*70}\n")

        # Increment debug counter for next step
        if action is not None:
            self.debug_step_counter += 1

        # Rendering (if needed)
        render_start = time.perf_counter() if (self.verbose and self.render_mode == "human") else None
        if self.render_mode == "human":
            self.render()
            if self.verbose and action is not None and (self.debug_step_counter - 1) % 10 == 0:
                render_time = (time.perf_counter() - render_start) * 1000
                total_step_time = (time.perf_counter() - step_start_time) * 1000
                print(f"  RENDER TIME: {render_time:>7.2f} ms")
                print(f"  TOTAL STEP:  {total_step_time:>7.2f} ms\n")

        return self.state, step_reward, terminated, truncated, info

    def _step_multi_car(self, action: np.ndarray | None):
        """
        Step all cars in parallel on the same track.

        Args:
            action: shape (num_cars, 2) or (num_cars,) - actions for all cars

        Returns:
            observations: (num_cars, 71)
            rewards: (num_cars,)
            terminated: (num_cars,)
            truncated: (num_cars,)
            infos: list[dict]
        """
        # Validate action shape
        if action is not None:
            assert action.shape == (self.num_cars, 2), \
                f"Expected action shape ({self.num_cars}, 2), got {action.shape}"

        step_rewards = np.zeros(self.num_cars, dtype=np.float32)
        terminated = np.zeros(self.num_cars, dtype=bool)
        truncated = np.zeros(self.num_cars, dtype=bool)
        infos = [{} for _ in range(self.num_cars)]

        # Step each car independently
        for car_idx, car in enumerate(self.cars):
            if action is not None:
                car_action = action[car_idx]

                # Apply continuous action to this car
                steer_action = -car_action[0]
                accel = np.clip(car_action[1], -1.0, 1.0)
                gas = accel if accel > 0 else 0.0
                brake = -accel if accel < 0 else 0.0
                car.steer(steer_action)
                car.gas(gas)
                car.brake(brake)

                # Step physics for this car
                car.step(1.0 / FPS)

                # Update contacts for this car ONLY (no car-car collision)
                # Temporarily set self.car for friction_detector
                original_car = self.car
                self.car = car
                self.friction_detector.update_contacts(car, self.road)
                self.car = original_car

                # Calculate reward for this car
                self.car_rewards[car_idx] -= STEP_PENALTY

                # Off-track penalty
                wheels_off = sum(1 for wheel in car.wheels if len(wheel.tiles) == 0)
                if wheels_off > OFFTRACK_THRESHOLD:
                    self.car_rewards[car_idx] -= OFFTRACK_PENALTY * wheels_off

                # Stationary penalty (discourages staying still)
                speed = np.sqrt(
                    car.hull.linearVelocity[0] ** 2 +
                    car.hull.linearVelocity[1] ** 2
                )
                if speed < STATIONARY_SPEED_THRESHOLD:
                    self.car_rewards[car_idx] -= STATIONARY_PENALTY

                # Check termination conditions for this car
                all_wheels_off = all(len(wheel.tiles) == 0 for wheel in car.wheels)
                if all_wheels_off:
                    terminated[car_idx] = True
                    infos[car_idx]["off_track"] = True
                    # Proportional penalty: lower penalty if more progress was made
                    # Farther crashes are rewarded with smaller penalties
                    current_progress = self.car_tile_visited_counts[car_idx] / len(self.track) if len(self.track) > 0 else 0.0
                    progress_multiplier = max(0.1, 1.0 - current_progress)  # Min 10% penalty even at 100% progress
                    step_rewards[car_idx] = -OFFTRACK_TERMINATION_PENALTY * progress_multiplier
                    infos[car_idx]["off_track_penalty"] = -OFFTRACK_TERMINATION_PENALTY * progress_multiplier
                    infos[car_idx]["progress_at_crash"] = current_progress

                # Check lap completion for this car
                if self.car_tile_visited_counts[car_idx] == len(self.track):
                    self.car_rewards[car_idx] += LAP_COMPLETION_REWARD
                    terminated[car_idx] = True
                    infos[car_idx]["lap_finished"] = True
                    infos[car_idx]["lap_completion_bonus"] = LAP_COMPLETION_REWARD

                # Track stationary car (for early termination)
                if self.terminate_stationary:
                    self.car_total_steps[car_idx] += 1

                    # Calculate speed for progress check
                    speed = np.sqrt(
                        car.hull.linearVelocity[0] ** 2 +
                        car.hull.linearVelocity[1] ** 2
                    )

                    # Check if car made progress
                    prev_tiles = self.car_tile_visited_counts[car_idx]
                    # Note: tiles are updated by friction_detector
                    current_tiles = self.car_tile_visited_counts[car_idx]
                    is_making_progress = (current_tiles > prev_tiles) or (speed > 0.5)

                    if is_making_progress:
                        self.car_frames_since_progress[car_idx] = 0
                    else:
                        self.car_frames_since_progress[car_idx] += 1

                    # Terminate early if stationary for too long
                    if (self.car_total_steps[car_idx] >= self.stationary_min_steps and
                            self.car_frames_since_progress[car_idx] >= self.stationary_patience):
                        truncated[car_idx] = True
                        infos[car_idx]['stationary_termination'] = True

                # Step reward for this car
                step_rewards[car_idx] = self.car_rewards[car_idx] - self.car_prev_rewards[car_idx]
                self.car_prev_rewards[car_idx] = self.car_rewards[car_idx]

        # Update time
        self.t += 1.0 / FPS
        self.episode_steps += 1

        # Check max episode steps (applies to all cars)
        if self.max_episode_steps is not None and self.episode_steps >= self.max_episode_steps:
            truncated[:] = True
            for info in infos:
                info['TimeLimit.truncated'] = True

        # Apply reward shaping (per car)
        if self.reward_shaping and action is not None:
            for car_idx in range(self.num_cars):
                if (terminated[car_idx] or truncated[car_idx]):
                    if self.episode_steps < self.min_episode_steps:
                        step_rewards[car_idx] += self.short_episode_penalty
                        infos[car_idx]['reward_shaping'] = self.short_episode_penalty

        # Get observations for all cars
        observations = self._get_all_observations()

        # Render if needed
        if self.render_mode == "human":
            self.render()

        return observations, step_rewards, terminated, truncated, infos

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        else:
            return self._render(self.render_mode)

    def _find_closest_track_segment(self, car_pos):
        """
        Find the track segment closest to the car.
        Returns: (segment_index, distance_to_segment, closest_point_on_segment)
        """
        min_dist = float('inf')
        closest_idx = 0
        closest_point = None

        car_x, car_y = car_pos[0], car_pos[1]

        for i in range(len(self.track)):
            # Get segment endpoints
            _, beta1, x1, y1 = self.track[i]
            _, beta2, x2, y2 = self.track[i - 1]

            # Vector from segment start to end
            seg_dx = x2 - x1
            seg_dy = y2 - y1
            seg_len_sq = seg_dx**2 + seg_dy**2

            if seg_len_sq < 1e-6:
                # Degenerate segment
                dist = np.sqrt((car_x - x1)**2 + (car_y - y1)**2)
                point = (x1, y1)
            else:
                # Project car position onto segment
                t = max(0, min(1, ((car_x - x1) * seg_dx + (car_y - y1) * seg_dy) / seg_len_sq))
                proj_x = x1 + t * seg_dx
                proj_y = y1 + t * seg_dy
                dist = np.sqrt((car_x - proj_x)**2 + (car_y - proj_y)**2)
                point = (proj_x, proj_y)

            if dist < min_dist:
                min_dist = dist
                closest_idx = i
                closest_point = point

        return closest_idx, min_dist, closest_point

    def _create_vector_state(self):
        """
        Create vector state representation (fast, informative).

        Returns 71-dimensional state vector (increased from 47 for better lookahead):
        - Car state (11): x, y, vx (body), vy (body), angle, angular_vel, wheel_contacts[4], track_progress
        - Track segment info (5): dist_to_center, angle_diff, curvature, t (position on segment [0-1]), segment_length
        - Lookahead waypoints (40): 20 waypoints × (x, y) in car-relative coordinates (increased from 10)
        - Speed (1): magnitude of velocity
        - Accelerations (2): longitudinal (body frame), lateral (body frame)
        - Slip angles (4): for each wheel [FL, FR, RL, RR]
        - Slip ratios (4): for each wheel [FL, FR, RL, RR]

        Note: ALL velocities and accelerations use BODY FRAME (longitudinal/lateral relative to car)
        for consistent coordinate system. This is more intuitive for racing control.

        Lookahead increased from 10 to 20 waypoints to allow high-speed braking.
        At 108 km/h, 20 waypoints = 70m = 2.33s lookahead (enough to brake for corners).
        """
        assert self.car is not None

        # 1. Get basic car state
        car_x = self.car.hull.position[0] / PLAYFIELD
        car_y = self.car.hull.position[1] / PLAYFIELD
        # Use body frame velocities (longitudinal/lateral) for consistency with accelerations
        vx = self.car.vx  # Longitudinal velocity (forward/backward in car frame)
        vy = self.car.vy  # Lateral velocity (left/right in car frame)
        angle = self.car.hull.angle / (2 * np.pi)
        angular_vel = self.car.hull.angularVelocity
        wheel_contacts = [1.0 if len(wheel.tiles) > 0 else 0.0 for wheel in self.car.wheels]
        track_progress = self.tile_visited_count / len(self.track) if len(self.track) > 0 else 0.0

        # Calculate speed (magnitude of velocity in body frame)
        speed = np.sqrt(vx ** 2 + vy ** 2)

        # Calculate accelerations in body frame (change in velocity)
        # Note: For the first step, prev_vx/vy are 0, so acceleration will be 0
        dt = 1.0 / FPS

        # Get true accelerations from the physics engine (body frame)
        ax = self.car.ax  # Longitudinal acceleration from car_dynamics
        ay = self.car.ay  # Lateral acceleration from car_dynamics

        # 2. Get track segment info
        car_world_pos = self.car.hull.position
        seg_idx, dist_to_center, closest_point = self._find_closest_track_segment(car_world_pos)

        # Get segment direction
        _, beta, x, y = self.track[seg_idx]
        track_angle = beta

        # Angle difference (normalized)
        angle_diff = (self.car.hull.angle - track_angle) / (2 * np.pi)
        # Wrap to [-0.5, 0.5]
        while angle_diff > 0.5:
            angle_diff -= 1.0
        while angle_diff < -0.5:
            angle_diff += 1.0

        # Calculate curvature (change in angle over distance)
        prev_idx = (seg_idx - 1) % len(self.track)
        next_idx = (seg_idx + 1) % len(self.track)
        prev_beta = self.track[prev_idx][1]
        next_beta = self.track[next_idx][1]
        curvature = (next_beta - prev_beta) / (2 * TRACK_DETAIL_STEP)

        # Distance along segment (normalized to [0, 1])
        _, _, x1, y1 = self.track[seg_idx]
        _, _, x2, y2 = self.track[prev_idx]
        seg_len = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if seg_len > 0:
            # t is the projection parameter [0, 1] along the segment
            t = ((car_world_pos[0] - x1) * (x2 - x1) + (car_world_pos[1] - y1) * (y2 - y1)) / (seg_len**2)
            # Clamp t to [0, 1] to keep it bounded
            t = np.clip(t, 0.0, 1.0)
        else:
            t = 0.0

        # Normalize distance to center by track width (0 = center, 1 = edge, >1 = off track)
        dist_to_center_norm = np.clip(dist_to_center / TRACK_WIDTH, 0.0, 2.0)  # Clip at 2x track width

        # Normalize segment length by typical track detail step
        seg_len_norm = seg_len / TRACK_DETAIL_STEP

        # 3. Get slip angles and slip ratios from stored tire forces
        # IMPORTANT: Use stored forces from car.last_tire_forces to avoid
        # recomputation which causes oscillations in RL input due to
        # tire force feedback loop
        slip_angles = []
        slip_ratios = []
        vertical_forces = []

        if hasattr(self.car, 'last_tire_forces') and self.car.last_tire_forces is not None:
            # Use pre-computed values (consistent with physics simulation)
            forces = self.car.last_tire_forces
            for i in range(4):
                if i in forces:
                    slip_angles.append(forces[i].get('slip_angle', 0.0))
                    slip_ratios.append(forces[i].get('slip_ratio', 0.0))
                    vertical_forces.append(forces[i].get('normal_force', 0.0))  # <-- ADD THIS
                else:
                    slip_angles.append(0.0)
                    slip_ratios.append(0.0)
                    vertical_forces.append(0.0)  # <-- ADD THIS
        else:
            # Fallback: compute if not available (shouldn't happen in normal operation)
            for i in range(4):
                wheel = self.car.wheels[i]

                # Wheel position relative to CG
                if i < 2:  # Front wheels
                    dist_cg = self.car.LF
                    y_pos = self.car.WIDTH / 2 if i == 0 else -self.car.WIDTH / 2
                    steer_ang = self.car.steering_angle
                else:  # Rear wheels
                    dist_cg = -self.car.LR
                    y_pos = self.car.WIDTH / 2 if i == 2 else -self.car.WIDTH / 2
                    steer_ang = 0.0

                # Velocity at wheel contact point (body frame)
                wheel_vx = self.car.vx - self.car.yaw_rate * y_pos
                wheel_vy = self.car.vy + self.car.yaw_rate * dist_cg

                # Slip angle (angle between tire heading and velocity)
                v_mag = np.sqrt(wheel_vx ** 2 + wheel_vy ** 2)
                if v_mag > 0.5:
                    slip_angle = np.arctan2(wheel_vy, wheel_vx + 1e-6) - steer_ang
                else:
                    slip_angle = 0.0

                # Slip ratio (wheel vs ground)
                wheel_linear_vel = wheel.omega * self.car.TIRE_RADIUS
                denom = max(abs(wheel_vx), abs(wheel_linear_vel))

                if denom < 0.1:  # If both speeds are near zero, there is no slip
                    slip_ratio = 0.0
                else:
                    slip_ratio = (wheel_linear_vel - wheel_vx) / denom

                slip_angles.append(slip_angle)
                slip_ratios.append(slip_ratio)
                vertical_forces.append(0.0)  # <-- ADD THIS (fallback will be 0)

        # 4. Get next N waypoints in car-relative coordinates
        # Vectorized computation for all 20 waypoints at once (20-30% faster)
        car_angle_rad = self.car.hull.angle
        cos_a = np.cos(car_angle_rad)
        sin_a = np.sin(car_angle_rad)

        # Compute all waypoint indices at once
        wp_indices = (seg_idx + np.arange(1, self.vector_lookahead + 1)) % len(self.track)
        wp_coords = np.array([self.track[i][2:4] for i in wp_indices])  # Extract (x, y) for all waypoints

        # Transform all waypoints to car-relative coordinates at once
        dx_dy = wp_coords - car_world_pos

        # Rotate all waypoints into car's frame using rotation matrix
        rel_coords = np.column_stack([
            dx_dy[:, 0] * cos_a + dx_dy[:, 1] * sin_a,
            -dx_dy[:, 0] * sin_a + dx_dy[:, 1] * cos_a
        ])

        # Normalize and flatten to list
        waypoints = (rel_coords / PLAYFIELD).flatten().tolist()

        # Normalization constants for better training stability
        MAX_VELOCITY = 30.0  # m/s (typical max speed ~25-30 m/s)
        MAX_ANGULAR_VEL = 5.0  # rad/s (typical max ~3-4 rad/s)
        MAX_ACCELERATION = 50.0  # m/s^2 (typical max ~30-40 m/s^2)
        MAX_CURVATURE = 1.0  # 1/m (typical sharp turn)
        MAX_SLIP_RATIO = 2.0  # Dimensionless (clip extreme values)
        MAX_VERTICAL_FORCE = 5000.0 # (approx 1000kg car * 1.5g) but it's PER WHEEL + some safety margin

        # Normalize velocities
        vx_norm = vx / MAX_VELOCITY
        vy_norm = vy / MAX_VELOCITY
        angular_vel_norm = angular_vel / MAX_ANGULAR_VEL

        # Normalize speed
        speed_norm = speed / MAX_VELOCITY

        # Normalize accelerations
        ax_norm = ax / MAX_ACCELERATION
        ay_norm = ay / MAX_ACCELERATION

        # Normalize curvature
        curvature_norm = curvature / MAX_CURVATURE

        # Normalize slip angles (from radians [-π, π] to [-1, 1])
        slip_angles_norm = [sa / np.pi for sa in slip_angles]

        # Normalize and clip slip ratios
        for sr in slip_ratios:
            if sr > 2.0:
                print("WARNING: SLIP RATIO > 2.0: {}".format(sr))
            elif sr < -2.0:
                print("WARNING: SLIP RATIO < -2.0")
        slip_ratios_norm = [sr / MAX_SLIP_RATIO for sr in slip_ratios]
#        slip_ratios_norm = [np.clip(sr / MAX_SLIP_RATIO, -1.0, 1.0) for sr in slip_ratios]

        # Normalize vertical forces <-- ADD THIS BLOCK
        for vf in vertical_forces:
            if vf > MAX_VERTICAL_FORCE * 2:
                print("WARNING: VERTICAL FORCE > MAX*2: {}".format(vf))
            elif vf < -MAX_VERTICAL_FORCE * 2:
                print("WARNING: VERTICAL FORCE < MAX*2")
        vertical_forces_norm = [vf / MAX_VERTICAL_FORCE for vf in vertical_forces]

        # ... right after you calculate slip_ratios_norm and vertical_forces_norm

        # New, more robust check
        if np.isnan(slip_ratios_norm).any() or np.isnan(vertical_forces_norm).any():
            print("=" * 50)
            print("WARNING: NaN DETECTED IN STATE VECTOR!")
            print(f"Slip Ratios: {slip_ratios_norm}")
            print(f"Vertical Forces: {vertical_forces_norm}")
            print("=" * 50)
            # This is what's causing the training collapse.
            # You should probably clip these values.

        if np.isinf(slip_ratios_norm).any() or np.isinf(vertical_forces_norm).any():
            print("=" * 50)
            print("WARNING: Inf DETECTED IN STATE VECTOR!")
            print(f"Slip Ratios: {slip_ratios_norm}")
            print(f"Vertical Forces: {vertical_forces_norm}")
            print("=" * 50)

        # Combine all features (71 total, increased from 47)
        # ALL FEATURES NOW NORMALIZED to similar scales for stable training
        # ALL VELOCITIES AND ACCELERATIONS IN BODY FRAME (consistent coordinate system)
        state = np.array([
            # Basic car state (11) - NORMALIZED, body frame velocities
            car_x, car_y, vx_norm, vy_norm, angle, angular_vel_norm,
            wheel_contacts[0], wheel_contacts[1], wheel_contacts[2], wheel_contacts[3],
            track_progress,
            # Track segment info (5) - NORMALIZED & BOUNDED
            dist_to_center_norm, angle_diff, curvature_norm, t, seg_len_norm,
            # Waypoints (40) - NORMALIZED
            *waypoints,
            # Speed (1) - NORMALIZED
            speed_norm,
            # Accelerations (2) - NORMALIZED, body frame
            ax_norm, ay_norm,
            # Slip angles (4) - NORMALIZED
            slip_angles_norm[0], slip_angles_norm[1], slip_angles_norm[2], slip_angles_norm[3],
            # Slip ratios (4) - NORMALIZED
            slip_ratios_norm[0], slip_ratios_norm[1], slip_ratios_norm[2], slip_ratios_norm[3],
            # Vertical forces (4) - NORMALIZED
            vertical_forces_norm[0], vertical_forces_norm[1], vertical_forces_norm[2], vertical_forces_norm[3]
        ], dtype=np.float32)

        return state

    def _render(self, mode: str):
        assert mode in self.metadata["render_modes"]

        pygame.font.init()
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))

        assert self.car is not None
        # Computing transformations
        angle = -self.car.hull.angle + (math.pi / 2.0)
        # Use constant zoom (no animation)
        zoom = ZOOM * SCALE
        scroll_x = -(self.car.hull.position[0]) * zoom
        scroll_y = -(self.car.hull.position[1]) * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])

        self._render_road(zoom, trans, angle)
        # Draw car (simplified for new physics engine, no Box2D)
        self._render_car(zoom, trans, angle, draw_particles=True)

        self.surf = pygame.transform.flip(self.surf, False, True)

        # showing stats
        self._render_indicators(WINDOW_W, WINDOW_H)

        font = pygame.font.Font(pygame.font.get_default_font(), 42)
        text = font.render("%04i" % self.reward, True, (255, 255, 255), (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (60, WINDOW_H - WINDOW_H * 2.5 / 40.0)
        self.surf.blit(text, text_rect)

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
        elif mode == "rgb_array":
            return self._create_image_array(self.surf, (VIDEO_W, VIDEO_H))
        else:
            return self.isopen

    def _render_car(self, zoom, translation, angle, draw_particles=True):
        """
        Render car(s) for full rendering. Handles both single and multi-car mode.
        """
        if self.num_cars == 1:
            # Single car mode (backward compatible)
            assert self.car is not None
            self._render_single_car(self.car, zoom, translation, angle, draw_particles)
        else:
            # Multi-car mode: render all cars
            for car in self.cars:
                self._render_single_car(car, zoom, translation, angle, draw_particles)

    def _render_single_car(self, car, zoom, translation, angle, draw_particles=True):
        """
        Render a single car and its wheels (compatible with new physics engine).
        Draws the car body AND the four wheels, rotating the front wheels with steering.
        """
        # 1. DRAW CAR BODY
        car_x, car_y = car.hull.position
        car_world_angle = car.hull.angle

        # Car corners in local frame (MATCHING PHYSICS COORDS)
        front_x = car.LF
        rear_x = -car.LR
        left_y = car.WIDTH / 2
        right_y = -car.WIDTH / 2

        corners = [
            (front_x, left_y),  # FL
            (front_x, right_y),  # FR
            (rear_x, right_y),  # RR
            (rear_x, left_y)  # RL
        ]

        # Hood/windshield polygon (triangle on the front)
        hood_corners = [
            (front_x, left_y * 0.7),  # Mid-left
            (front_x, right_y * 0.7),  # Mid-right
            (front_x * 0.5, 0.0)  # Point halfway to center
        ]

        # Rotate corners to world orientation and translate to world position
        cos_a = np.cos(car_world_angle)
        sin_a = np.sin(car_world_angle)

        body_poly = []
        for dx, dy in corners:
            # Rotate
            x = dx * cos_a - dy * sin_a
            y = dx * sin_a + dy * cos_a
            # Translate
            body_poly.append((car_x + x, car_y + y))

        # Transform hood polygon to world frame
        hood_poly = []
        for dx, dy in hood_corners:
            x = dx * cos_a - dy * sin_a
            y = dx * sin_a + dy * cos_a
            hood_poly.append((car_x + x, car_y + y))

        # Apply camera transform (rotate by camera angle, zoom, translate)
        body_poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in body_poly]
        body_poly = [(c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in body_poly]

        # Transform hood to screen coordinates
        hood_poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in hood_poly]
        hood_poly = [(c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in hood_poly]

        # Draw car body
        color = [int(c * 255) for c in car.hull.color]
        gfxdraw.filled_polygon(self.surf, body_poly, color)

        # Draw hood
        gfxdraw.filled_polygon(self.surf, hood_poly, (200, 200, 200))  # Light grey

        # 2. DRAW WHEELS
        tire_length = car.TIRE_RADIUS * 2  # Length of the tire
        tire_width = car.TIRE_WIDTH
        hw = tire_width / 2  # half-width
        hl = tire_length / 2  # half-length (radius)

        # Define wheel corners in local frame (centered at 0,0)
        # Length (hl) along X-axis, Width (hw) along Y-axis
        wheel_corners_local = [
            (hl, hw),  # Front-Left
            (hl, -hw),  # Front-Right
            (-hl, -hw),  # Rear-Right
            (-hl, hw)  # Rear-Left
        ]

        for i, wheel in enumerate(car.wheels):
            # Get wheel's world position (from dynamics)
            wx, wy = wheel.position

            # Get wheel's world angle
            wheel_world_angle = car.hull.angle
            if i < 2:  # Front wheels
                wheel_world_angle += car.steering_angle

            # Rotate corners to world orientation and translate to world position
            cos_w = np.cos(wheel_world_angle)
            sin_w = np.sin(wheel_world_angle)

            wheel_poly_world = []
            for dx, dy in wheel_corners_local:
                # Rotate
                rx = dx * cos_w - dy * sin_w
                ry = dx * sin_w + dy * cos_w
                # Translate
                wheel_poly_world.append((wx + rx, wy + ry))

            # Apply camera transform (same as body)
            wheel_poly_screen = [pygame.math.Vector2(c).rotate_rad(angle) for c in wheel_poly_world]
            wheel_poly_screen = [(c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in
                                 wheel_poly_screen]

            # Draw the wheel (black)
            gfxdraw.filled_polygon(self.surf, wheel_poly_screen, (0, 0, 0))

    def _render_road(self, zoom, translation, angle):
        bounds = PLAYFIELD
        field = [
            (bounds, bounds),
            (bounds, -bounds),
            (-bounds, -bounds),
            (-bounds, bounds),
        ]

        # draw background
        self._draw_colored_polygon(
            self.surf, field, self.bg_color, zoom, translation, angle, clip=False
        )

        # draw grass patches
        grass = []
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                grass.append(
                    [
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + GRASS_DIM),
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + GRASS_DIM),
                    ]
                )
        for poly in grass:
            self._draw_colored_polygon(
                self.surf, poly, self.grass_color, zoom, translation, angle
            )

        # draw road
        for poly, color in self.road_poly:
            # converting to pixel coordinates
            poly = [(p[0], p[1]) for p in poly]
            color = [int(c) for c in color]
            self._draw_colored_polygon(self.surf, poly, color, zoom, translation, angle)

    def _render_indicators(self, W, H):
        s = W / 40.0
        h = H / 40.0
        color = (0, 0, 0)
        polygon = [(W, H), (W, H - 5 * h), (0, H - 5 * h), (0, H)]
        pygame.draw.polygon(self.surf, color=color, points=polygon)

        def vertical_ind(place, val):
            return [
                (place * s, H - (h + h * val)),
                ((place + 1) * s, H - (h + h * val)),
                ((place + 1) * s, H - h),
                ((place + 0) * s, H - h),
            ]

        def horiz_ind(place, val):
            return [
                ((place + 0) * s, H - 4 * h),
                ((place + val) * s, H - 4 * h),
                ((place + val) * s, H - 2 * h),
                ((place + 0) * s, H - 2 * h),
            ]

        assert self.car is not None
        true_speed = np.sqrt(
            np.square(self.car.hull.linearVelocity[0])
            + np.square(self.car.hull.linearVelocity[1])
        )

        # simple wrapper to render if the indicator value is above a threshold
        def render_if_min(value, points, color):
            if abs(value) > 1e-4:
                pygame.draw.polygon(self.surf, points=points, color=color)

        render_if_min(true_speed, vertical_ind(5, 0.02 * true_speed), (255, 255, 255))
        # ABS sensors
        render_if_min(
            self.car.wheels[0].omega,
            vertical_ind(7, 0.01 * self.car.wheels[0].omega),
            (0, 0, 255),
        )
        render_if_min(
            self.car.wheels[1].omega,
            vertical_ind(8, 0.01 * self.car.wheels[1].omega),
            (0, 0, 255),
        )
        render_if_min(
            self.car.wheels[2].omega,
            vertical_ind(9, 0.01 * self.car.wheels[2].omega),
            (51, 0, 255),
        )
        render_if_min(
            self.car.wheels[3].omega,
            vertical_ind(10, 0.01 * self.car.wheels[3].omega),
            (51, 0, 255),
        )

        render_if_min(
            self.car.steering_angle,
            horiz_ind(20, -10.0 * self.car.steering_angle),
            (0, 255, 0),
        )
        render_if_min(
            self.car.yaw_rate,
            horiz_ind(30, -0.8 * self.car.yaw_rate),
            (255, 0, 0),
        )

    def _draw_colored_polygon(
        self, surface, poly, color, zoom, translation, angle, clip=True
    ):
        poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in poly]
        poly = [
            (c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in poly
        ]
        # This checks if the polygon is out of bounds of the screen, and we skip drawing if so.
        # Instead of calculating exactly if the polygon and screen overlap,
        # we simply check if the polygon is in a larger bounding box whose dimension
        # is greater than the screen by MAX_SHAPE_DIM, which is the maximum
        # diagonal length of an environment object
        if not clip or any(
            (-MAX_SHAPE_DIM <= coord[0] <= WINDOW_W + MAX_SHAPE_DIM)
            and (-MAX_SHAPE_DIM <= coord[1] <= WINDOW_H + MAX_SHAPE_DIM)
            for coord in poly
        ):
            gfxdraw.aapolygon(self.surf, poly, color)
            gfxdraw.filled_polygon(self.surf, poly, color)

    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.isopen = False
            pygame.quit()


if __name__ == "__main__":
    a = np.array([0.0, 0.0, 0.0])

    def register_input():
        global quit, restart
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    a[0] = -1.0
                if event.key == pygame.K_RIGHT:
                    a[0] = +1.0
                if event.key == pygame.K_UP:
                    a[1] = +1.0
                if event.key == pygame.K_DOWN:
                    a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
                if event.key == pygame.K_RETURN:
                    restart = True
                if event.key == pygame.K_ESCAPE:
                    quit = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    a[0] = 0
                if event.key == pygame.K_RIGHT:
                    a[0] = 0
                if event.key == pygame.K_UP:
                    a[1] = 0
                if event.key == pygame.K_DOWN:
                    a[2] = 0

            if event.type == pygame.QUIT:
                quit = True

    env = CarRacing(render_mode="human")

    quit = False
    while not quit:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            register_input()
            s, r, terminated, truncated, info = env.step(a)
            total_reward += r
            if steps % 200 == 0 or terminated or truncated:
                print("\naction " + str([f"{x:+0.2f}" for x in a]))
                print(f"step {steps} total_reward {total_reward:+0.2f}")
            steps += 1
            if terminated or truncated or restart or quit:
                break
    env.close()
