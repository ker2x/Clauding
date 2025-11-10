__credits__ = ["Andrea PIERRÉ"]

import math
import time

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from .car_dynamics import Car
from gymnasium.error import DependencyNotInstalled, InvalidAction
from gymnasium.utils import EzPickle


# Box2D no longer needed - using custom 2D physics engine
# Removed Box2D dependency for cleaner, more interpretable physics

try:
    # As pygame is necessary for using the environment (reset and step) even without a render mode
    #   therefore, pygame is a necessary import for the environment.
    import pygame
    from pygame import gfxdraw
except ImportError as e:
    raise DependencyNotInstalled(
        'pygame is not installed, run `pip install "gymnasium[box2d]"`'
    ) from e

try:
    import cv2
except ImportError as e:
    raise DependencyNotInstalled(
        'opencv is not installed, run `pip install opencv-python`'
    ) from e

# visual mode resolution
STATE_W = 96
STATE_H = 96

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

# Reward structure configuration
NUM_CHECKPOINTS = 15        # Number of checkpoints to divide track into (~30 tiles each for 300-tile track)
CHECKPOINT_REWARD = 100.0   # Reward for reaching each checkpoint (total = NUM_CHECKPOINTS * CHECKPOINT_REWARD)
LAP_COMPLETION_REWARD = 500.0  # Large reward for completing a full lap (encourages finishing)
FORWARD_VEL_REWARD = 0.1    # Reward per m/s of forward velocity per frame (0.0 = disabled, try 0.05-0.1 to enable)
STEP_PENALTY = 1.0          # Penalty per frame (encourages speed via less total penalty) - increased to strongly favor fast laps
OFFTRACK_PENALTY = 2.0      # Penalty per wheel off track per frame
OFFTRACK_THRESHOLD = 2      # Number of wheels that can be off track before penalty applies (allows aggressive lines)


class FrictionDetector:
    """
    Detects wheel-track collisions using accurate polygon-based geometry.
    Replaces Box2D contact listener with spatial geometry queries.
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

                # Handle tile visitation (only count when first wheel touches it)
                if not tile.road_visited:
                    tile.road_visited = True
                    self.env.tile_visited_count += 1

                    # Checkpoint system: reward when reaching NEXT checkpoint in sequence
                    # This prevents backward driving exploits where AI wraps around to get rewards
                    current_checkpoint = tile.idx // self.env.checkpoint_size
                    expected_next_checkpoint = self.env.last_checkpoint_reached + 1

                    # Allow wrapping: after last checkpoint (14), next is 0 (lap completion)
                    if expected_next_checkpoint >= self.env.num_checkpoints:
                        expected_next_checkpoint = 0

                    # Check that car is moving forward (not backward)
                    # Get car's forward velocity (vx in body frame)
                    car_forward_velocity = self.env.car.vx if hasattr(self.env.car, 'vx') else 0.0
                    is_moving_forward = car_forward_velocity > 0.1  # Must be moving forward at > 0.1 m/s

                    # Only reward if:
                    # 1. Reaching the NEXT checkpoint in sequence (prevents backward farming)
                    # 2. Moving forward (prevents backward driving exploits)
                    if current_checkpoint == expected_next_checkpoint and is_moving_forward:
                        self.env.last_checkpoint_reached = current_checkpoint
                        self.env.reward += self.env.checkpoint_reward
                        if self.env.verbose:
                            progress_pct = (current_checkpoint + 1) / self.env.num_checkpoints * 100
                            print(f"  ✓ Checkpoint {current_checkpoint + 1}/{self.env.num_checkpoints} "
                                  f"reached! ({progress_pct:.0f}% complete, +{self.env.checkpoint_reward} reward)")
                    elif not is_moving_forward and self.env.verbose:
                        # Debug: car reached checkpoint while moving backward
                        print(f"  ⚠ Checkpoint {current_checkpoint + 1} reached while moving BACKWARD "
                              f"(vx={car_forward_velocity:.2f} m/s) - NO REWARD")

                    # Lap completion check
                    if (
                        tile.idx == 0 and
                        self.env.tile_visited_count / len(self.env.track) > self.lap_complete_percent
                    ):
                        self.env.new_lap = True


class CarRacing(gym.Env, EzPickle):
    """
    ## Description
    The easiest control task to learn from pixels - a top-down
    racing environment. The generated track is random every episode.

    Some indicators are shown at the bottom of the window along with the
    state RGB buffer. From left to right: true speed, four ABS sensors,
    steering wheel position, and gyroscope.
    To play yourself (it's rather fast for humans), type:
    ```shell
    python gymnasium/envs/box2d/car_racing.py
    ```
    Remember: it's a powerful rear-wheel drive car - don't press the accelerator
    and turn at the same time.

    ## Action Space
    If continuous there are 2 actions:
    - 0: steering, -1 is full left, +1 is full right
    - 1: acceleration, -1 is full brake, +1 is full gas

    If discrete there are 5 actions:
    - 0: do nothing
    - 1: steer right
    - 2: steer left
    - 3: gas
    - 4: brake

    ## Observation Space

    Depends on `state_mode` parameter:
    - "vector": 36-dimensional track geometry with lookahead (RECOMMENDED - fast and informative)
    - "visual": 96x96 RGB image (slow - not recommended for training, useful for watching)

    **Recommendation**: Use `state_mode="vector"` for training. It's 3-5x faster than visual mode
    and provides sufficient information for the agent to learn proper racing behavior.

    ## Rewards
    The reward structure uses a sparse checkpoint system combined with step penalty.
    All reward parameters are configurable at the top of this file (lines 64-70).

    **Sparse rewards (main objective):**
    - Checkpoint rewards: +CHECKPOINT_REWARD points for each of NUM_CHECKPOINTS checkpoints
      (default: 10 checkpoints × 100 points = 1000 total)
    - Each checkpoint is ~30 tiles (for typical 300-tile track), making them achievable through exploration
    - Must visit tiles in sequence to reach next checkpoint

    **Dense penalties (constraints and speed incentive):**
    - Per-step penalty: -STEP_PENALTY every frame (default: -0.5, implicitly encourages reaching checkpoints quickly)
    - Off-track penalty: -OFFTRACK_PENALTY per wheel off-track per frame when >OFFTRACK_THRESHOLD wheels off
      (default: -1.0 per wheel when >2 wheels off, allows aggressive racing with 2 wheels off)

    **Optional dense reward:**
    - Forward velocity: +FORWARD_VEL_REWARD per m/s of forward velocity per frame
      (default: 0.0 = disabled, set to 0.05-0.1 to enable)

    Example with defaults: Reaching checkpoint 5 (50% progress) in 366 frames:
    - Checkpoint rewards: 5 * 100 = +500
    - Step penalty: -0.5 * 366 = -183
    - Total: ~317 points

    ## Starting State
    The car starts at rest in the center of the road.

    ## Episode Termination
    The episode finishes when all the tiles are visited. The car can also go outside the playfield -
     that is, far off the track, in which case it will receive -100 reward and die.

    Additionally, if `terminate_stationary=True`, episodes will be truncated early if the car makes
     no progress (no new tiles visited) for `stationary_patience` frames (default: 100), after a
     minimum of `stationary_min_steps` steps (default: 50). This prevents agents from learning to
     sit still and waste compute time.
"""
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "state_pixels",
            "agent_view",  # Shows what the agent actually sees (96x96 optimized view)
        ],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        verbose: bool = False,
        lap_complete_percent: float = 0.95,
        domain_randomize: bool = False,
        continuous: bool = True,
        terminate_stationary: bool = True,
        stationary_patience: int = 50,
        stationary_min_steps: int = 50,
        state_mode: str = "vector",
        max_episode_steps: int | None = 1500,
        reward_shaping: bool = True,
        min_episode_steps: int = 150,
        short_episode_penalty: float = -50.0,
    ):
        """
        Args:
            state_mode: "vector" (compact track geometry vector with lookahead - RECOMMENDED),
                        or "visual" (96x96 RGB images - slow, not recommended for training).
                        Default is "vector" for best performance and training results.
            max_episode_steps: Maximum steps per episode (default: 1500). None for unlimited.
            reward_shaping: Apply penalty for short episodes (default: True)
            min_episode_steps: Minimum episode length before penalty (default: 150)
            short_episode_penalty: Penalty for episodes shorter than min_episode_steps (default: -50.0)
        """
        EzPickle.__init__(
            self,
            render_mode,
            verbose,
            lap_complete_percent,
            domain_randomize,
            continuous,
            terminate_stationary,
            stationary_patience,
            stationary_min_steps,
            state_mode,
            max_episode_steps,
            reward_shaping,
            min_episode_steps,
            short_episode_penalty,
        )
        self.continuous = continuous
        self.domain_randomize = domain_randomize
        self.lap_complete_percent = lap_complete_percent
        self.terminate_stationary = terminate_stationary
        self.stationary_patience = stationary_patience
        self.stationary_min_steps = stationary_min_steps
        self.state_mode = state_mode
        self.max_episode_steps = max_episode_steps
        self.reward_shaping = reward_shaping
        self.min_episode_steps = min_episode_steps
        self.short_episode_penalty = short_episode_penalty
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
        self.car: Car | None = None
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

        # Checkpoint system for sparse rewards (configured at top of file)
        self.num_checkpoints = NUM_CHECKPOINTS
        self.checkpoint_reward = CHECKPOINT_REWARD

        # Continuous: 2D action space [steering, acceleration]
        # steering: [-1, +1], acceleration: [-1 (brake), +1 (gas)]
        if self.continuous:
            self.action_space = spaces.Box(
                np.array([-1, -1]).astype(np.float32),
                np.array([+1, +1]).astype(np.float32),
            )  # steer, acceleration
        else:
            self.action_space = spaces.Discrete(5)
            # do nothing, right, left, gas, brake

        # Observation space depends on state_mode
        if self.state_mode == "vector":
            # Vector state: car state (11) + track segment info (5) + lookahead waypoints (40)
            # + speed (1) + longitudinal accel (1) + lateral accel (1)
            # + slip angles (4) + slip ratios (4)
            # = 67 values total (increased from 47 to support 20 waypoint lookahead)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(67,), dtype=np.float32
            )
        else:
            # Visual state: 96x96 RGB image
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
            )

        self.render_mode = render_mode

    def _destroy(self):
        if not self.road:
            return
        # Tiles are just objects now, no Box2D bodies to destroy
        self.road = []
        assert self.car is not None
        self.car.destroy()

    def _init_colors(self):
        if self.domain_randomize:
            # domain randomize the bg and grass colour
            self.road_color = self.np_random.uniform(0, 210, size=3)

            self.bg_color = self.np_random.uniform(0, 210, size=3)

            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3)
            self.grass_color[idx] += 20
        else:
            # default colours
            self.road_color = np.array([102, 102, 102])
            self.bg_color = np.array([102, 204, 102])
            self.grass_color = np.array([102, 230, 102])

    def _reinit_colors(self, randomize):
        assert (
            self.domain_randomize
        ), "domain_randomize must be True to use this function."

        if randomize:
            # domain randomize the bg and grass colour
            self.road_color = self.np_random.uniform(0, 210, size=3)

            self.bg_color = self.np_random.uniform(0, 210, size=3)

            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3)
            self.grass_color[idx] += 20

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
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.new_lap = False
        self.road_poly = []
        self.debug_step_counter = 0

        # Stationary car tracking
        self.frames_since_progress = 0
        self.total_steps = 0

        # Episode step counter for timeout and reward shaping
        self.episode_steps = 0

        # Previous velocity for acceleration computation
        self.prev_vx = 0.0
        self.prev_vy = 0.0

        # Checkpoint tracking (initialized after track creation)
        self.last_checkpoint_reached = -1
        self.checkpoint_size = 0  # Will be set after track is created

        if self.domain_randomize:
            randomize = True
            if isinstance(options, dict):
                if "randomize" in options:
                    randomize = options["randomize"]

            self._reinit_colors(randomize)

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose:
                print(
                    "retry to generate track (normal if there are not many"
                    "instances of this message)"
                )

        # Initialize checkpoint system after track is created
        self.checkpoint_size = len(self.track) // self.num_checkpoints
        if self.verbose:
            print(f"Track has {len(self.track)} tiles, {self.num_checkpoints} checkpoints "
                  f"of ~{self.checkpoint_size} tiles each")

        init_beta, init_x, init_y = self.track[0][1:4]
        # The car's "front" is its +X axis in physics.
        # The track's "forward" direction is 90 degrees (pi/2) from its
        # normal (beta). We set the car's initial yaw to align these.
        init_yaw = init_beta + (math.pi / 2.0)

        # Create the car
        self.car = Car(self.world, init_yaw, init_x, init_y)

        if self.render_mode == "human":
            self.render()
        return self.step(None)[0], {}

    def step(self, action: np.ndarray | int):
        assert self.car is not None

        # Start timing for verbose mode
        step_start_time = time.perf_counter() if self.verbose else None

        # Initialize action vars for debug print
        gas, brake = 0.0, 0.0
        steer_action = 0.0
        accel = 0.0

        if action is not None:
            if self.continuous:
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
            else:
                if not self.action_space.contains(action):
                    raise InvalidAction(
                        f"you passed the invalid action `{action}`. "
                        f"The supported action_space is `{self.action_space}`"
                    )
                steer_action = -0.6 * (action == 1) + 0.6 * (action == 2)
                gas = 0.2 * (action == 3)
                brake = 0.8 * (action == 4)

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

        # Create state based on state_mode
        state_start = time.perf_counter() if self.verbose else None
        if self.state_mode == "vector":
            # Fast vector state (no rendering) - 36D with track geometry
            self.state = self._create_vector_state()
        elif self.render_mode is not None:
            # Visual state with rendering
            self.state = self._render("state_pixels")
        else:
            # Headless visual mode: create minimal state without rendering
            self.state = self._create_headless_state()
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
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0

            # Calculate speed magnitude (for stationary detection and debug output)
            speed = np.sqrt(
                self.car.hull.linearVelocity[0] ** 2 +
                self.car.hull.linearVelocity[1] ** 2
            )

            # Calculate forward velocity (project velocity onto car's forward direction)
            # Car's forward direction is its heading angle
            car_forward_x = np.cos(self.car.hull.angle)
            car_forward_y = np.sin(self.car.hull.angle)

            # Dot product of velocity with forward direction
            forward_velocity = (
                self.car.hull.linearVelocity[0] * car_forward_x +
                self.car.hull.linearVelocity[1] * car_forward_y
            )

            # Reward forward progress only (backward movement = no reward)
            # Configured at top of file: FORWARD_VEL_REWARD (currently 0.0 = disabled)
            # Hypothesis: velocity is implicitly rewarded by reaching checkpoints faster
            # Can be re-enabled if agent struggles to learn (set to 0.05 or 0.1)
            self.reward += FORWARD_VEL_REWARD * max(0, forward_velocity)

            # Continuous penalty for wheels off track (no sharp boundaries to exploit)
            wheels_off_track = sum(1 for wheel in self.car.wheels if len(wheel.tiles) == 0)
            if wheels_off_track > OFFTRACK_THRESHOLD:
                self.reward -= OFFTRACK_PENALTY * wheels_off_track

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
                step_reward = -100

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

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        elif self.render_mode == "agent_view":
            return self._render_agent_view()
        else:
            return self._render(self.render_mode)

    def _render_agent_view(self):
        """
        Render mode that shows exactly what the agent sees during training.
        Returns the 96x96 optimized view in human-viewable window.
        """
        pygame.font.init()
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            # Create a larger window to show the small 96x96 view
            self.screen = pygame.display.set_mode((STATE_W * 6, STATE_H * 6))
            pygame.display.set_caption("Agent View (96x96 scaled 6x)")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        # Get the agent's actual view
        agent_view = self._create_headless_state()

        # Convert to pygame surface
        # agent_view is (96, 96, 3) in range [0, 255]
        surf = pygame.surfarray.make_surface(np.transpose(agent_view, (1, 0, 2)))

        # Scale up 6x for visibility
        scaled_surf = pygame.transform.scale(surf, (STATE_W * 6, STATE_H * 6))

        # Display
        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        self.screen.fill(0)
        self.screen.blit(scaled_surf, (0, 0))

        # Add text overlay showing it's agent view
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        text = font.render("Agent View (96x96 @ 6x scale)", True, (255, 255, 0), (0, 0, 0))
        self.screen.blit(text, (10, 10))

        pygame.display.flip()

        return agent_view

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

        Returns 67-dimensional state vector (increased from 47 for better lookahead):
        - Car state (11): x, y, vx, vy, angle, angular_vel, wheel_contacts[4], track_progress
        - Track segment info (5): dist_to_center, angle_diff, curvature, dist_along_segment, segment_length
        - Lookahead waypoints (40): 20 waypoints × (x, y) in car-relative coordinates (increased from 10)
        - Speed (1): magnitude of velocity
        - Accelerations (2): longitudinal (body frame), lateral (body frame)
        - Slip angles (4): for each wheel [FL, FR, RL, RR]
        - Slip ratios (4): for each wheel [FL, FR, RL, RR]

        Note: Lookahead increased from 10 to 20 waypoints to allow high-speed braking.
        At 108 km/h, 20 waypoints = 70m = 2.33s lookahead (enough to brake for corners).
        """
        assert self.car is not None

        # 1. Get basic car state
        car_x = self.car.hull.position[0] / PLAYFIELD
        car_y = self.car.hull.position[1] / PLAYFIELD
        vx = self.car.hull.linearVelocity[0]
        vy = self.car.hull.linearVelocity[1]
        angle = self.car.hull.angle / (2 * np.pi)
        angular_vel = self.car.hull.angularVelocity
        wheel_contacts = [1.0 if len(wheel.tiles) > 0 else 0.0 for wheel in self.car.wheels]
        track_progress = self.tile_visited_count / len(self.track) if len(self.track) > 0 else 0.0

        # Calculate speed (magnitude of velocity)
        speed = np.sqrt(vx ** 2 + vy ** 2)

        # Calculate accelerations in body frame (change in velocity)
        # Note: For the first step, prev_vx/vy are 0, so acceleration will be 0
        dt = 1.0 / FPS
        ax = (self.car.vx - self.prev_vx) / dt if dt > 0 else 0.0  # Longitudinal
        ay = (self.car.vy - self.prev_vy) / dt if dt > 0 else 0.0  # Lateral

        # Update previous velocities for next step
        self.prev_vx = self.car.vx
        self.prev_vy = self.car.vy

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

        # Distance along segment (normalized)
        _, _, x1, y1 = self.track[seg_idx]
        _, _, x2, y2 = self.track[prev_idx]
        seg_len = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if seg_len > 0:
            t = ((car_world_pos[0] - x1) * (x2 - x1) + (car_world_pos[1] - y1) * (y2 - y1)) / (seg_len**2)
            dist_along = t * seg_len
        else:
            dist_along = 0.0

        # Normalize distance to center by track width
        dist_to_center_norm = dist_to_center / TRACK_WIDTH

        # 3. Get slip angles and slip ratios from stored tire forces
        # IMPORTANT: Use stored forces from car.last_tire_forces to avoid
        # recomputation which causes oscillations in RL input due to
        # tire force feedback loop
        slip_angles = []
        slip_ratios = []

        if hasattr(self.car, 'last_tire_forces') and self.car.last_tire_forces is not None:
            # Use pre-computed values (consistent with physics simulation)
            forces = self.car.last_tire_forces
            for i in range(4):
                if i in forces:
                    slip_angles.append(forces[i].get('slip_angle', 0.0))
                    slip_ratios.append(forces[i].get('slip_ratio', 0.0))
                else:
                    slip_angles.append(0.0)
                    slip_ratios.append(0.0)
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

        # 4. Get next N waypoints in car-relative coordinates
        waypoints = []
        car_angle_rad = self.car.hull.angle
        cos_a = np.cos(car_angle_rad)
        sin_a = np.sin(car_angle_rad)

        for i in range(self.vector_lookahead):
            wp_idx = (seg_idx + i + 1) % len(self.track)
            _, _, wp_x, wp_y = self.track[wp_idx]

            # Transform to car-relative coordinates
            dx = wp_x - car_world_pos[0]
            dy = wp_y - car_world_pos[1]

            # Rotate into car's frame
            rel_x = dx * cos_a + dy * sin_a
            rel_y = -dx * sin_a + dy * cos_a

            # Normalize by a reasonable distance scale
            waypoints.extend([rel_x / PLAYFIELD, rel_y / PLAYFIELD])

        # Combine all features (67 total, increased from 47)
        state = np.array([
            # Basic car state (11)
            car_x, car_y, vx, vy, angle, angular_vel,
            wheel_contacts[0], wheel_contacts[1], wheel_contacts[2], wheel_contacts[3],
            track_progress,
            # Track segment info (5)
            dist_to_center_norm, angle_diff, curvature, dist_along / TRACK_DETAIL_STEP, seg_len / TRACK_DETAIL_STEP,
            # Waypoints (40 - increased from 20 for better lookahead)
            *waypoints,
            # Speed (1)
            speed,
            # Accelerations (2)
            ax, ay,
            # Slip angles (4)
            slip_angles[0], slip_angles[1], slip_angles[2], slip_angles[3],
            # Slip ratios (4)
            slip_ratios[0], slip_ratios[1], slip_ratios[2], slip_ratios[3]
        ], dtype=np.float32)

        return state

    def _create_headless_state(self):
        """
        Fast headless rendering for training (3-5x faster than full rendering).
        Creates minimal pygame surface at target resolution without expensive elements.
        Compatible with new physics engine (no Box2D fixtures).
        """
        # Use smaller surface size directly
        target_size = (STATE_W, STATE_H)
        surf = pygame.Surface(target_size)

        # Simple rendering without zoom or rotation
        assert self.car is not None

        # Calculate simpler camera transform (no rotation)
        zoom_factor = ZOOM * SCALE * STATE_W / WINDOW_W
        scroll_x = -self.car.hull.position[0] * zoom_factor + STATE_W / 2
        scroll_y = -self.car.hull.position[1] * zoom_factor + STATE_H / 2

        # Draw background (solid color, no grass patches)
        surf.fill(self.bg_color)

        # Draw road (simplified, no anti-aliasing)
        for poly, color in self.road_poly:
            # Transform to screen coordinates
            screen_poly = []
            for p in poly:
                x = p[0] * zoom_factor + scroll_x
                y = p[1] * zoom_factor + scroll_y
                screen_poly.append((x, y))

            # Simple filled polygon (no anti-aliasing)
            try:
                pygame.draw.polygon(surf, color, screen_poly)
            except (pygame.error, ValueError, TypeError):
                pass  # Skip if polygon is off-screen or has invalid coordinates

        # Draw car body (simplified for new physics engine)
        self._draw_car_simple(surf, zoom_factor, scroll_x, scroll_y)

        # Flip vertically to match standard rendering
        surf = pygame.transform.flip(surf, False, True)

        # Convert to numpy array
        return self._create_image_array(surf, (STATE_W, STATE_H))

    def _draw_car_simple(self, surf, zoom_factor, scroll_x, scroll_y):
        """
        Draw car body using simple geometry (no Box2D fixtures).
        Draws a simple box representing the car.
        """
        assert self.car is not None

        # Car dimensions in local frame
        car_x, car_y = self.car.hull.position
        angle = -self.car.hull.angle + (math.pi / 2.0)

        # Draw car as a simple rectangle
        # Approximate car as 4m long, 2m wide
        car_length = 2.0
        car_width = 1.0

        # Car corners in local frame
        corners_local = [
            (-car_width / 2, car_length / 2),   # Front left
            (car_width / 2, car_length / 2),    # Front right
            (car_width / 2, -car_length / 2),   # Rear right
            (-car_width / 2, -car_length / 2),  # Rear left
        ]

        # Rotate and translate to world frame
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        corners_world = []
        for dx, dy in corners_local:
            # Rotate
            x = dx * cos_a - dy * sin_a
            y = dx * sin_a + dy * cos_a
            # Translate
            corners_world.append((car_x + x, car_y + y))

        # Transform to screen coordinates
        screen_corners = []
        for x, y in corners_world:
            sx = x * zoom_factor + scroll_x
            sy = y * zoom_factor + scroll_y
            screen_corners.append((sx, sy))

        # Draw car body
        try:
            color = [int(c * 255) for c in self.car.hull.color]
            pygame.draw.polygon(surf, color, screen_corners)
        except (pygame.error, ValueError, TypeError, AttributeError):
            pass  # Skip if car rendering fails (e.g., color not set or invalid coordinates)

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
        self._render_car(zoom, trans, angle, mode not in ["state_pixels_list", "state_pixels"])

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
        elif mode == "state_pixels":
            return self._create_image_array(self.surf, (STATE_W, STATE_H))
        else:
            return self.isopen

    def _render_car(self, zoom, translation, angle, draw_particles=True):
        """
        Render car and particles for full rendering (compatible with new physics engine).
        Draws the car body AND the four wheels, rotating the front wheels with steering.
        """
        assert self.car is not None

        # 1. DRAW CAR BODY
        car_x, car_y = self.car.hull.position
        car_world_angle = self.car.hull.angle

        # Car corners in local frame (MATCHING PHYSICS COORDS)
        front_x = self.car.LF
        rear_x = -self.car.LR
        left_y = self.car.WIDTH / 2
        right_y = -self.car.WIDTH / 2

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
        color = [int(c * 255) for c in self.car.hull.color]
        gfxdraw.filled_polygon(self.surf, body_poly, color)

        # Draw hood
        gfxdraw.filled_polygon(self.surf, hood_poly, (200, 200, 200))  # Light grey

        # 2. DRAW WHEELS
        tire_length = self.car.TIRE_RADIUS * 2  # Length of the tire
        tire_width = self.car.TIRE_WIDTH
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

        for i, wheel in enumerate(self.car.wheels):
            # Get wheel's world position (from dynamics)
            wx, wy = wheel.position

            # Get wheel's world angle
            wheel_world_angle = self.car.hull.angle
            if i < 2:  # Front wheels
                wheel_world_angle += self.car.steering_angle

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
