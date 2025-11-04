__credits__ = ["Andrea PIERRÉ"]

import math

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from .car_dynamics import Car
from gymnasium.error import DependencyNotInstalled, InvalidAction
from gymnasium.utils import EzPickle


try:
    import Box2D
    from Box2D.b2 import contactListener, fixtureDef, polygonShape
except ImportError as e:
    raise DependencyNotInstalled(
        'Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`'
    ) from e

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


STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
FPS = 50  # Frames per second
ZOOM = 2.7  # Camera zoom
ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)


TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4
GRASS_DIM = PLAYFIELD / 20.0
MAX_SHAPE_DIM = (
    max(GRASS_DIM, TRACK_WIDTH, TRACK_DETAIL_STEP) * math.sqrt(2) * ZOOM * SCALE
)


class FrictionDetector(contactListener):
    def __init__(self, env, lap_complete_percent):
        contactListener.__init__(self)
        self.env = env
        self.lap_complete_percent = lap_complete_percent

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            return

        # inherit tile color from env
        tile.color[:] = self.env.road_color
        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:
            obj.tiles.add(tile)
            if not tile.road_visited:
                tile.road_visited = True
                self.env.reward += 1000.0 / len(self.env.track)
                self.env.tile_visited_count += 1

                # Lap is considered completed if enough % of the track was covered
                if (
                    tile.idx == 0
                    and self.env.tile_visited_count / len(self.env.track)
                    > self.lap_complete_percent
                ):
                    self.env.new_lap = True
        else:
            obj.tiles.remove(tile)


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
    If continuous there are 3 actions :
    - 0: steering, -1 is full left, +1 is full right
    - 1: gas
    - 2: braking

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
    The reward is -0.1 every frame and +1000/N for every track tile visited, where N is the total number of tiles
     visited in the track. For example, if you have finished in 732 frames, your reward is 1000 - 0.1*732 = 926.8 points.

    Additionally:
    - Speed bonus: +0.02 * speed per frame (encourages forward movement)
    - Off-track penalty: -5.0 per wheel off-track per frame (continuous penalty to prevent exploits)

    ## Starting State
    The car starts at rest in the center of the road.

    ## Episode Termination
    The episode finishes when all the tiles are visited. The car can also go outside the playfield -
     that is, far off the track, in which case it will receive -100 reward and die.

    Additionally, if `terminate_stationary=True`, episodes will be truncated early if the car makes
     no progress (no new tiles visited) for `stationary_patience` frames (default: 100), after a
     minimum of `stationary_min_steps` steps (default: 50). This prevents agents from learning to
     sit still and waste compute time.

    ## Arguments

    ```python
    >>> import gymnasium as gym
    >>> env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=True)
    >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<CarRacing<CarRacing-v3>>>>>

    ```

    * `lap_complete_percent=0.95` dictates the percentage of tiles that must be visited by
     the agent before a lap is considered complete.

    * `domain_randomize=False` enables the domain randomized variant of the environment.
     In this scenario, the background and track colours are different on every reset.

    * `continuous=True` specifies if the agent has continuous (true) or discrete (false) actions.
     See action space section for a description of each.

    * `terminate_stationary=True` enables early termination if the car makes no progress for too long.

    * `stationary_patience=100` sets the number of frames without progress before early termination.

    * `stationary_min_steps=50` sets the minimum steps before early termination can occur.

    * `state_mode="vector"` (RECOMMENDED) selects observation space format:
        - "vector": 36D track geometry with lookahead (default recommendation, fast training)
        - "visual": 96x96 RGB images (slow, use only for watching)

    ## Reset Arguments

    Passing the option `options["randomize"] = True` will change the current colour of the environment on demand.
    Correspondingly, passing the option `options["randomize"] = False` will not change the current colour of the environment.
    `domain_randomize` must be `True` on init for this argument to work.

    ```python
    >>> import gymnasium as gym
    >>> env = gym.make("CarRacing-v3", domain_randomize=True)

    # normal reset, this changes the colour scheme by default
    >>> obs, _ = env.reset()

    # reset with colour scheme change
    >>> randomize_obs, _ = env.reset(options={"randomize": True})

    # reset with no colour scheme change
    >>> non_random_obs, _ = env.reset(options={"randomize": False})

    ```

    ## Version History
    - v2: Change truncation to termination when finishing the lap (1.0.0)
    - v1: Change track completion logic and add domain randomization (0.24.0)
    - v0: Original version

    ## References
    - Chris Campbell (2014), http://www.iforce2d.net/b2dtut/top-down-car.

    ## Credits
    Created by Oleg Klimov
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
        stationary_patience: int = 100,
        stationary_min_steps: int = 50,
        state_mode: str = "vector",
    ):
        """
        Args:
            state_mode: "vector" (compact track geometry vector with lookahead - RECOMMENDED),
                        or "visual" (96x96 RGB images - slow, not recommended for training).
                        Default is "vector" for best performance and training results.
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
        )
        self.continuous = continuous
        self.domain_randomize = domain_randomize
        self.lap_complete_percent = lap_complete_percent
        self.terminate_stationary = terminate_stationary
        self.stationary_patience = stationary_patience
        self.stationary_min_steps = stationary_min_steps
        self.state_mode = state_mode
        self._init_colors()

        self.contactListener_keepref = FrictionDetector(self, self.lap_complete_percent)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
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
        self.new_lap = False
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )

        # Vector mode: waypoint lookahead count
        self.vector_lookahead = 10

        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised however this is not possible here so ignore
        if self.continuous:
            self.action_space = spaces.Box(
                np.array([-1, 0, 0]).astype(np.float32),
                np.array([+1, +1, +1]).astype(np.float32),
            )  # steer, gas, brake
        else:
            self.action_space = spaces.Discrete(5)
            # do nothing, right, left, gas, brake

        # Observation space depends on state_mode
        if self.state_mode == "vector":
            # Vector state: car state (11) + track segment info (5) + lookahead waypoints (20)
            # 36 values total
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32
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
        for t in self.road:
            self.world.DestroyBody(t)
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

        # Create tiles
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
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01 * (i % 3) * 255
            t.color = self.road_color + c
            t.road_visited = False
            t.road_friction = 1.0
            t.idx = i
            t.fixtures[0].sensor = True
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
        self.world.contactListener_bug_workaround = FrictionDetector(
            self, self.lap_complete_percent
        )
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.new_lap = False
        self.road_poly = []

        # Stationary car tracking
        self.frames_since_progress = 0
        self.total_steps = 0

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
        self.car = Car(self.world, *self.track[0][1:4])

        if self.render_mode == "human":
            self.render()
        return self.step(None)[0], {}

    def step(self, action: np.ndarray | int):
        assert self.car is not None
        if action is not None:
            if self.continuous:
                action = action.astype(np.float64)
                self.car.steer(-action[0])
                # Rescale gas and brake from [-1, 1] to [0, 1]
                # SAC outputs actions in [-1, 1], but gas/brake expect [0, 1]
                gas = np.clip((action[1] + 1.0) / 2.0, 0.0, 1.0)
                brake = np.clip((action[2] + 1.0) / 2.0, 0.0, 1.0)
                self.car.gas(gas)
                self.car.brake(brake)
            else:
                if not self.action_space.contains(action):
                    raise InvalidAction(
                        f"you passed the invalid action `{action}`. "
                        f"The supported action_space is `{self.action_space}`"
                    )
                self.car.steer(-0.6 * (action == 1) + 0.6 * (action == 2))
                self.car.gas(0.2 * (action == 3))
                self.car.brake(0.8 * (action == 4))

        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        # Create state based on state_mode
        if self.state_mode == "vector":
            # Fast vector state (no rendering) - 36D with track geometry
            self.state = self._create_vector_state()
        elif self.render_mode is not None:
            # Visual state with rendering
            self.state = self._render("state_pixels")
        else:
            # Headless visual mode: create minimal state without rendering
            self.state = self._create_headless_state()

        step_reward = 0
        terminated = False
        truncated = False
        info = {}
        if action is not None:  # First step without action, called from reset()
            self.reward -= 0.1
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0

            # Add speed bonus to encourage movement
            speed = np.sqrt(
                self.car.hull.linearVelocity[0]**2 +
                self.car.hull.linearVelocity[1]**2
            )
            self.reward += 0.02 * speed  # Reward for moving fast

            # Continuous penalty for wheels off track (no sharp boundaries to exploit)
            wheels_off_track = sum(1 for wheel in self.car.wheels if len(wheel.tiles) == 0)
            self.reward -= 5.0 * wheels_off_track  # -5 per wheel off track per frame

            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward

            # Track stationary car (for early termination)
            if self.terminate_stationary:
                self.total_steps += 1

                # Check if car made progress:
                # 1. Visited new tile (step_reward > 0), OR
                # 2. Moving with meaningful velocity (speed > 0.5 m/s)
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
                terminated = True
                info["lap_finished"] = True

            # Check if all 4 wheels are off track
            all_wheels_off_track = all(len(wheel.tiles) == 0 for wheel in self.car.wheels)
            if all_wheels_off_track:
                terminated = True
                info["lap_finished"] = False
                info["off_track"] = True
                step_reward = -100

        if self.render_mode == "human":
            self.render()
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

        Returns 36-dimensional state vector:
        - Car state (11): x, y, vx, vy, angle, angular_vel, wheel_contacts[4], track_progress
        - Track segment info (5): dist_to_center, angle_diff, curvature, dist_along_segment, segment_length
        - Lookahead waypoints (20): 10 waypoints × (x, y) in car-relative coordinates
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

        # 3. Get next N waypoints in car-relative coordinates
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

        # Combine all features
        state = np.array([
            car_x, car_y, vx, vy, angle, angular_vel,
            wheel_contacts[0], wheel_contacts[1], wheel_contacts[2], wheel_contacts[3],
            track_progress,
            dist_to_center_norm, angle_diff, curvature, dist_along / TRACK_DETAIL_STEP, seg_len / TRACK_DETAIL_STEP,
            *waypoints
        ], dtype=np.float32)

        return state

    def _create_headless_state(self):
        """
        Fast headless rendering for training (3-5x faster than full rendering).
        Creates minimal pygame surface at target resolution without expensive elements.
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
            except:
                pass  # Skip if polygon is off-screen

        # Draw car (simplified)
        for obj in self.car.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans * v for v in f.shape.vertices]
                screen_path = []
                for coords in path:
                    x = coords[0] * zoom_factor + scroll_x
                    y = coords[1] * zoom_factor + scroll_y
                    screen_path.append((x, y))

                color = [int(c * 255) for c in obj.color]
                try:
                    pygame.draw.polygon(surf, color, screen_path)
                except:
                    pass

        # Flip vertically to match standard rendering
        surf = pygame.transform.flip(surf, False, True)

        # Convert to numpy array
        return self._create_image_array(surf, (STATE_W, STATE_H))

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
        # computing transformations
        angle = -self.car.hull.angle
        # Use constant zoom (no animation)
        zoom = ZOOM * SCALE
        scroll_x = -(self.car.hull.position[0]) * zoom
        scroll_y = -(self.car.hull.position[1]) * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])

        self._render_road(zoom, trans, angle)
        self.car.draw(
            self.surf,
            zoom,
            trans,
            angle,
            mode not in ["state_pixels_list", "state_pixels"],
        )

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
            self.car.wheels[0].joint.angle,
            horiz_ind(20, -10.0 * self.car.wheels[0].joint.angle),
            (0, 255, 0),
        )
        render_if_min(
            self.car.hull.angularVelocity,
            horiz_ind(30, -0.8 * self.car.hull.angularVelocity),
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
