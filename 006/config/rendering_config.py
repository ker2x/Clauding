"""
Rendering configuration for CarRacing environment.

This module centralizes all rendering-related constants used in the
CarRacing environment, including video resolution, window size, camera
settings, and track visualization parameters.
"""

from dataclasses import dataclass, field
import math


@dataclass
class VideoConfig:
    """Video/rendering resolution settings."""
    VIDEO_W: int = 600  # Video width (pixels)
    VIDEO_H: int = 400  # Video height (pixels)
    WINDOW_W: int = 1000  # Window width with GUI (pixels)
    WINDOW_H: int = 800  # Window height with GUI (pixels)
    FPS: int = 50  # Frames per second


@dataclass
class CameraConfig:
    """Camera and world scaling settings."""
    SCALE: float = 6.0  # Track scale factor
    ZOOM: float = 2.7  # Camera zoom level
    PLAYFIELD: float = 2000 / SCALE  # Game over boundary

    @property
    def track_rad(self):
        """Track is heavily morphed circle with this radius."""
        return 900 / self.SCALE


@dataclass
class TrackVisualsConfig:
    """Track visualization parameters."""
    TRACK_DETAIL_STEP: float = 21 / 6.0  # Detail step (using SCALE=6.0)
    TRACK_TURN_RATE: float = 0.31  # Turn rate for track generation
    TRACK_WIDTH: float = 40 / 6.0  # Track width (using SCALE=6.0)
    BORDER: float = 8 / 6.0  # Border width (using SCALE=6.0)
    BORDER_MIN_COUNT: int = 4  # Minimum border count
    GRASS_DIM: float = (2000 / 6.0) / 20.0  # Grass tile dimension (PLAYFIELD/20)

    def __post_init__(self):
        """Calculate derived parameters."""
        # MAX_SHAPE_DIM for rendering
        self.max_shape_dim = (
            max(self.GRASS_DIM, self.TRACK_WIDTH, self.TRACK_DETAIL_STEP)
            * math.sqrt(2) * 2.7 * 6.0  # sqrt(2) * ZOOM * SCALE
        )


@dataclass
class FrictionDetectionConfig:
    """Friction detection and collision parameters."""
    NEAR_TRACK_THRESHOLD: float = 0.3  # Tolerance for wheel-on-track detection
    SPATIAL_CHECK_RANGE: int = 30  # ±N tiles around car position for collision checks


@dataclass
class StateNormalizationConfig:
    """
    Normalization constants for vector state creation.

    These values are used to normalize state variables to [-1, 1] range
    for neural network input.
    """
    MAX_VELOCITY: float = 30.0  # m/s (used for velocity normalization)
    MAX_ANGULAR_VEL: float = 5.0  # rad/s
    MAX_ACCELERATION: float = 50.0  # m/s^2
    MAX_CURVATURE: float = 1.0  # rad/m
    MAX_SLIP_RATIO: float = 2.0  # dimensionless
    MAX_VERTICAL_FORCE: float = 5000.0  # N


@dataclass
class RenderingConfig:
    """
    Complete rendering configuration combining all rendering parameter groups.

    Usage:
        config = RenderingConfig()
        print(config.video.VIDEO_W)  # 600
        print(config.camera.ZOOM)  # 2.7
    """
    video: VideoConfig = field(default_factory=VideoConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    track_visuals: TrackVisualsConfig = field(default_factory=TrackVisualsConfig)
    friction_detection: FrictionDetectionConfig = field(default_factory=FrictionDetectionConfig)
    state_normalization: StateNormalizationConfig = field(default_factory=StateNormalizationConfig)
