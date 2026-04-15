"""Physical and game constants for Toribash 2D.

This module contains all tunable constants that control physics simulation,
game rules, and arena dimensions. All values use centimeters as the base unit
for consistency with pymunk's coordinate system.

Physics Constants:
    - Gravity: 900 cm/s² downward (realistic fall speed)
    - Timestep: 1/60s (60 FPS physics)
    - Solver iterations: 20 (balance of stability vs speed)

Arena:
    - Ground at y=50cm
    - 600x400cm playable area
    - Fighters spawn 50cm left/right of center

Note:
    Motor max forces are intentionally high (200K+) to enable ragdolls
    to stand against gravity. Lower values cause collapse.
"""

# =============================================================================
# Physics Constants
# =============================================================================

# Gravity vector applied to all dynamic bodies (cm/s²).
# Negative Y points downward in pymunk's coordinate system.
GRAVITY: tuple[float, float] = (0, -900)

# Physics timestep in seconds. 1/60s gives 60 FPS simulation.
# Must match the frame rate used in rendering for smooth playback.
DT: float = 1 / 60

# Number of solver iterations per physics step.
# Higher = more accurate joint constraints, slower simulation.
# 20 iterations is a good balance for standing ragdolls.
SPACE_ITERATIONS: int = 20


# =============================================================================
# Game Constants
# =============================================================================

# Number of physics steps to simulate per game turn.
# 30 steps × 1/60s = 0.5 seconds per turn (simulated time).
# This is the "thinking" time players have to set joint states.
STEPS_PER_TURN: int = 30

# Maximum number of turns before a match ends.
# 20 turns × 0.5s = 10 seconds total simulated time per match.
MAX_TURNS: int = 20


# =============================================================================
# Arena Dimensions
# =============================================================================

# Y-coordinate of the ground surface. Fighters spawn with feet at this height.
# The arena floor is a 5-pixel thick static segment at this level.
GROUND_Y: float = 50.0

# Arena width in centimeters. Used for spawning and bounds checking.
ARENA_WIDTH: float = 600.0

# Arena height in centimeters. Measured from ground upward.
ARENA_HEIGHT: float = 400.0

# Horizontal offset from center for each fighter's spawn position.
# Each fighter spawns SPAWN_OFFSET_X cm away from the center line.
# Total spawn distance between fighters = 2 × SPAWN_OFFSET_X
SPAWN_OFFSET_X: float = 50.0


# =============================================================================
# Collision Categories (Bitmasks)
# =============================================================================

# Collision categories determine which objects can collide with each other.
# Bodies have a collision_type and a category bitmask; collisions occur
# when shapes' category and collision_type overlap appropriately.

# Category for the ground plane (static body).
# Both players collide with ground.
CAT_GROUND: int = 0b0001

# Category for all segments belonging to player A.
# Collides with ground and player B.
CAT_PLAYER_A: int = 0b0010

# Category for all segments belonging to player B.
# Collides with ground and player A.
CAT_PLAYER_B: int = 0b0100


# =============================================================================
# Damage & Dismemberment Thresholds
# =============================================================================

# Minimum impulse magnitude (in pymunk units) to register as damage.
# Impulses below this threshold are ignored as minor contact.
# This prevents tiny nudges from scoring damage.
DAMAGE_IMPULSE_THRESHOLD: float = 500.0

# Impulse magnitude required to dismember (detach) a limb.
# When a joint receives more than this impulse, it breaks off.
# 5000 is ~10× the damage threshold for significant impacts only.
DISMEMBER_IMPULSE: float = 5000.0


# =============================================================================
# Joint Motor Parameters
# =============================================================================

# Default angular velocity for CONTRACT/EXTEND actions (radians/second).
# Controls how fast joints move when commanded.
# 15 rad/s ≈ 86°/s, reasonable human-like speed.
DEFAULT_MOTOR_RATE: float = 15.0

# Default maximum torque for HOLD/CONTRACT/EXTEND states.
# Must be high enough to overcome gravity and stand upright.
# 200K dyne-cm ≈ 20 N·m, strong enough for standing.
DEFAULT_MOTOR_MAX_FORCE: float = 200000.0

# Maximum torque when in RELAX state (0 = no resistance).
# Relaxed joints let gravity move the limb freely.
RELAX_MAX_FORCE: float = 0.0
