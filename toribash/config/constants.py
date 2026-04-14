"""Game constants for Toribash 2D."""

# Physics
GRAVITY = (0, -900)  # cm/s^2 (downward)
DT = 1 / 60  # physics timestep
SPACE_ITERATIONS = 20  # pymunk solver iterations for stability

# Game
STEPS_PER_TURN = 30  # physics frames per turn (~0.5s real time)
MAX_TURNS = 20  # turns before match ends

# Arena
GROUND_Y = 50.0  # ground surface y-coordinate (cm)
ARENA_WIDTH = 600.0  # arena width (cm)
ARENA_HEIGHT = 400.0  # arena height (cm)
SPAWN_OFFSET_X = 100.0  # horizontal offset from center for each fighter

# Collision categories
CAT_GROUND = 0b0001
CAT_PLAYER_A = 0b0010
CAT_PLAYER_B = 0b0100

# Damage
DAMAGE_IMPULSE_THRESHOLD = 500.0  # minimum impulse to register as damage
DISMEMBER_IMPULSE = 5000.0  # impulse to detach a limb

# Joint motor
DEFAULT_MOTOR_RATE = 15.0  # rad/s for contract/extend
DEFAULT_MOTOR_MAX_FORCE = 50000.0  # max force for hold/contract/extend
RELAX_MAX_FORCE = 0.0  # force when relaxed
