import pygame
import numpy as np
import random
import math
from enum import Enum

import jax
import jax.numpy as jnp
from jax import jit

pygame.init()

WIDTH, HEIGHT = 1200, 800
FPS = 60

class PresetMode(Enum):
    COMPETITION = 1      # Species compete (mostly negative interactions)
    COOPERATION = 2      # Species cooperate (mostly positive)
    PREDATOR_PREY = 3    # Some species prey on others
    BALANCED = 4         # Mix of all interactions
    CHAOTIC = 5          # Random complex interactions

class CoupledReactionDiffusion:
    def __init__(self):
        self.width = WIDTH
        self.height = HEIGHT
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Coupled Reaction-Diffusion: Multi-Species Emergent Patterns")
        self.clock = pygame.time.Clock()
        self.running = True

        # 6 species (RGB YMMC): each has u and v grids
        self.u = [jnp.ones((HEIGHT, WIDTH), dtype=jnp.float32) for _ in range(6)]
        self.v = [jnp.zeros((HEIGHT, WIDTH), dtype=jnp.float32) for _ in range(6)]

        # Species colors
        self.colors = [
            (255, 100, 100),    # Red
            (100, 255, 100),    # Green
            (100, 100, 255),    # Blue
            (255, 255, 100),    # Yellow
            (255, 100, 255),    # Magenta
            (100, 255, 255),    # Cyan
        ]

        # Interaction matrix (how species affect each other)
        # interaction[i, j] > 0 means species i activates species j
        # interaction[i, j] < 0 means species i inhibits species j
        self.preset = PresetMode.CHAOTIC
        self._create_interaction_matrix()

        self.time = 0
        self.pause = False

        # Pre-compile the update function
        self.update_step_jit = jit(self._update_step_impl)

        self._init_seed()

        print(f"JAX Device: {jax.default_backend()}")
        print(f"Preset: {self.preset.name}")

    def _create_interaction_matrix(self):
        """Create interaction matrix based on preset"""
        if self.preset == PresetMode.COMPETITION:
            # Mostly negative interactions (species compete) - moderate strength
            self.interaction_matrix = np.array([
                [-0.15, -0.18, -0.15, -0.1, -0.16, -0.12],
                [-0.16, -0.15, -0.2, -0.12, -0.15, -0.18],
                [-0.15, -0.18, -0.15, -0.16, -0.12, -0.15],
                [-0.12, -0.15, -0.16, -0.15, -0.18, -0.1],
                [-0.18, -0.15, -0.12, -0.16, -0.15, -0.2],
                [-0.15, -0.16, -0.15, -0.12, -0.18, -0.15],
            ], dtype=np.float32)

        elif self.preset == PresetMode.COOPERATION:
            # Mostly positive interactions - moderate strength
            self.interaction_matrix = np.array([
                [0.08, 0.15, 0.12, 0.1, 0.14, 0.12],
                [0.14, 0.08, 0.15, 0.12, 0.1, 0.14],
                [0.12, 0.14, 0.08, 0.15, 0.12, 0.1],
                [0.1, 0.12, 0.14, 0.08, 0.15, 0.12],
                [0.15, 0.1, 0.12, 0.14, 0.08, 0.15],
                [0.12, 0.15, 0.1, 0.12, 0.14, 0.08],
            ], dtype=np.float32)

        elif self.preset == PresetMode.PREDATOR_PREY:
            # Some species prey on others - moderate strength
            self.interaction_matrix = np.array([
                [-0.12,  0.2,   0.15,  0.12,  0.18, -0.15],
                [-0.18, -0.12, -0.22,  0.18, -0.15,  0.22],
                [ 0.15, -0.18, -0.12, -0.15, -0.18,  0.24],
                [ 0.1,   0.15, -0.15, -0.12,  0.15, -0.12],
                [ 0.15, -0.15, -0.15,  0.1,  -0.12, -0.22],
                [-0.15,  0.2,   0.22, -0.1,  -0.2,  -0.1],
            ], dtype=np.float32)

        elif self.preset == PresetMode.BALANCED:
            # Mix of interactions - moderate strength
            self.interaction_matrix = np.array([
                [-0.1,   0.15,  0.12,  0.1,   0.14, -0.1],
                [-0.12, -0.1,  -0.15,  0.14, -0.08,  0.15],
                [ 0.12, -0.14, -0.1,  -0.12, -0.14,  0.16],
                [ 0.1,   0.12, -0.12, -0.1,   0.14, -0.08],
                [ 0.15, -0.08, -0.12,  0.1,  -0.1,  -0.15],
                [-0.08,  0.16,  0.15, -0.1,  -0.14, -0.1],
            ], dtype=np.float32)

        elif self.preset == PresetMode.CHAOTIC:
            # Random interactions (moderate scale)
            self.interaction_matrix = np.random.uniform(-0.2, 0.2, (6, 6)).astype(np.float32)
            # Keep diagonal small (self-interaction)
            for i in range(6):
                self.interaction_matrix[i, i] = random.uniform(-0.1, 0.05)

    def _init_seed(self):
        """Initialize with random seed patterns"""
        u_list = [np.ones((HEIGHT, WIDTH), dtype=np.float32) for _ in range(6)]
        v_list = [np.zeros((HEIGHT, WIDTH), dtype=np.float32) for _ in range(6)]

        # Create different seed patterns for each species
        for species in range(6):
            # Random spots
            for _ in range(15):
                x = random.randint(10, WIDTH-10)
                y = random.randint(10, HEIGHT-10)
                r = random.randint(5, 15)
                for i in range(max(0, y-r), min(HEIGHT, y+r)):
                    for j in range(max(0, x-r), min(WIDTH, x+r)):
                        if (i-y)**2 + (j-x)**2 <= r**2:
                            v_list[species][i, j] = 1.0

        self.u = [jnp.array(u, dtype=jnp.float32) for u in u_list]
        self.v = [jnp.array(v, dtype=jnp.float32) for v in v_list]

    def _update_step_impl(self, u_list, v_list, feed, kill, Du, Dv, dt, interaction_matrix):
        """Vectorized coupled update using JAX (JIT compiled)"""
        u_new = []
        v_new = []

        for species in range(6):
            u = u_list[species]
            v = v_list[species]

            # Compute Laplacian using roll (toroidal wrapping)
            lap_u = (jnp.roll(u, 1, axis=0) + jnp.roll(u, -1, axis=0) +
                     jnp.roll(u, 1, axis=1) + jnp.roll(u, -1, axis=1) - 4 * u)
            lap_v = (jnp.roll(v, 1, axis=0) + jnp.roll(v, -1, axis=0) +
                     jnp.roll(v, 1, axis=1) + jnp.roll(v, -1, axis=1) - 4 * v)

            # Base Gray-Scott reaction
            u_react = u + dt * (Du * lap_u - u * v**2 + feed * (1 - u))
            v_react = v + dt * (Dv * lap_v + u * v**2 - (kill + feed) * v)

            # Add cross-species interaction (stronger for visible boundaries)
            cross_u = jnp.zeros_like(u)
            cross_v = jnp.zeros_like(v)

            for other in range(6):
                if other == species:
                    continue
                interaction = interaction_matrix[species, other]
                # Species interaction: other species' v affects this species' u and v
                cross_u += interaction * v_list[other] * 0.08
                cross_v += interaction * v_list[other] * 0.05

            u_new_val = u_react + cross_u
            v_new_val = v_react + cross_v

            # Clamp values
            u_new.append(jnp.clip(u_new_val, 0, 1))
            v_new.append(jnp.clip(v_new_val, 0, 1))

        return u_new, v_new

    def update(self):
        """Update simulation"""
        if self.pause:
            return

        # Parameters (tuned for interesting patterns)
        feed = 0.035
        kill = 0.062
        Du = 0.16
        Dv = 0.08
        dt = 1.0
        steps = 6

        # Run update steps
        for _ in range(steps):
            self.u, self.v = self.update_step_jit(
                self.u, self.v, feed, kill, Du, Dv, dt,
                jnp.array(self.interaction_matrix)
            )

        self.time += 1

    def draw(self):
        """Render to screen with multi-species coloring"""
        # Convert JAX arrays to numpy
        v_np = [np.array(v) for v in self.v]

        # Vectorized color blending (no loops!)
        r_val = v_np[0]  # Red species
        g_val = v_np[1]  # Green species
        b_val = v_np[2]  # Blue species
        y_val = v_np[3]  # Yellow species
        m_val = v_np[4]  # Magenta species
        c_val = v_np[5]  # Cyan species

        # Mix colors based on species concentrations - bright and vibrant
        color_r = r_val * 255 + y_val * 200 + m_val * 200
        color_g = g_val * 255 + y_val * 200 + c_val * 200
        color_b = b_val * 255 + m_val * 200 + c_val * 200

        # Stack and clip all at once
        img_array = np.stack([color_r, color_g, color_b], axis=2)
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        # Blit to pygame
        surf = pygame.surfarray.make_surface(img_array.swapaxes(0, 1))
        self.screen.blit(surf, (0, 0))

        # Draw info
        font = pygame.font.Font(None, 18)
        info_text = f"Preset: {self.preset.name} | "
        info_text += f"Time: {self.time} | "
        info_text += f"Backend: {jax.default_backend()} | "
        info_text += f"Keys: 1-5=Presets | SPACE=Reset | P=Pause | Q=Quit"
        text_surf = font.render(info_text, True, (255, 255, 255))
        self.screen.blit(text_surf, (10, 10))

        pygame.display.flip()

    def handle_events(self):
        """Handle input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self._init_seed()
                    self.time = 0
                elif event.key == pygame.K_p:
                    self.pause = not self.pause
                elif event.key == pygame.K_1:
                    self.preset = PresetMode.COMPETITION
                    self._create_interaction_matrix()
                    self._init_seed()
                    self.time = 0
                elif event.key == pygame.K_2:
                    self.preset = PresetMode.COOPERATION
                    self._create_interaction_matrix()
                    self._init_seed()
                    self.time = 0
                elif event.key == pygame.K_3:
                    self.preset = PresetMode.PREDATOR_PREY
                    self._create_interaction_matrix()
                    self._init_seed()
                    self.time = 0
                elif event.key == pygame.K_4:
                    self.preset = PresetMode.BALANCED
                    self._create_interaction_matrix()
                    self._init_seed()
                    self.time = 0
                elif event.key == pygame.K_5:
                    self.preset = PresetMode.CHAOTIC
                    self._create_interaction_matrix()
                    self._init_seed()
                    self.time = 0

    def run(self):
        """Main loop"""
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)

        pygame.quit()

if __name__ == "__main__":
    sim = CoupledReactionDiffusion()
    sim.run()
