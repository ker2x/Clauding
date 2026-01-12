import pygame
import numpy as np
import random
import math
from enum import Enum

import jax
import jax.numpy as jnp
from jax import jit

pygame.init()

WIDTH, HEIGHT = 1200, 650
FPS = 60

class Pattern(Enum):
    SPOTS = 1
    STRIPES = 2
    CORAL = 3
    WAVES = 4
    BUBBLES = 5
    SPIRAL = 6

class ReactionDiffusionGPU:
    def __init__(self):
        self.width = WIDTH
        self.height = HEIGHT
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Reaction-Diffusion (GPU Accelerated): Turing Patterns")
        self.clock = pygame.time.Clock()
        self.running = True

        # Reaction-diffusion grids (JAX arrays)
        self.u = jnp.ones((HEIGHT, WIDTH), dtype=jnp.float32)
        self.v = jnp.zeros((HEIGHT, WIDTH), dtype=jnp.float32)

        self.pattern = Pattern.SPOTS
        self.init_pattern()

        self.time = 0
        self.pause = False

        # Pre-compile the update function
        self.update_step_jit = jit(self._update_step_impl)

        print(f"JAX Device: {jax.default_backend()}")

    def init_pattern(self):
        """Initialize grid with pattern-specific seed"""
        u = np.ones((HEIGHT, WIDTH), dtype=np.float32)
        v = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

        if self.pattern == Pattern.SPOTS:
            for _ in range(100):
                x, y = random.randint(10, WIDTH-10), random.randint(10, HEIGHT-10)
                r = random.randint(5, 20)
                for i in range(max(0, y-r), min(HEIGHT, y+r)):
                    for j in range(max(0, x-r), min(WIDTH, x+r)):
                        if (i-y)**2 + (j-x)**2 <= r**2:
                            v[i, j] = 1.0

        elif self.pattern == Pattern.STRIPES:
            for i in range(HEIGHT):
                if (i // 40) % 2 == 0:
                    v[i, :] = 0.5 + random.uniform(-0.1, 0.1)

        elif self.pattern == Pattern.CORAL:
            cy, cx = HEIGHT // 2, WIDTH // 2
            for i in range(HEIGHT):
                for j in range(WIDTH):
                    dist = math.sqrt((i - cy)**2 + (j - cx)**2)
                    if 80 < dist < 120:
                        v[i, j] = 1.0 + random.uniform(-0.1, 0.1)

        elif self.pattern == Pattern.WAVES:
            for i in range(HEIGHT):
                for j in range(WIDTH):
                    v[i, j] = 0.5 + 0.3 * math.sin(j * 0.02)

        elif self.pattern == Pattern.BUBBLES:
            for _ in range(8):
                x = random.randint(50, WIDTH-50)
                y = random.randint(50, HEIGHT-50)
                r = random.randint(30, 60)
                for i in range(max(0, y-r), min(HEIGHT, y+r)):
                    for j in range(max(0, x-r), min(WIDTH, x+r)):
                        if (i-y)**2 + (j-x)**2 <= r**2:
                            v[i, j] = 1.0

        elif self.pattern == Pattern.SPIRAL:
            cy, cx = HEIGHT // 2, WIDTH // 2
            for i in range(HEIGHT):
                for j in range(WIDTH):
                    dy, dx = i - cy, j - cx
                    angle = math.atan2(dy, dx)
                    dist = math.sqrt(dy**2 + dx**2)
                    spiral = math.sin(dist * 0.1 - angle * 2)
                    if spiral > 0.5:
                        v[i, j] = spiral

        self.u = jnp.array(u, dtype=jnp.float32)
        self.v = jnp.array(v, dtype=jnp.float32)

    def _update_step_impl(self, u, v, feed, kill, Du, Dv, dt):
        """Vectorized update using JAX (JIT compiled)"""
        # Compute Laplacian using roll (toroidal wrapping)
        lap_u = (jnp.roll(u, 1, axis=0) + jnp.roll(u, -1, axis=0) +
                 jnp.roll(u, 1, axis=1) + jnp.roll(u, -1, axis=1) - 4 * u)
        lap_v = (jnp.roll(v, 1, axis=0) + jnp.roll(v, -1, axis=0) +
                 jnp.roll(v, 1, axis=1) + jnp.roll(v, -1, axis=1) - 4 * v)

        # Gray-Scott reaction-diffusion equations (vectorized)
        u_new = u + dt * (Du * lap_u - u * v**2 + feed * (1 - u))
        v_new = v + dt * (Dv * lap_v + u * v**2 - (kill + feed) * v)

        # Clamp values
        u_new = jnp.clip(u_new, 0, 1)
        v_new = jnp.clip(v_new, 0, 1)

        return u_new, v_new

    def update(self):
        """Update simulation"""
        if self.pause:
            return

        # Pattern-specific parameters
        if self.pattern == Pattern.SPOTS:
            feed, kill = 0.0367, 0.0649
            dt = 1.0
            steps = 2
        elif self.pattern == Pattern.STRIPES:
            feed, kill = 0.0545, 0.062
            dt = 1.0
            steps = 2
        elif self.pattern == Pattern.CORAL:
            feed, kill = 0.03, 0.055
            dt = 1.0
            steps = 2
        elif self.pattern == Pattern.WAVES:
            feed, kill = 0.025, 0.051
            dt = 1.0
            steps = 1
        elif self.pattern == Pattern.BUBBLES:
            feed, kill = 0.044, 0.051
            dt = 1.0
            steps = 2
        elif self.pattern == Pattern.SPIRAL:
            feed, kill = 0.0395, 0.0649
            dt = 1.0
            steps = 2

        Du, Dv = 0.16, 0.08

        # Run update steps
        for _ in range(steps):
            self.u, self.v = self.update_step_jit(self.u, self.v, feed, kill, Du, Dv, dt)

        self.time += 1

    def draw(self):
        """Render to screen with vectorized colorization"""
        # Convert JAX arrays to numpy for rendering
        v_np = np.array(self.v)
        u_np = np.array(self.u)

        # Vectorized color computation
        r = (v_np * 255).astype(np.uint8)
        g = ((1 - v_np) * u_np * 255).astype(np.uint8)
        b = (np.clip(0.5 + 0.5 * np.sin(v_np * math.pi), 0, 1) * 255).astype(np.uint8)

        # Stack channels
        img_array = np.stack([r, g, b], axis=2)

        # Blit to pygame
        surf = pygame.surfarray.make_surface(img_array.swapaxes(0, 1))
        self.screen.blit(surf, (0, 0))

        # Draw info
        font = pygame.font.Font(None, 20)
        info_text = f"Pattern: {self.pattern.name} | "
        info_text += f"Time: {self.time} | "
        info_text += f"Backend: {jax.default_backend()} | "
        info_text += f"Keys: 1-6=Patterns | SPACE=Reset | P=Pause | Q=Quit"
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
                    self.init_pattern()
                    self.time = 0
                elif event.key == pygame.K_p:
                    self.pause = not self.pause
                elif event.key == pygame.K_1:
                    self.pattern = Pattern.SPOTS
                    self.init_pattern()
                    self.time = 0
                elif event.key == pygame.K_2:
                    self.pattern = Pattern.STRIPES
                    self.init_pattern()
                    self.time = 0
                elif event.key == pygame.K_3:
                    self.pattern = Pattern.CORAL
                    self.init_pattern()
                    self.time = 0
                elif event.key == pygame.K_4:
                    self.pattern = Pattern.WAVES
                    self.init_pattern()
                    self.time = 0
                elif event.key == pygame.K_5:
                    self.pattern = Pattern.BUBBLES
                    self.init_pattern()
                    self.time = 0
                elif event.key == pygame.K_6:
                    self.pattern = Pattern.SPIRAL
                    self.init_pattern()
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
    canvas = ReactionDiffusionGPU()
    canvas.run()
