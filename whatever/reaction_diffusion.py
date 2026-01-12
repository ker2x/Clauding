import pygame
import numpy as np
import random
import math
from enum import Enum

pygame.init()

WIDTH, HEIGHT = 1400, 900
FPS = 60

class Pattern(Enum):
    SPOTS = 1
    STRIPES = 2
    CORAL = 3
    WAVES = 4
    BUBBLES = 5
    SPIRAL = 6

class ReactionDiffusionCanvas:
    def __init__(self):
        self.width = WIDTH
        self.height = HEIGHT
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Reaction-Diffusion: Turing Patterns")
        self.clock = pygame.time.Clock()
        self.running = True

        # Reaction-diffusion grid
        self.u = np.ones((HEIGHT, WIDTH), dtype=np.float32)
        self.v = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

        # Initialize with random seed
        self.pattern = Pattern.SPOTS
        self.init_pattern()

        self.time = 0
        self.pause = False

    def init_pattern(self):
        """Initialize grid with pattern-specific seed"""
        self.u = np.ones((HEIGHT, WIDTH), dtype=np.float32)
        self.v = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

        if self.pattern == Pattern.SPOTS:
            # Random spots scattered across
            for _ in range(100):
                x, y = random.randint(10, WIDTH-10), random.randint(10, HEIGHT-10)
                r = random.randint(5, 20)
                for i in range(max(0, y-r), min(HEIGHT, y+r)):
                    for j in range(max(0, x-r), min(WIDTH, x+r)):
                        if (i-y)**2 + (j-x)**2 <= r**2:
                            self.v[i, j] = 1.0

        elif self.pattern == Pattern.STRIPES:
            # Horizontal bands
            for i in range(HEIGHT):
                if (i // 40) % 2 == 0:
                    self.v[i, :] = 0.5 + random.uniform(-0.1, 0.1)

        elif self.pattern == Pattern.CORAL:
            # Ring seed in center
            cy, cx = HEIGHT // 2, WIDTH // 2
            for i in range(HEIGHT):
                for j in range(WIDTH):
                    dist = math.sqrt((i - cy)**2 + (j - cx)**2)
                    if 80 < dist < 120:
                        self.v[i, j] = 1.0 + random.uniform(-0.1, 0.1)

        elif self.pattern == Pattern.WAVES:
            # Wave pattern
            for i in range(HEIGHT):
                for j in range(WIDTH):
                    self.v[i, j] = 0.5 + 0.3 * math.sin(j * 0.02)

        elif self.pattern == Pattern.BUBBLES:
            # Large circles
            for _ in range(8):
                x = random.randint(50, WIDTH-50)
                y = random.randint(50, HEIGHT-50)
                r = random.randint(30, 60)
                for i in range(max(0, y-r), min(HEIGHT, y+r)):
                    for j in range(max(0, x-r), min(WIDTH, x+r)):
                        if (i-y)**2 + (j-x)**2 <= r**2:
                            self.v[i, j] = 1.0

        elif self.pattern == Pattern.SPIRAL:
            # Spiral seed
            cy, cx = HEIGHT // 2, WIDTH // 2
            for i in range(HEIGHT):
                for j in range(WIDTH):
                    dy, dx = i - cy, j - cx
                    angle = math.atan2(dy, dx)
                    dist = math.sqrt(dy**2 + dx**2)
                    spiral = math.sin(dist * 0.1 - angle * 2)
                    if spiral > 0.5:
                        self.v[i, j] = spiral

    def laplacian(self, grid):
        """Compute Laplacian (discrete approximation)"""
        lap = np.zeros_like(grid)
        h = grid.shape[0]
        w = grid.shape[1]

        for i in range(1, h-1):
            for j in range(1, w-1):
                lap[i, j] = (grid[i+1, j] + grid[i-1, j] +
                            grid[i, j+1] + grid[i, j-1] -
                            4 * grid[i, j])

        # Handle boundaries with wrapping
        lap[0, :] = lap[h-2, :]
        lap[h-1, :] = lap[1, :]
        lap[:, 0] = lap[:, w-2]
        lap[:, w-1] = lap[:, 1]

        return lap

    def update(self):
        """Update using Gray-Scott reaction-diffusion equations"""
        if self.pause:
            return

        # Gray-Scott parameters (different per pattern)
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
            feed, kill = 0.04, 0.06
            dt = 1.0
            steps = 2
        elif self.pattern == Pattern.SPIRAL:
            feed, kill = 0.0395, 0.0649
            dt = 1.0
            steps = 2

        # Diffusion rates
        Du, Dv = 0.16, 0.08

        for _ in range(steps):
            # Laplacian
            lap_u = self.laplacian(self.u)
            lap_v = self.laplacian(self.v)

            # Reaction-diffusion equation
            u_new = self.u + dt * (Du * lap_u - self.u * self.v**2 + feed * (1 - self.u))
            v_new = self.v + dt * (Dv * lap_v + self.u * self.v**2 - (kill + feed) * self.v)

            # Clamp values
            self.u = np.clip(u_new, 0, 1)
            self.v = np.clip(v_new, 0, 1)

        self.time += 1

    def draw(self):
        """Render to screen"""
        # Create RGB image from u and v
        img_array = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        # Colorize based on v concentration
        for i in range(HEIGHT):
            for j in range(WIDTH):
                v_val = self.v[i, j]
                u_val = self.u[i, j]

                # Create psychedelic colors
                r = int(255 * v_val)
                g = int(255 * (1 - v_val) * u_val)
                b = int(255 * (0.5 + 0.5 * math.sin(v_val * math.pi)))

                img_array[i, j] = [r, g, b]

        # Convert to pygame surface
        surf = pygame.surfarray.make_surface(img_array.swapaxes(0, 1))
        self.screen.blit(surf, (0, 0))

        # Draw info
        font = pygame.font.Font(None, 20)
        info_text = f"Pattern: {self.pattern.name} | "
        info_text += f"Time: {self.time} | "
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
    canvas = ReactionDiffusionCanvas()
    canvas.run()
