import pygame
import numpy as np
import math
import random
from enum import Enum
from dataclasses import dataclass
from collections import deque

# Initialize Pygame
pygame.init()

# Configuration
WIDTH, HEIGHT = 1200, 800
FPS = 60

class Mode(Enum):
    PARTICLES = 1
    FRACTAL = 2
    CELLULAR = 3
    ATTRACTORS = 4
    FLOW_FIELD = 5

@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    ax: float
    ay: float
    life: float = 1.0
    trail: list = None

    def __post_init__(self):
        if self.trail is None:
            self.trail = deque(maxlen=30)

class GenerativeArtCanvas:
    def __init__(self):
        self.width = WIDTH
        self.height = HEIGHT
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Generative Art Workspace")
        self.clock = pygame.time.Clock()
        self.running = True

        self.mode = Mode.PARTICLES
        self.time = 0

        # Mode-specific state
        self.particles = []
        self.cellular_grid = None
        self.attractor_points = deque(maxlen=10000)
        self.fractal_depth = 1

        self.init_particles()
        self.init_cellular()

    def init_particles(self):
        """Initialize particle system"""
        self.particles = []
        num_particles = 300
        for _ in range(num_particles):
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            self.particles.append(Particle(
                x=x, y=y,
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed,
                ax=0, ay=0
            ))

    def init_cellular(self):
        """Initialize cellular automata grid"""
        cell_size = 4
        self.cell_size = cell_size
        grid_w = self.width // cell_size
        grid_h = self.height // cell_size
        self.cellular_grid = np.random.choice([0, 1], size=(grid_h, grid_w), p=[0.8, 0.2])

    def update_particles(self):
        """Update particle system with various forces"""
        time_factor = self.time / 100

        for particle in self.particles:
            # Perlin-like noise using sine waves (creating organic movement)
            noise_x = math.sin(particle.x * 0.005 + time_factor) * 0.3
            noise_y = math.cos(particle.y * 0.005 + time_factor) * 0.3

            # Attraction to center (subtle)
            dx = self.width / 2 - particle.x
            dy = self.height / 2 - particle.y
            dist = math.sqrt(dx*dx + dy*dy) + 1
            attraction = 0.0005

            particle.ax = (dx / dist) * attraction + noise_x
            particle.ay = (dy / dist) * attraction + noise_y

            # Also add some repulsion from edges
            edge_margin = 100
            if particle.x < edge_margin:
                particle.ax += 0.02
            if particle.x > self.width - edge_margin:
                particle.ax -= 0.02
            if particle.y < edge_margin:
                particle.ay += 0.02
            if particle.y > self.height - edge_margin:
                particle.ay -= 0.02

            # Update velocity and position
            particle.vx += particle.ax
            particle.vy += particle.ay

            # Damping
            particle.vx *= 0.98
            particle.vy *= 0.98

            particle.x += particle.vx
            particle.y += particle.vy

            # Store position for trail
            particle.trail.append((int(particle.x), int(particle.y)))

            # Wrap around edges
            if particle.x < 0:
                particle.x = self.width
            if particle.x > self.width:
                particle.x = 0
            if particle.y < 0:
                particle.y = self.height
            if particle.y > self.height:
                particle.y = 0

    def update_cellular(self):
        """Update cellular automata with Conway's Game of Life rules"""
        if self.cellular_grid is None:
            return

        new_grid = self.cellular_grid.copy()
        h, w = self.cellular_grid.shape

        for i in range(h):
            for j in range(w):
                # Count neighbors
                neighbors = 0
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = (i + di) % h, (j + dj) % w
                        neighbors += self.cellular_grid[ni, nj]

                # Apply rules with some randomness for variation
                if self.cellular_grid[i, j] == 1:
                    if neighbors < 2 or neighbors > 3:
                        new_grid[i, j] = 0
                else:
                    if neighbors == 3 or (neighbors == 2 and random.random() < 0.05):
                        new_grid[i, j] = 1

        self.cellular_grid = new_grid

    def update_attractors(self):
        """Update strange attractor (Lorenz system)"""
        if not hasattr(self, 'attractor_state'):
            self.attractor_state = np.array([1.0, 1.0, 1.0])

        # Lorenz system parameters
        sigma, rho, beta = 10, 28, 8/3

        # Runge-Kutta 4th order integration
        h = 0.001
        x, y, z = self.attractor_state

        for _ in range(10):
            k1_x = sigma * (y - x)
            k1_y = x * (rho - z) - y
            k1_z = x * y - beta * z

            k2_x = sigma * ((y + k1_y*h/2) - (x + k1_x*h/2))
            k2_y = (x + k1_x*h/2) * (rho - (z + k1_z*h/2)) - (y + k1_y*h/2)
            k2_z = (x + k1_x*h/2) * (y + k1_y*h/2) - beta * (z + k1_z*h/2)

            k3_x = sigma * ((y + k2_y*h/2) - (x + k2_x*h/2))
            k3_y = (x + k2_x*h/2) * (rho - (z + k2_z*h/2)) - (y + k2_y*h/2)
            k3_z = (x + k2_x*h/2) * (y + k2_y*h/2) - beta * (z + k2_z*h/2)

            k4_x = sigma * ((y + k3_y*h) - (x + k3_x*h))
            k4_y = (x + k3_x*h) * (rho - (z + k3_z*h)) - (y + k3_y*h)
            k4_z = (x + k3_x*h) * (y + k3_y*h) - beta * (z + k3_z*h)

            x += h * (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
            y += h * (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6
            z += h * (k1_z + 2*k2_z + 2*k3_z + k4_z) / 6

        self.attractor_state = np.array([x, y, z])

        # Map to screen coordinates
        scale = 15
        screen_x = self.width / 2 + x * scale
        screen_y = self.height / 2 + y * scale

        if 0 <= screen_x < self.width and 0 <= screen_y < self.height:
            self.attractor_points.append((int(screen_x), int(screen_y), z))

    def draw_particles(self, surface):
        """Draw particle system with trails"""
        surface.fill((10, 10, 15))

        # Draw trails
        for particle in self.particles:
            if len(particle.trail) > 1:
                points = list(particle.trail)
                if len(points) > 1:
                    for i in range(len(points)-1):
                        alpha = int(255 * (i / len(points)))
                        color = (100 + alpha // 3, 150 + alpha // 4, 200)
                        pygame.draw.line(surface, color, points[i], points[i+1], 1)

        # Draw particles
        for particle in self.particles:
            color = (50, 150, 255)
            pygame.draw.circle(surface, color, (int(particle.x), int(particle.y)), 3)

    def draw_cellular(self, surface):
        """Draw cellular automata"""
        surface.fill((20, 20, 20))

        if self.cellular_grid is not None:
            h, w = self.cellular_grid.shape
            for i in range(h):
                for j in range(w):
                    if self.cellular_grid[i, j] == 1:
                        x = j * self.cell_size
                        y = i * self.cell_size

                        # Color based on neighbor count for visual interest
                        neighbors = 0
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                if di == 0 and dj == 0:
                                    continue
                                ni, nj = (i + di) % h, (j + dj) % w
                                neighbors += self.cellular_grid[ni, nj]

                        intensity = min(255, 50 + neighbors * 30)
                        color = (intensity // 2, intensity, intensity)
                        pygame.draw.rect(surface, color,
                                       (x, y, self.cell_size, self.cell_size))

    def draw_attractors(self, surface):
        """Draw strange attractor points"""
        surface.fill((5, 5, 10))

        # Draw trails of attractor
        if len(self.attractor_points) > 1:
            points_list = list(self.attractor_points)
            for i in range(len(points_list)-1):
                p1 = points_list[i]
                p2 = points_list[i+1]
                # Color based on Z value (height)
                z_norm = (p1[2] + 30) / 60
                z_norm = max(0, min(1, z_norm))
                r = int(255 * z_norm)
                g = int(200 * (1 - abs(z_norm - 0.5) * 2))
                b = int(255 * (1 - z_norm))
                pygame.draw.line(surface, (r, g, b), (p1[0], p1[1]), (p2[0], p2[1]), 1)

        # Draw current point
        if len(self.attractor_points) > 0:
            p = self.attractor_points[-1]
            pygame.draw.circle(surface, (255, 150, 100), (p[0], p[1]), 2)

    def draw_fractal(self, surface):
        """Draw recursive fractal tree"""
        surface.fill((5, 10, 15))

        def draw_branch(x, y, angle, length, depth):
            if depth == 0 or length < 2:
                return

            # End point
            end_x = x + math.cos(angle) * length
            end_y = y + math.sin(angle) * length

            # Color based on depth
            color_val = int(255 * (depth / 12))
            color = (color_val, 100 + min(100, color_val // 2), max(50, 200 - color_val // 2))

            pygame.draw.line(surface, color, (x, y), (end_x, end_y), max(1, depth // 2))

            # Recursive branches with slight randomness
            angle_offset = 0.4 + math.sin(self.time / 100) * 0.1
            draw_branch(end_x, end_y, angle - angle_offset, length * 0.7, depth - 1)
            draw_branch(end_x, end_y, angle + angle_offset, length * 0.7, depth - 1)

        start_x = self.width / 2
        start_y = self.height
        start_angle = -math.pi / 2
        start_length = 100

        draw_branch(start_x, start_y, start_angle, start_length, 12)

    def draw_flow_field(self, surface):
        """Draw a flow field based on Perlin-like noise"""
        surface.fill((15, 15, 20))

        resolution = 20
        for y in range(0, self.height, resolution):
            for x in range(0, self.width, resolution):
                # Create flow angle using sine/cosine waves
                angle = math.sin(x * 0.01 + self.time * 0.02) * 0.5 + \
                       math.cos(y * 0.01 + self.time * 0.02) * 0.5

                # Draw flow line
                line_length = 15
                end_x = x + math.cos(angle) * line_length
                end_y = y + math.sin(angle) * line_length

                # Color based on angle
                color = (
                    int(128 + 127 * math.sin(angle)),
                    int(128 + 127 * math.cos(angle + math.pi/2)),
                    int(128 + 127 * math.sin(angle + math.pi))
                )

                pygame.draw.line(surface, color, (x, y), (end_x, end_y), 2)

    def update(self):
        """Update simulation"""
        if self.mode == Mode.PARTICLES:
            self.update_particles()
        elif self.mode == Mode.CELLULAR:
            self.update_cellular()
        elif self.mode == Mode.ATTRACTORS:
            self.update_attractors()

        self.time += 1

    def draw(self):
        """Draw based on current mode"""
        if self.mode == Mode.PARTICLES:
            self.draw_particles(self.screen)
        elif self.mode == Mode.FRACTAL:
            self.draw_fractal(self.screen)
        elif self.mode == Mode.CELLULAR:
            self.draw_cellular(self.screen)
        elif self.mode == Mode.ATTRACTORS:
            self.draw_attractors(self.screen)
        elif self.mode == Mode.FLOW_FIELD:
            self.draw_flow_field(self.screen)

        # Draw mode indicator and instructions
        font = pygame.font.Font(None, 24)
        mode_text = f"Mode: {self.mode.name} | Keys: 1=Particles, 2=Fractal, 3=Cellular, 4=Attractors, 5=FlowField | SPACE=Reset | Q=Quit"
        text_surface = font.render(mode_text, True, (200, 200, 200))
        self.screen.blit(text_surface, (10, 10))

        pygame.display.flip()

    def handle_events(self):
        """Handle input events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.running = False
                elif event.key == pygame.K_1:
                    self.mode = Mode.PARTICLES
                    self.time = 0
                elif event.key == pygame.K_2:
                    self.mode = Mode.FRACTAL
                    self.time = 0
                elif event.key == pygame.K_3:
                    self.mode = Mode.CELLULAR
                    self.init_cellular()
                    self.time = 0
                elif event.key == pygame.K_4:
                    self.mode = Mode.ATTRACTORS
                    self.attractor_points.clear()
                    self.time = 0
                elif event.key == pygame.K_5:
                    self.mode = Mode.FLOW_FIELD
                    self.time = 0
                elif event.key == pygame.K_SPACE:
                    if self.mode == Mode.PARTICLES:
                        self.init_particles()
                    elif self.mode == Mode.CELLULAR:
                        self.init_cellular()
                    elif self.mode == Mode.ATTRACTORS:
                        self.attractor_points.clear()
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
    canvas = GenerativeArtCanvas()
    canvas.run()
