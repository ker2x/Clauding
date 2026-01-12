import pygame
import numpy as np
import numba as nb
import random
import math
from collections import deque

WIDTH, HEIGHT = 1200, 800
FPS = 60
PARTICLE_COUNT = 3000

COLORS = [
    (255, 100, 100),    # Red
    (100, 255, 100),    # Green
    (100, 100, 255),    # Blue
    (255, 255, 100),    # Yellow
    (255, 100, 255),    # Magenta
    (100, 255, 255),    # Cyan
]

NUM_GROUPS = len(COLORS)

INTERACTION_MATRIX = np.array([
    [-0.15,  0.5,   0.3,   0.2,   0.4,  -0.3 ],
    [-0.4,  -0.15, -0.5,   0.4,  -0.3,   0.5 ],
    [ 0.3,  -0.45, -0.15, -0.35, -0.4,   0.55],
    [ 0.15,  0.35, -0.3,  -0.15,  0.25, -0.2 ],
    [ 0.35, -0.25, -0.35,  0.2,  -0.15, -0.5 ],
    [-0.25,  0.45,  0.5,  -0.15, -0.45, -0.15],
], dtype=np.float32)

@nb.njit(parallel=True, cache=True)
def update_particles_jitted(positions, velocities, groups, interaction_matrix, dt, width, height, num_particles):
    """
    JIT-compiled physics update.
    This runs at near-C speed with parallel execution on CPU.
    """
    margin = 10.0

    for i in nb.prange(num_particles):
        px, py = positions[i, 0], positions[i, 1]
        vx, vy = velocities[i, 0], velocities[i, 1]
        gi = int(groups[i])

        # Compute force from all other particles
        fx, fy = 0.0, 0.0

        for j in range(num_particles):
            if i == j:
                continue

            ox, oy = positions[j, 0], positions[j, 1]
            gj = int(groups[j])

            # Distance
            dx = ox - px
            dy = oy - py
            dist_sq = dx*dx + dy*dy + 1e-6
            dist = math.sqrt(dist_sq)

            # Distance-dependent force (VERY strong)
            if dist < 30.0:
                force_mag = -25.0 / (dist + 0.5)
            elif dist < 150.0:
                force_mag = 20.0 / (dist + 0.5)
            else:
                force_mag = 5.0 / (dist + 0.5)

            # Apply interaction strength
            interaction = interaction_matrix[gi, gj]
            force_mag = force_mag * interaction * 5.0

            # Direction
            dir_x = dx / (dist + 1e-6)
            dir_y = dy / (dist + 1e-6)

            fx += force_mag * dir_x
            fy += force_mag * dir_y

        # Update velocity
        vx = (vx + fx * dt) * 0.80
        vy = (vy + fy * dt) * 0.80

        # Update position
        px += vx * dt
        py += vy * dt

        # Boundary conditions
        if px < margin:
            px = margin
            vx = -vx
        if px > width - margin:
            px = width - margin
            vx = -vx
        if py < margin:
            py = margin
            vy = -vy
        if py > height - margin:
            py = height - margin
            vy = -vy

        # Write back
        positions[i, 0] = px
        positions[i, 1] = py
        velocities[i, 0] = vx
        velocities[i, 1] = vy


class ParticleSwarmFast:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Particle Swarm: Fast (Numba JIT)")
        self.clock = pygame.time.Clock()
        self.running = True

        self.time = 0
        self.paused = False

        # Initialize particles
        self.positions = np.zeros((PARTICLE_COUNT, 2), dtype=np.float32)
        self.velocities = np.zeros((PARTICLE_COUNT, 2), dtype=np.float32)
        self.groups = np.zeros(PARTICLE_COUNT, dtype=np.int32)

        self._spawn_particles()

        # Trails
        self.trails = [deque(maxlen=15) for _ in range(PARTICLE_COUNT)]
        self.trail_counter = 0

        print("Numba JIT compiler warming up...")
        # Warm up JIT
        test_pos = np.zeros((100, 2), dtype=np.float32)
        test_vel = np.zeros((100, 2), dtype=np.float32)
        test_groups = np.zeros(100, dtype=np.int32)
        update_particles_jitted(test_pos, test_vel, test_groups,
                               INTERACTION_MATRIX, 0.016, float(WIDTH), float(HEIGHT), 100)
        print("JIT compilation complete! Ready to go.")

    def _spawn_particles(self):
        """Initialize particle positions and velocities"""
        particles_per_group = PARTICLE_COUNT // NUM_GROUPS

        idx = 0
        for group_id in range(NUM_GROUPS):
            cx = (WIDTH / (NUM_GROUPS + 1)) * (group_id + 1)
            cy = HEIGHT / 2

            for _ in range(particles_per_group):
                x = cx + random.gauss(0, 40)
                y = cy + random.gauss(0, 40)
                x = max(10, min(WIDTH - 10, x))
                y = max(10, min(HEIGHT - 10, y))

                vx = random.uniform(-1, 1)
                vy = random.uniform(-1, 1)

                self.positions[idx] = [x, y]
                self.velocities[idx] = [vx, vy]
                self.groups[idx] = group_id
                idx += 1

    def update(self):
        """Update particles using JIT-compiled physics"""
        if self.paused:
            return

        # Call JIT-compiled update
        update_particles_jitted(self.positions, self.velocities, self.groups,
                               INTERACTION_MATRIX, 0.016, float(WIDTH), float(HEIGHT),
                               PARTICLE_COUNT)

        # Update trails occasionally
        self.trail_counter += 1
        if self.trail_counter % 2 == 0:
            for i in range(PARTICLE_COUNT):
                self.trails[i].append((self.positions[i, 0], self.positions[i, 1]))

        self.time += 1

    def draw(self):
        """Render to screen"""
        self.screen.fill((15, 15, 20))

        # Draw trails
        for i in range(PARTICLE_COUNT):
            if len(self.trails[i]) > 1:
                trail = list(self.trails[i])
                group = self.groups[i]
                color = COLORS[group]
                dimmed = tuple(c // 2 for c in color)

                for j in range(len(trail) - 1):
                    p1, p2 = trail[j], trail[j + 1]
                    pygame.draw.line(self.screen, dimmed,
                                   (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 1)

        # Draw particles
        for i in range(PARTICLE_COUNT):
            x, y = self.positions[i]
            group = self.groups[i]
            color = COLORS[group]
            pygame.draw.circle(self.screen, color, (int(x), int(y)), 3)

        # Draw info
        font = pygame.font.Font(None, 20)
        info = f"Particles: {PARTICLE_COUNT} | Time: {self.time} | "
        info += f"P=Pause | R=Reset | Q=Quit"
        if self.paused:
            info += " [PAUSED]"

        text_surf = font.render(info, True, (200, 200, 200))
        self.screen.blit(text_surf, (10, 10))

        pygame.display.flip()

    def handle_events(self):
        """Handle input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.running = False
                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self._spawn_particles()
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
    swarm = ParticleSwarmFast()
    swarm.run()
