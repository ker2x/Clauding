import pygame
import numpy as np
import torch
import random
import math
from dataclasses import dataclass
from collections import deque

pygame.init()

WIDTH, HEIGHT = 1200, 800
FPS = 60
MAX_PARTICLES = 3000

# Color groups (RGB)
COLORS = {
    0: (255, 100, 100),    # Red
    1: (100, 255, 100),    # Green
    2: (100, 100, 255),    # Blue
    3: (255, 255, 100),    # Yellow
    4: (255, 100, 255),    # Magenta
    5: (100, 255, 255),    # Cyan
}

NUM_GROUPS = len(COLORS)

class ParticleSwarm:
    def __init__(self):
        self.width = WIDTH
        self.height = HEIGHT
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Particle Swarm: Emergent Behavior")
        self.clock = pygame.time.Clock()
        self.running = True

        self.time = 0
        self.paused = False

        # Determine device first
        # Trails for visual effect (reduced resolution)
        self.trails = [deque(maxlen=20) for _ in range(MAX_PARTICLES)]
        self.trail_skip = 2  # Draw trail every 2nd frame

        # Determine device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")  # Apple Silicon GPU
        else:
            self.device = torch.device("cpu")

        # Particle storage - initialized by _spawn_particles
        self.positions = None
        self.velocities = None
        self.groups = None
        self.active = None
        self.num_particles = 0

        # Interaction matrix
        self.interaction_matrix = torch.tensor(self._create_interaction_matrix_np(),
                                               dtype=torch.float32, device=self.device)

        # Initialize with particles
        self._spawn_particles()

        print(f"PyTorch Device: {self.device}")
        print(f"Interaction Matrix (strength per group pair):")
        for i in range(NUM_GROUPS):
            print(f"  Group {i}: {self.interaction_matrix[i]}")

    def _create_interaction_matrix_np(self):
        """
        Create interaction matrix defining how groups affect each other.
        Tuned to create interesting emergent patterns.
        """
        matrix = np.zeros((NUM_GROUPS, NUM_GROUPS), dtype=np.float32)

        # Red (0) <-> Green (1): complex dance
        matrix[0, 1] = 0.5   # Red attracted to Green
        matrix[1, 0] = -0.4  # Green repelled by Red

        # Red (0) <-> Blue (2): mutual attraction
        matrix[0, 2] = 0.3
        matrix[2, 0] = 0.3

        # Red (0) <-> Yellow (3): weak attraction
        matrix[0, 3] = 0.2
        matrix[3, 0] = 0.15

        # Red (0) <-> Magenta (4): attraction
        matrix[0, 4] = 0.4
        matrix[4, 0] = 0.35

        # Red (0) <-> Cyan (5): repulsion
        matrix[0, 5] = -0.3
        matrix[5, 0] = -0.25

        # Green (1) <-> Blue (2): repulsion
        matrix[1, 2] = -0.5
        matrix[2, 1] = -0.45

        # Green (1) <-> Yellow (3): attraction
        matrix[1, 3] = 0.4
        matrix[3, 1] = 0.35

        # Green (1) <-> Magenta (4): repulsion
        matrix[1, 4] = -0.3
        matrix[4, 1] = -0.25

        # Green (1) <-> Cyan (5): attraction
        matrix[1, 5] = 0.5
        matrix[5, 1] = 0.45

        # Blue (2) <-> Yellow (3): repulsion
        matrix[2, 3] = -0.35
        matrix[3, 2] = -0.3

        # Blue (2) <-> Magenta (4): repulsion
        matrix[2, 4] = -0.4
        matrix[4, 2] = -0.35

        # Blue (2) <-> Cyan (5): attraction
        matrix[2, 5] = 0.55
        matrix[5, 2] = 0.5

        # Yellow (3) <-> Magenta (4): attraction
        matrix[3, 4] = 0.25
        matrix[4, 3] = 0.2

        # Yellow (3) <-> Cyan (5): weak repulsion
        matrix[3, 5] = -0.2
        matrix[5, 3] = -0.15

        # Magenta (4) <-> Cyan (5): mutual repulsion
        matrix[4, 5] = -0.5
        matrix[5, 4] = -0.45

        # Self-interaction (particles of same color)
        for i in range(NUM_GROUPS):
            matrix[i, i] = -0.15  # Slight repulsion within groups (prevents clumping)

        return matrix.astype(np.float32)

    def _spawn_particles(self):
        """Spawn initial particles"""
        positions = []
        velocities = []
        groups = []

        particles_per_group = 350

        for group_id in range(NUM_GROUPS):
            # Spawn in a region biased by group
            cx = (self.width / (NUM_GROUPS + 1)) * (group_id + 1)
            cy = self.height / 2

            for _ in range(particles_per_group):
                x = cx + random.gauss(0, 40)
                y = cy + random.gauss(0, 40)
                x = max(10, min(self.width - 10, x))
                y = max(10, min(self.height - 10, y))

                vx = random.uniform(-1, 1)
                vy = random.uniform(-1, 1)

                positions.append([x, y])
                velocities.append([vx, vy])
                groups.append(group_id)

        self.num_particles = len(positions)

        # Initialize torch tensors on GPU/MPS
        pos_array = np.zeros((MAX_PARTICLES, 2), dtype=np.float32)
        vel_array = np.zeros((MAX_PARTICLES, 2), dtype=np.float32)
        grp_array = np.zeros(MAX_PARTICLES, dtype=np.int32)
        act_array = np.zeros(MAX_PARTICLES, dtype=bool)

        pos_array[:self.num_particles] = positions
        vel_array[:self.num_particles] = velocities
        grp_array[:self.num_particles] = groups
        act_array[:self.num_particles] = True

        self.positions = torch.tensor(pos_array, device=self.device)
        self.velocities = torch.tensor(vel_array, device=self.device)
        self.groups = torch.tensor(grp_array, device=self.device, dtype=torch.int32)
        self.active = torch.tensor(act_array, device=self.device)

    def _physics_update_torch(self, positions, velocities, groups, interaction_matrix, dt):
        """
        Vectorized physics update using PyTorch.
        Compute forces between all particle pairs.
        """
        n = positions.shape[0]

        # Compute pairwise differences
        diff = positions[:, None, :] - positions[None, :, :]  # (n, n, 2)
        dist_sq = torch.sum(diff**2, dim=2) + 1e-6  # (n, n)
        dist = torch.sqrt(dist_sq)

        # Distance-dependent force scaling
        force_scalar = torch.zeros((n, n), device=self.device, dtype=torch.float32)

        # Repulsion zone (0 to 30 pixels)
        repulsion_mask = dist < 30
        force_scalar[repulsion_mask] = -1.0 / (dist[repulsion_mask] + 1)

        # Attraction zone (30 to 150 pixels)
        attraction_mask = (dist >= 30) & (dist < 150)
        force_scalar[attraction_mask] = 1.0 / (dist[attraction_mask] + 1)

        # Decay zone (beyond 150)
        decay_mask = dist >= 150
        force_scalar[decay_mask] = 0.2 / (dist[decay_mask] + 1)

        # Apply interaction matrix
        interaction_strengths = interaction_matrix[groups[:, None], groups[None, :]]  # (n, n)
        force_scalar = force_scalar * interaction_strengths * 3.0

        # Normalize differences for direction
        direction = diff / (dist[:, :, None] + 1e-6)

        # Compute forces
        forces = torch.sum(force_scalar[:, :, None] * direction, dim=1)  # (n, 2)

        # Update velocities
        new_velocities = velocities + forces * dt * 0.5
        new_velocities = new_velocities * 0.92

        # Update positions
        new_positions = positions + new_velocities * dt

        # Boundary conditions
        margin = 10
        new_positions[:, 0] = torch.clamp(new_positions[:, 0], margin, WIDTH - margin)
        new_positions[:, 1] = torch.clamp(new_positions[:, 1], margin, HEIGHT - margin)

        # Velocity reversal on boundary hit
        hit_x_boundary = (new_positions[:, 0] <= margin) | (new_positions[:, 0] >= WIDTH - margin)
        hit_y_boundary = (new_positions[:, 1] <= margin) | (new_positions[:, 1] >= HEIGHT - margin)

        new_velocities[hit_x_boundary, 0] *= -1.0
        new_velocities[hit_y_boundary, 1] *= -1.0

        return new_positions, new_velocities

    def update(self):
        """Update simulation"""
        if self.paused:
            return

        dt = 0.016  # ~60 FPS

        self.positions, self.velocities = self._physics_update_torch(
            self.positions, self.velocities, self.groups,
            self.interaction_matrix, dt
        )

        # Update trails occasionally (on CPU side)
        if self.time % self.trail_skip == 0:
            positions_np = self.positions.detach().cpu().numpy()
            for i in range(self.num_particles):
                self.trails[i].append(tuple(positions_np[i]))

        self.time += 1

    def draw(self):
        """Render particles and trails"""
        self.screen.fill((15, 15, 20))

        positions_np = self.positions[:self.num_particles].detach().cpu().numpy()
        groups_np = self.groups[:self.num_particles].detach().cpu().numpy()

        # Draw trails
        for i in range(self.num_particles):
            if len(self.trails[i]) > 1:
                trail = list(self.trails[i])
                color = COLORS[groups_np[i]]
                for j in range(len(trail) - 1):
                    p1, p2 = trail[j], trail[j + 1]
                    intensity = int(200 * (j / len(trail)))
                    dimmed_color = tuple(c // 2 for c in color)
                    pygame.draw.line(self.screen, dimmed_color,
                                   (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 1)

        # Draw particles
        for i in range(self.num_particles):
            x, y = positions_np[i]
            group = groups_np[i]
            color = COLORS[group]
            pygame.draw.circle(self.screen, color, (int(x), int(y)), 3)

        # Draw info
        font = pygame.font.Font(None, 20)
        info = f"Particles: {self.num_particles} | Time: {self.time} | "
        info += f"P=Pause | R=Reset | Q=Quit"
        if self.paused:
            info += " [PAUSED]"

        text_surf = font.render(info, True, (200, 200, 200))
        self.screen.blit(text_surf, (10, 10))

        # Draw legend
        legend_y = self.height - 30
        for group_id, color in COLORS.items():
            x_pos = 10 + group_id * 190
            pygame.draw.circle(self.screen, color, (x_pos, legend_y), 5)
            font_small = pygame.font.Font(None, 16)
            text = font_small.render(f"Group {group_id}", True, color)
            self.screen.blit(text, (x_pos + 15, legend_y - 8))

        pygame.display.flip()

    def handle_events(self):
        """Handle input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            if event.type == pygame.KEYDOWN:
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
    swarm = ParticleSwarm()
    swarm.run()
