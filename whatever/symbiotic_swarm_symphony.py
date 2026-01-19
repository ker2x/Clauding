import pygame
import numpy as np
import random
import math
from enum import Enum

import jax
import jax.numpy as jnp
from jax import jit

pygame.init()
pygame.mixer.init()  # For optional audio

WIDTH, HEIGHT = 1200, 800
FPS = 60
NUM_PARTICLES = 8000
NUM_SPECIES = 3
TRAIL_SIZE = 256  # Lower resolution for trail grid
DIFFUSE_STEPS = 2
DECAY_RATE = 0.995  # Slower decay for longer trails

# Audio setup (optional)
AUDIO_ENABLED = True
if AUDIO_ENABLED:
    pygame.mixer.set_num_channels(4)

class Species(Enum):
    RED = 0
    GREEN = 1
    BLUE = 2

class SymbioticSwarmSymphony:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF)
        pygame.display.set_caption("Symbiotic Swarm Symphony: Emergent Art")
        self.clock = pygame.time.Clock()
        self.running = True

        # Particles: position, velocity, species
        self.positions = np.random.uniform(0, WIDTH, (NUM_PARTICLES, 2)).astype(np.float32)
        self.velocities = np.random.uniform(-50, 50, (NUM_PARTICLES, 2)).astype(np.float32)
        self.species = np.random.randint(0, NUM_SPECIES, NUM_PARTICLES)

        # Interaction matrix: attraction/repulsion between species
        self.interaction_matrix = np.array([
            [0.5, -0.3, 0.1],   # RED: self-attract, repel GREEN, attract BLUE
            [-0.3, 0.5, -0.1],  # GREEN: self-attract, repel RED, repel BLUE
            [0.1, -0.1, 0.5]    # BLUE: self-attract, repel GREEN, self-attract
        ], dtype=np.float32)

        # Trail system for chemical communication
        self.trail_grid = np.zeros((TRAIL_SIZE, TRAIL_SIZE, NUM_SPECIES), dtype=np.float32)
        self.trail_surface = pygame.Surface((TRAIL_SIZE, TRAIL_SIZE))

        # Parameters
        self.force_strength = 200.0
        self.friction = 0.95
        self.max_speed = 100.0
        self.deposit_rate = 1.0  # Increased for visibility
        self.sense_distance = 50.0

        # Color evolution
        self.species_colors = [
            np.array([255.0, 100.0, 100.0]),  # Red with tint
            np.array([100.0, 255.0, 100.0]),  # Green with tint
            np.array([100.0, 100.0, 255.0])   # Blue with tint
        ]
        self.color_mutation_rate = 0.01  # More frequent mutations

        # Audio tones for each species (simple sine waves)
        self.audio_channels = []
        if AUDIO_ENABLED:
            for i in range(NUM_SPECIES):
                channel = pygame.mixer.Channel(i)
                self.audio_channels.append(channel)

        # Pre-compile JAX functions for performance
        self.update_physics_jit = jit(self._update_physics_impl)
        self.update_trails_jit = jit(self._update_trails_impl)

        print(f"JAX Device: {jax.default_backend()}")

    def _update_physics_impl(self, positions, velocities, species, interaction_matrix,
                           force_strength, friction, max_speed, sense_distance):
        """JAX-compiled physics update"""
        num_particles = positions.shape[0]

        # Compute all pairwise distances at once (vectorized)
        diff = positions[:, None, :] - positions[None, :, :]  # [N, N, 2]
        dist = jnp.sqrt(jnp.sum(diff**2, axis=2))  # [N, N]

        # Create distance mask (avoid self-interaction)
        dist_mask = (dist < sense_distance) & (dist > 1e-6)  # [N, N]

        # Get species pairs
        species_i = species[:, None]  # [N, 1]
        species_j = species[None, :]  # [1, N]
        interaction_strength = interaction_matrix[species_i, species_j]  # [N, N]

        # Distance-dependent force curve
        # Repel very close, attract at medium distance, neutral far
        force_mag = jnp.where(dist < 20, -force_strength / (dist + 1),
                            jnp.where(dist < sense_distance,
                                    force_strength * interaction_strength / (dist + 1),
                                    0))

        # Apply distance mask
        force_mag = force_mag * dist_mask

        # Compute force vectors
        force_dir = diff / (dist[:, :, None] + 1e-6)  # [N, N, 2]
        forces = force_dir * force_mag[:, :, None]  # [N, N, 2]

        # Sum forces for each particle
        total_forces = jnp.sum(forces, axis=1)  # [N, 2]

        # Update velocities
        velocities = velocities + total_forces * 0.016  # dt approximation
        velocities = velocities * friction

        # Limit speed
        speed = jnp.sqrt(jnp.sum(velocities**2, axis=1))
        speed_clamped = jnp.minimum(speed, max_speed)
        velocities = velocities * (speed_clamped / (speed + 1e-6))[:, None]

        # Update positions
        positions = positions + velocities * 0.016

        # Wrap around screen
        positions = jnp.mod(positions, jnp.array([WIDTH, HEIGHT]))

        return positions, velocities

    def _update_trails_impl(self, trail_grid, decay_rate):
        """JAX-compiled trail diffusion and decay"""
        # Diffuse (simple box blur)
        diffused = (
            jnp.roll(trail_grid, 1, axis=0) + jnp.roll(trail_grid, -1, axis=0) +
            jnp.roll(trail_grid, 1, axis=1) + jnp.roll(trail_grid, -1, axis=1)
        ) * 0.25

        # Decay
        return diffused * decay_rate

    def update(self):
        # Convert to JAX arrays for computation
        positions_jax = jnp.array(self.positions)
        velocities_jax = jnp.array(self.velocities)
        species_jax = jnp.array(self.species)
        matrix_jax = jnp.array(self.interaction_matrix)

        # Physics update
        positions_jax, velocities_jax = self.update_physics_jit(
            positions_jax, velocities_jax, species_jax, matrix_jax,
            self.force_strength, self.friction, self.max_speed, self.sense_distance
        )

        # Convert back to numpy
        self.positions = np.array(positions_jax)
        self.velocities = np.array(velocities_jax)

        # Deposit trails
        grid_x = (self.positions[:, 0] / WIDTH * TRAIL_SIZE).astype(int) % TRAIL_SIZE
        grid_y = (self.positions[:, 1] / HEIGHT * TRAIL_SIZE).astype(int) % TRAIL_SIZE

        for i in range(NUM_PARTICLES):
            species_idx = self.species[i]
            self.trail_grid[grid_y[i], grid_x[i], species_idx] += self.deposit_rate

        # Update trails (diffuse and decay)
        trail_jax = jnp.array(self.trail_grid)
        trail_jax = self.update_trails_jit(trail_jax, DECAY_RATE)
        self.trail_grid = np.array(trail_jax)

        # Diffuse multiple times for smoother trails
        for _ in range(DIFFUSE_STEPS - 1):
            trail_jax = self.update_trails_jit(trail_jax, 1.0)
            self.trail_grid = np.array(trail_jax)

        # Evolve colors occasionally
        if random.random() < self.color_mutation_rate:
            species_idx = random.randint(0, NUM_SPECIES - 1)
            # More dramatic color mutations
            self.species_colors[species_idx] += np.random.normal(0, 10, 3)
            self.species_colors[species_idx] = np.clip(self.species_colors[species_idx], 50, 255)

        # Audio: generate tones based on species density
        if AUDIO_ENABLED:
            for i in range(NUM_SPECIES):
                density = np.mean(self.trail_grid[:, :, i])
                frequency = 220 + density * 440  # Base freq + density modulation
                volume = min(density * 0.1, 1.0)

                # Simple tone generation (placeholder - would need actual audio synthesis)
                # For now, just set channel volume
                if hasattr(self.audio_channels[i], 'set_volume'):
                    self.audio_channels[i].set_volume(volume)

    def draw(self):
        # Clear screen
        self.screen.fill((10, 10, 20))

        # Draw trails with amplified intensity
        trail_rgb = np.zeros((TRAIL_SIZE, TRAIL_SIZE, 3), dtype=np.uint8)
        for i in range(NUM_SPECIES):
            # Amplify trail intensity for visibility
            intensity = np.clip(self.trail_grid[:, :, i] * 1000, 0, 255).astype(np.uint8)
            color = self.species_colors[i].astype(np.uint8)
            trail_rgb[:, :, 0] = np.maximum(trail_rgb[:, :, 0], (intensity * color[0] // 255).astype(np.uint8))
            trail_rgb[:, :, 1] = np.maximum(trail_rgb[:, :, 1], (intensity * color[1] // 255).astype(np.uint8))
            trail_rgb[:, :, 2] = np.maximum(trail_rgb[:, :, 2], (intensity * color[2] // 255).astype(np.uint8))

        # Scale up to screen size
        trail_surface = pygame.surfarray.make_surface(trail_rgb)
        scaled_trail = pygame.transform.scale(trail_surface, (WIDTH, HEIGHT))
        self.screen.blit(scaled_trail, (0, 0))

        # Draw particles
        for i in range(NUM_PARTICLES):
            x, y = self.positions[i]
            species_idx = self.species[i]
            color = tuple(self.species_colors[species_idx].astype(int))
            pygame.draw.circle(self.screen, color, (int(x), int(y)), 2)

        # UI
        font = pygame.font.SysFont('monospace', 16)
        texts = [
            f"Force: {self.force_strength:.1f}",
            f"Friction: {self.friction:.3f}",
            f"Max Speed: {self.max_speed:.1f}",
            f"Sense Dist: {self.sense_distance:.1f}",
            f"Deposit: {self.deposit_rate:.3f}",
            "[Q/W] Force  [E/R] Friction  [T/Y] Speed  [U/I] Sense  [O/P] Deposit  [Space] Reset"
        ]

        for i, text in enumerate(texts):
            color = (200, 200, 200) if i < 5 else (150, 150, 150)
            surface = font.render(text, True, color)
            self.screen.blit(surface, (10, 10 + i * 20))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Reset simulation
                    self.positions = np.random.uniform(0, WIDTH, (NUM_PARTICLES, 2)).astype(np.float32)
                    self.velocities = np.random.uniform(-50, 50, (NUM_PARTICLES, 2)).astype(np.float32)
                    self.trail_grid.fill(0)
                elif event.key == pygame.K_q:
                    self.force_strength = max(50, self.force_strength - 50)
                elif event.key == pygame.K_w:
                    self.force_strength = min(500, self.force_strength + 50)
                elif event.key == pygame.K_e:
                    self.friction = max(0.8, self.friction - 0.05)
                elif event.key == pygame.K_r:
                    self.friction = min(0.99, self.friction + 0.05)
                elif event.key == pygame.K_t:
                    self.max_speed = max(20, self.max_speed - 20)
                elif event.key == pygame.K_y:
                    self.max_speed = min(200, self.max_speed + 20)
                elif event.key == pygame.K_u:
                    self.sense_distance = max(20, self.sense_distance - 10)
                elif event.key == pygame.K_i:
                    self.sense_distance = min(150, self.sense_distance + 10)
                elif event.key == pygame.K_o:
                    self.deposit_rate = max(0.01, self.deposit_rate - 0.05)
                elif event.key == pygame.K_p:
                    self.deposit_rate = min(0.5, self.deposit_rate + 0.05)

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()

if __name__ == "__main__":
    sim = SymbioticSwarmSymphony()
    sim.run()