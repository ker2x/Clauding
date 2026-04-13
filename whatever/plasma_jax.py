#!/usr/bin/env python3
"""
Plasma Field Simulation - JAX/GPU version
GPU-accelerated charged particle simulation.

Physics:
- Coulomb force: F = k * q1 * q2 / r^2
- Lorentz force: F = q(E + v × B)
- Uses JAX for Metal GPU acceleration on Apple Silicon

Controls:
  Q - Quit
  R - Reset particles
  F - Cycle field mode
  Space - Pause/Resume

IMPORTANT: This version uses spatial sampling to avoid O(n²) memory explosion.
"""

import pygame
import jax
import jax.numpy as jnp
from jax import jit, lax
import math
import numpy as np

# Configuration - REDUCED from original to prevent memory crash
WIDTH, HEIGHT = 1024, 768
NUM_PARTICLES = 8000  # Was 50000 - too much memory for O(n²) approach
NUM_TYPES = 3

# Sample size for pairwise interactions (memory-safe)
SAMPLE_SIZE = 600  # Only compute forces with this many random particles

# Physics
COULOMB_STRENGTH = 0.015
MAGNETIC_STRENGTH = 0.008
FRICTION = 0.985
ELECTRIC_Y = 0.3

# Colors
COLORS = [
    (25, 150, 255, 200),   # Negative: Cyan
    (200, 200, 200, 100),  # Neutral: White
    (255, 100, 25, 200),   # Positive: Orange
]


def compute_external_field(charges, velocities, positions, time, field_mode):
    """Compute external electromagnetic field forces.
    
    Uses conditional masking instead of lax.switch for JAX JIT compatibility.
    field_mode: 0=uniform, 1=pole, 2=vortex
    """
    # Initialize forces
    forces = jnp.zeros((NUM_PARTICLES, 2))
    
    # Uniform E and B field (mode 0)
    uniform_forces = jnp.zeros((NUM_PARTICLES, 2))
    uniform_forces = uniform_forces.at[:, 1].add(charges * ELECTRIC_Y)
    uniform_forces = uniform_forces.at[:, 0].add(velocities[:, 1] * MAGNETIC_STRENGTH * 100)
    uniform_forces = uniform_forces.at[:, 1].add(-velocities[:, 0] * MAGNETIC_STRENGTH * 100)
    
    # Central pole field (mode 1)
    pole_forces = jnp.zeros((NUM_PARTICLES, 2))
    center = jnp.array([WIDTH / 2, HEIGHT / 2])
    to_center = center - positions
    dists = jnp.linalg.norm(to_center, axis=1, keepdims=True) + 1.0
    radial = to_center / dists
    tangent = jnp.concatenate([-radial[:, 1:2], radial[:, 0:1]], axis=1)
    strength = 1.0 / (dists ** 2 * 0.001 + 1.0) * 2.0
    pole_forces = pole_forces + charges[:, jnp.newaxis] * radial * strength * 20
    pole_forces = pole_forces + charges[:, jnp.newaxis] * tangent * strength * 10
    
    # Rotating vortex field (mode 2)
    vortex_forces = jnp.zeros((NUM_PARTICLES, 2))
    offset = positions - center
    dists2 = jnp.linalg.norm(offset, axis=1, keepdims=True) + 1.0
    angles = jnp.arctan2(offset[:, 1], offset[:, 0])
    rot_angle = angles + time * 0.02
    field_dir = jnp.stack([jnp.cos(rot_angle), jnp.sin(rot_angle)], axis=1)
    strength2 = 1.0 / (dists2 * 0.01 + 1.0) * 0.5
    vortex_forces = vortex_forces + charges[:, jnp.newaxis] * field_dir * strength2 * 15
    
    # Select based on field_mode using masking (JAX-safe)
    m0 = (field_mode == 0).astype(jnp.float32)
    m1 = (field_mode == 1).astype(jnp.float32)
    m2 = (field_mode == 2).astype(jnp.float32)
    
    forces = forces + uniform_forces * m0 + pole_forces * m1 + vortex_forces * m2
    return forces

@jit
def compute_coulomb_sampled(positions, charges, sample_indices):
    """Compute Coulomb forces using sampled particle subset (memory-safe O(n*m))."""
    # Get sampled positions and charges
    sampled_pos = positions[sample_indices]
    sampled_charge = charges[sample_indices]
    
    # Compute displacements: each particle to each sampled particle
    # Shape: (NUM_PARTICLES, SAMPLE_SIZE, 2)
    delta = positions[:, jnp.newaxis, :] - sampled_pos[jnp.newaxis, :, :]
    
    # Handle wrapping - fix shape mismatch by expanding the condition properly
    # delta has shape (N, M, 2), we need to add (N, M, 1) to it
    wrap_x = jnp.array([WIDTH, 0.0])
    wrap_y = jnp.array([0.0, HEIGHT])
    
    # Wrap delta based on condition
    cond_x_pos = delta[:, :, 0:1] > WIDTH * 0.5
    cond_x_neg = delta[:, :, 0:1] < -WIDTH * 0.5
    cond_y_pos = delta[:, :, 1:2] > HEIGHT * 0.5
    cond_y_neg = delta[:, :, 1:2] < -HEIGHT * 0.5
    
    delta = jnp.where(cond_x_pos, delta - wrap_x, delta)
    delta = jnp.where(cond_x_neg, delta + wrap_x, delta)
    delta = jnp.where(cond_y_pos, delta - wrap_y, delta)
    delta = jnp.where(cond_y_neg, delta + wrap_y, delta)
    
    # Squared distances
    dists2 = jnp.sum(delta ** 2, axis=2)
    
    # Mask for valid pairs (within radius, not too close)
    r_max = 40.0
    mask = (dists2 < r_max ** 2) & (dists2 > 0.1)
    
    # Coulomb force
    dists = jnp.sqrt(dists2 + 0.001)
    directions = delta / dists[:, :, jnp.newaxis]
    coulomb = COULOMB_STRENGTH * charges[:, jnp.newaxis] * sampled_charge[jnp.newaxis, :] / (dists2 + 1.0)
    coulomb = coulomb * mask.astype(float)
    
    # Scale up forces to compensate for sampling
    scale = float(NUM_PARTICLES) / float(SAMPLE_SIZE)
    
    # Sum forces over sampled particles
    force_sum = jnp.sum(directions * coulomb[:, :, jnp.newaxis], axis=1)
    return force_sum * 100.0 * scale

@jit
def update_plasma(positions, velocities, charges, time, field_mode, sample_indices):
    """JAX-accelerated plasma physics with memory-safe sampling."""
    
    # Compute external field forces
    external_forces = compute_external_field(charges, velocities, positions, time, field_mode)
    
    # Coulomb forces with sampled particles (memory-safe!)
    coulomb_forces = compute_coulomb_sampled(positions, charges, sample_indices)
    
    forces = external_forces + coulomb_forces
    
    # Thermal noise - use jnp.floor for JAX-safe conversion
    time_int = jnp.floor(time * 100).astype(jnp.int32)
    key = jax.random.PRNGKey(time_int)
    noise = jax.random.normal(key, (NUM_PARTICLES, 2)) * 0.05
    forces = forces + noise
    
    # Update velocity and position
    velocities = velocities + forces
    velocities = velocities * FRICTION
    
    # Clamp velocity
    speeds = jnp.linalg.norm(velocities, axis=1, keepdims=True)
    speeds = jnp.maximum(speeds, 0.001)
    velocities = jnp.where(speeds > 10, velocities / speeds * 10, velocities)
    
    positions = positions + velocities
    
    # Wrap position using jnp.mod
    positions = jnp.mod(positions + jnp.array([WIDTH, HEIGHT]), jnp.array([WIDTH, HEIGHT]))
    
    return positions, velocities


def init_particles():
    """Initialize particles."""
    key = jax.random.PRNGKey(42)
    
    angles = jax.random.uniform(key, (NUM_PARTICLES,), minval=0, maxval=2*math.pi)
    radii = jax.random.uniform(key, (NUM_PARTICLES,), maxval=WIDTH * 0.4)
    
    positions = jnp.column_stack([
        WIDTH / 2 + jnp.cos(angles) * radii,
        HEIGHT / 2 + jnp.sin(angles) * radii
    ])
    
    speeds = jax.random.uniform(key, (NUM_PARTICLES,), minval=0.5, maxval=2.5)
    velocities = jnp.column_stack([-jnp.sin(angles) * speeds, jnp.cos(angles) * speeds])
    
    r = jax.random.uniform(key, (NUM_PARTICLES,))
    types = jnp.where(r < 0.4, 0, jnp.where(r < 0.6, 1, 2))
    charges = jnp.where(r < 0.4, -1.0, jnp.where(r < 0.6, 0.0, 1.0))
    
    return positions, velocities, types, charges


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Plasma Field - JAX (Memory-Safe)")
    clock = pygame.time.Clock()
    
    print(f"Using device: {jax.default_backend()}")
    print(f"Particles: {NUM_PARTICLES}, Sample size: {SAMPLE_SIZE}")
    
    # Initialize on GPU
    positions, velocities, types, charges = init_particles()
    types_np = np.array(types)
    
    # Create sample indices for Coulomb interactions (stable across frames)
    sample_key = jax.random.PRNGKey(123)
    sample_indices = jax.random.choice(sample_key, NUM_PARTICLES, (SAMPLE_SIZE,), replace=False)
    
    running = True
    paused = False
    field_mode = 0
    field_modes = ["Uniform E+B", "Central Pole", "Rotating Vortex"]
    time = 0.0
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    positions, velocities, types, charges = init_particles()
                    types_np = np.array(types)
                elif event.key == pygame.K_f:
                    field_mode = (field_mode + 1) % 3
                    print(f"Field Mode: {field_modes[field_mode]}")
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print("Paused" if paused else "Resumed")
        
        if not paused:
            time += 0.016
            positions, velocities = update_plasma(
                positions, velocities, charges, time, field_mode, sample_indices
            )
        
        # Render (transfer from GPU to CPU)
        screen.fill((0, 0, 5))
        
        pos_np = np.array(positions)
        surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        
        for i in range(NUM_PARTICLES):
            x, y = int(pos_np[i, 0]), int(pos_np[i, 1])
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                pygame.draw.circle(surf, COLORS[types_np[i]], (x, y), 1)
        
        screen.blit(surf, (0, 0))
        
        # FPS
        fps = clock.get_fps()
        font = pygame.font.Font(None, 24)
        fps_text = font.render(f"FPS: {fps:.1f} | Mode: {field_modes[field_mode]}", True, (150, 150, 150))
        screen.blit(fps_text, (10, 10))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()


if __name__ == "__main__":
    main()
