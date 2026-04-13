#!/usr/bin/env python3
"""
Plasma Field Simulation - Python version
Charged particle simulation with electromagnetic fields.

Physics:
- Coulomb force: F = k * q1 * q2 / r^2
- Lorentz force: F = q(E + v × B)
- Thermal noise for realistic plasma behavior

Controls:
  Q - Quit
  R - Reset particles
  F - Cycle field mode (0=uniform, 1=pole, 2=vortex)
  Space - Pause/Resume
"""

import pygame
import numpy as np
import math
import random

# Configuration
WIDTH, HEIGHT = 1024, 768
NUM_PARTICLES = 20000  # Lower than Metal version for CPU performance
NUM_TYPES = 3  # negative, neutral, positive

# Physics
INTERACTION_RADIUS = 30.0
COULOMB_STRENGTH = 0.0003
MAGNETIC_STRENGTH = 0.0002
FRICTION = 0.98
THERMAL_NOISE = 0.03

# Colors (RGBA)
COLORS = [
    (25, 150, 255, 200),   # Negative: Cyan (electrons)
    (200, 200, 200, 100),  # Neutral: White
    (255, 100, 25, 200),    # Positive: Orange (ions)
]

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Plasma Field - Python")
    clock = pygame.time.Clock()
    
    # Initialize particles
    particles = init_particles()
    
    running = True
    paused = False
    field_mode = 0
    field_modes = ["Uniform E+B", "Central Pole", "Rotating Vortex"]
    time = 0
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    particles = init_particles()
                elif event.key == pygame.K_f:
                    field_mode = (field_mode + 1) % 3
                    print(f"Field Mode: {field_modes[field_mode]}")
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print("Paused" if paused else "Resumed")
        
        if not paused:
            time += 1
            particles = update_particles(particles, time, field_mode)
        
        # Render
        screen.fill((0, 0, 5))  # Dark blue background
        
        # Draw particles (as points)
        positions = particles['pos']
        types = particles['type']
        
        # Create surface for particles
        surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for i in range(NUM_PARTICLES):
            x, y = int(positions[i, 0]), int(positions[i, 1])
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                pygame.draw.circle(surf, COLORS[types[i]], (x, y), 1)
        
        screen.blit(surf, (0, 0))
        
        # FPS counter
        fps = clock.get_fps()
        font = pygame.font.Font(None, 24)
        fps_text = font.render(f"FPS: {fps:.1f} | Mode: {field_modes[field_mode]}", True, (150, 150, 150))
        screen.blit(fps_text, (10, 10))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()


def init_particles():
    """Initialize particle positions, velocities, and charges."""
    # Positions: clustered near center with vortex initial velocity
    angles = np.random.uniform(0, 2 * math.pi, NUM_PARTICLES)
    radii = np.random.uniform(0, WIDTH * 0.4, NUM_PARTICLES)
    
    positions = np.column_stack([
        WIDTH / 2 + np.cos(angles) * radii,
        HEIGHT / 2 + np.sin(angles) * radii
    ])
    
    # Velocities: tangential for vortex effect
    speeds = np.random.uniform(0.5, 2.5, NUM_PARTICLES)
    velocities = np.column_stack([
        -np.sin(angles) * speeds,
        np.cos(angles) * speeds
    ])
    
    # Charges: 40% negative, 20% neutral, 40% positive
    r = np.random.random(NUM_PARTICLES)
    types = np.where(r < 0.4, 0, np.where(r < 0.6, 1, 2))
    charges = np.where(r < 0.4, -1.0, np.where(r < 0.6, 0.0, 1.0))
    
    return {'pos': positions, 'vel': velocities, 'type': types, 'charge': charges}


def update_particles(particles, time, field_mode):
    """Update particle physics."""
    pos = particles['pos']
    vel = particles['vel']
    types = particles['type']
    charges = particles['charge']
    
    forces = np.zeros((NUM_PARTICLES, 2))
    
    # External field forces
    if field_mode == 0:
        # Uniform E and B fields
        forces[:, 0] += charges * 0.0  # Ex
        forces[:, 1] += charges * 0.05  # Ey
        # Lorentz: v × B (B is out of screen)
        forces[:, 0] += vel[:, 1] * MAGNETIC_STRENGTH * 50
        forces[:, 1] -= vel[:, 0] * MAGNETIC_STRENGTH * 50
    
    elif field_mode == 1:
        # Central pole field
        center = np.array([WIDTH / 2, HEIGHT / 2])
        to_center = center - pos
        dists = np.linalg.norm(to_center, axis=1, keepdims=True) + 1.0
        radial = to_center / dists
        tangent = np.column_stack([-radial[:, 1], radial[:, 0]])
        
        strength = 1.0 / (dists ** 2 + 1.0) * 0.5
        forces += charges[:, np.newaxis] * radial * strength * 20
        forces += charges[:, np.newaxis] * tangent * strength * 10
    
    elif field_mode == 2:
        # Rotating vortex
        center = np.array([WIDTH / 2, HEIGHT / 2])
        offset = pos - center
        dists = np.linalg.norm(offset, axis=1, keepdims=True) + 1.0
        angles = np.arctan2(offset[:, 1], offset[:, 0])
        
        rot_angle = angles + time * 0.02
        field_dir = np.column_stack([np.cos(rot_angle), np.sin(rot_angle)])
        
        strength = 1.0 / (dists * 0.02 + 1.0) * 0.3
        forces += charges[:, np.newaxis] * field_dir * strength * 15
    
    # Particle-particle Coulomb interactions (simplified - sample subset)
    # Full O(n²) is too slow, so we use spatial sampling
    sample_size = min(500, NUM_PARTICLES)
    indices = np.random.choice(NUM_PARTICLES, sample_size, replace=False)
    
    for i in indices:
        for j in range(NUM_PARTICLES):
            if i == j:
                continue
            delta = pos[j] - pos[i]
            
            # Wrap around
            if delta[0] > WIDTH * 0.5:
                delta[0] -= WIDTH
            if delta[0] < -WIDTH * 0.5:
                delta[0] += WIDTH
            if delta[1] > HEIGHT * 0.5:
                delta[1] -= HEIGHT
            if delta[1] < -HEIGHT * 0.5:
                delta[1] += HEIGHT
            
            dist2 = delta[0]**2 + delta[1]**2
            
            if dist2 < INTERACTION_RADIUS**2 and dist2 > 0.1:
                dist = math.sqrt(dist2)
                direction = delta / dist
                coulomb = COULOMB_STRENGTH * charges[i] * charges[j] / (dist2 + 1.0)
                forces[i] += direction * coulomb * 50
    
    # Thermal noise
    noise = np.random.uniform(-THERMAL_NOISE, THERMAL_NOISE, (NUM_PARTICLES, 2))
    forces += noise
    
    # Update velocity and position
    vel += forces
    vel *= FRICTION
    
    # Clamp velocity
    speeds = np.linalg.norm(vel, axis=1, keepdims=True)
    speeds = np.maximum(speeds, 0.001)
    vel = np.where(speeds > 10, vel / speeds * 10, vel)
    
    pos += vel
    
    # Wrap position
    pos[:, 0] = pos[:, 0] % WIDTH
    pos[:, 1] = pos[:, 1] % HEIGHT
    
    return {'pos': pos, 'vel': vel, 'type': types, 'charge': charges}


if __name__ == "__main__":
    main()
