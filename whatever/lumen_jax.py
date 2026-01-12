import os
import time
import math
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
import numpy as np
import pygame
import random

# --- Configuration ---
WIDTH, HEIGHT = 1200, 900
NUM_RAYS = 512 
MAX_BOUNCES = 4 # Increased to allow rays to pass through multiple creatures
MAX_POP = 100
START_POP = 15
NUM_LIGHTS = 3
FPS = 60

# --- JAX Physics Kernels ---

@jit
def intersect_circles_jax(P, D, circles):
    PC = P[:, None, :] - circles[None, :, :2]
    b = 2.0 * jnp.sum(D[:, None, :] * PC, axis=-1)
    c = jnp.sum(PC * PC, axis=-1) - circles[None, :, 2]**2
    delta = b**2 - 4.0 * c
    
    t1 = (-b - jnp.sqrt(jnp.clip(delta, 0))) / 2.0
    t2 = (-b + jnp.sqrt(jnp.clip(delta, 0))) / 2.0
    t = jnp.where(t1 > 0.1, t1, t2)
    hit = (delta > 0) & (t > 0.1)
    
    t_valid = jnp.where(hit, t, 2000.0)
    best_idx = jnp.argmin(t_valid, axis=1)
    best_t = jnp.min(t_valid, axis=1)
    best_idx = jnp.where(best_t < 1999.0, best_idx, -1)
    return best_t, best_idx

@jit
def cast_light_rays_jax(origin, all_circles):
    """ Cast rays that pass through creatures but refract on lenses """
    angles = jnp.linspace(0, 2*math.pi, NUM_RAYS)
    D = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
    P = origin[None, :].repeat(NUM_RAYS, axis=0)
    
    def bounce_step(carry, _):
        curr_P, curr_D, c_hits_acc = carry
        
        t, idx = intersect_circles_jax(curr_P, curr_D, all_circles)
        hit_P = curr_P + curr_D * t[:, None]
        
        # Check if hit is a creature
        is_creature = (idx >= 0) & (idx < MAX_POP)
        # Record hit (we use a mask to update a multi-hit buffer)
        # Note: In a fixed JIT loop, multi-hit is hard. 
        # We'll just pass the hit indices for THIS bounce back.
        
        is_lens = (idx >= MAX_POP)
        safe_idx = jnp.maximum(idx, 0)
        C_lens = all_circles[safe_idx, :2]
        
        # Refraction for lenses
        to_center = C_lens - hit_P
        to_center /= (jnp.linalg.norm(to_center, axis=1, keepdims=True) + 1e-6)
        refracted_D = (curr_D * 0.6 + to_center * 0.6)
        refracted_D /= (jnp.linalg.norm(refracted_D, axis=1, keepdims=True) + 1e-6)
        
        # Logic: If it's a creature, continue STRAIGHT. If it's a lens, REFRACT.
        # If it's nothing, stay put.
        next_D = jnp.where(is_lens[:, None], refracted_D, curr_D)
        # Push forward slightly to avoid double-hitting the same boundary
        next_P = jnp.where(idx[:, None] >= 0, hit_P + next_D * 0.5, curr_P)
        
        # Capture current hit indices for metabolism
        return (next_P, next_D, c_hits_acc), (curr_P, hit_P, idx)

    initial_carry = (P, D, jnp.full(NUM_RAYS, -1))
    _, all_steps = lax.scan(bounce_step, initial_carry, jnp.arange(MAX_BOUNCES))
    
    # all_steps format: (starts, ends, hit_indices)
    return all_steps

@jit
def step_physics_jax(pos, vel, angle, energy, dna, alive, lights, lenses, key, time_f):
    # 1. Dynamic Lights (Oscillation)
    osc_lights = lights.at[:, 0].add(jnp.sin(time_f * 0.05) * 40.0)
    osc_lights = osc_lights.at[:, 1].add(jnp.cos(time_f * 0.05) * 40.0)

    # 2. Optics & Photonic Metabolism (Translucent)
    creature_nodes = jnp.concatenate([pos, jnp.full((MAX_POP, 1), 20.0)], axis=-1)
    creature_nodes = jnp.where(alive[:, None], creature_nodes, jnp.array([0., 0., -100.0]))
    all_circles = jnp.concatenate([creature_nodes, lenses], axis=0)
    
    # ray_data: [starts, ends, idxs] -> [BOUNCES, 3, NUM_RAYS, 2 or 1]
    all_ray_data = vmap(lambda lp: cast_light_rays_jax(lp, all_circles))(osc_lights)
    
    # Extract hit indices: [NUM_LIGHTS, BOUNCES, NUM_RAYS]
    light_hit_indices = all_ray_data[2]
    
    def sum_hits(c_idx):
        # Count hits across all lights AND all bounces (penetration)
        hits = jnp.sum(light_hit_indices == c_idx)
        return hits * 0.05 # Lowered gain slightly because multiple hits possible
    
    feeding_gain = vmap(sum_hits)(jnp.arange(MAX_POP))
    
    # 3. Sensors
    def get_sensors(p, a):
        sens_angles = jnp.array([-0.6, 0.0, 0.6])
        def get_one(sa):
            spos = p + jnp.array([jnp.cos(a + sa), jnp.sin(a + sa)]) * 30.0
            l_val = jnp.sum(1500.0 / (jnp.linalg.norm(osc_lights - spos, axis=1) + 1.0))
            return l_val * 0.01
        return vmap(get_one)(sens_angles)

    all_sensors = vmap(get_sensors)(pos, angle)
    
    # 4. Brain & Movement
    w = dna[:, 0:6].reshape(MAX_POP, 2, 3)
    b = dna[:, 6:8]
    brain_out = jnp.tanh(jnp.squeeze(jnp.matmul(w, all_sensors[:, :, None])) + b)
    thrust = (brain_out[:, 0] + 1.2) * 2.1
    turn = brain_out[:, 1] * 0.22
    
    # Biology
    new_angle = angle + turn
    dist_mat = jnp.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    crowding = jnp.sum(dist_mat < 100.0, axis=1) - 1.0
    
    new_energy = energy - 0.22 - (jnp.abs(thrust) * 0.06) - (crowding * 0.12) + feeding_gain
    new_energy = jnp.clip(new_energy, 0, 100)
    is_alive = alive & (new_energy > 0)
    
    # Movement & Boundaries
    new_vel = jnp.stack([jnp.cos(new_angle), jnp.sin(new_angle)], axis=-1) * thrust[:, None]
    new_pos = pos + new_vel * is_alive[:, None]
    new_pos = jnp.clip(new_pos, jnp.array([0., 0.]), jnp.array([WIDTH, HEIGHT]))
    
    # 5. Collision (Lens, Light, Creature)
    def resolve_collisions(p_in):
        # Lens
        diff_lens = p_in[:, None, :] - lenses[None, :, :2]
        dist_lens = jnp.linalg.norm(diff_lens, axis=-1)
        overlap_lens = jnp.clip(20.0 + lenses[:, 2] - dist_lens, 0)
        push_lens = (diff_lens / (dist_lens[:, :, None] + 1e-6)) * overlap_lens[:, :, None]
        p_p = p_in + jnp.sum(push_lens, axis=1)
        # Light
        diff_l = p_p[:, None, :] - osc_lights[None, :, :2]
        dist_l = jnp.linalg.norm(diff_l, axis=-1)
        overlap_l = jnp.clip(35.0 - dist_l, 0)
        p_p += jnp.sum((diff_l / (dist_l[:, :, None] + 1e-6)) * overlap_l[:, :, None], axis=1)
        # Creature
        diff_c = p_p[:, None, :] - p_p[None, :, :]
        dist_c = jnp.linalg.norm(diff_c, axis=-1)
        mask_c = (dist_c < 40.0) & (jnp.eye(MAX_POP) == 0) & (alive[:, None] & alive[None, :])
        push_c = (diff_c / (dist_c[:, :, None] + 1e-6)) * (jnp.clip(40.0 - dist_c, 0)[:, :, None] * 0.5)
        return p_p + jnp.sum(push_c * mask_c[:, :, None], axis=1)
    
    new_pos = resolve_collisions(new_pos)
    
    return new_pos, new_vel, new_angle, new_energy, is_alive, key, all_sensors, all_ray_data, osc_lights

class LumenJAX:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF | pygame.HWACCEL)
        pygame.display.set_caption("LUMEN: Photonic Evolution (JAX v1.4 Translucent)")
        self.clock = pygame.time.Clock()
        
        self.pos = jnp.array(np.random.uniform(100, WIDTH-100, (MAX_POP, 2)))
        self.vel = jnp.zeros((MAX_POP, 2))
        self.angle = jnp.array(np.random.uniform(0, 2*math.pi, (MAX_POP,)))
        self.energy = jnp.full((MAX_POP,), 50.0)
        self.dna = jnp.array(np.random.randn(MAX_POP, 8))
        self.alive = jnp.concatenate([jnp.ones(START_POP), jnp.zeros(MAX_POP - START_POP)]).astype(bool)
        self.sensor_vals = jnp.zeros((MAX_POP, 3))
        
        self.lights = jnp.array([[WIDTH*0.25, HEIGHT*0.25], [WIDTH*0.75, HEIGHT*0.25], [WIDTH*0.5, HEIGHT*0.75]])
        self.lenses = jnp.array([[WIDTH*0.5, HEIGHT*0.5, 90.0], [WIDTH*0.2, HEIGHT*0.7, 70.0], [WIDTH*0.8, HEIGHT*0.7, 70.0]])
        
        self.key = jax.random.PRNGKey(int(time.time()))
        self.frame_count = 0.0
        self.running = True

    def update(self):
        self.frame_count += 1.0
        self.pos, self.vel, self.angle, self.energy, self.alive, self.key, self.sensor_vals, self.ray_data, self.cur_lights = step_physics_jax(
            self.pos, self.vel, self.angle, self.energy, self.dna, self.alive,
            self.lights, self.lenses, self.key, self.frame_count
        )
        
        # Reproduction (Python-side)
        alive_ptr = np.array(self.alive)
        energy_ptr = np.array(self.energy)
        dna_ptr = np.array(self.dna)
        pos_ptr = np.array(self.pos)
        angle_ptr = np.array(self.angle)
        
        for i in np.where(alive_ptr)[0]:
            if energy_ptr[i] > 95:
                dead_slots = np.where(~alive_ptr)[0]
                if len(dead_slots) > 0:
                    slot = dead_slots[0]; alive_ptr[slot] = True
                    dna_ptr[slot] = dna_ptr[i] + np.random.randn(8) * 0.15
                    pos_ptr[slot] = pos_ptr[i]; angle_ptr[slot] = angle_ptr[i]
                    energy_ptr[slot] = 40.0; energy_ptr[i] = 40.0
        
        if np.sum(alive_ptr) < 5:
            slot = np.where(~alive_ptr)[0][0]
            alive_ptr[slot] = True; energy_ptr[slot] = 50.0
            pos_ptr[slot] = [random.uniform(0, WIDTH), random.uniform(0, HEIGHT)]
            
        self.alive = jnp.array(alive_ptr); self.energy = jnp.array(energy_ptr)
        self.dna = jnp.array(dna_ptr); self.pos = jnp.array(pos_ptr); self.angle = jnp.array(angle_ptr)

    def draw(self):
        self.screen.fill((2, 2, 8))
        bloom = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        # ray_data: (starts, ends, idxs)
        all_starts, all_ends, _ = self.ray_data
        starts_np = np.array(all_starts); ends_np = np.array(all_ends)
        
        for l in range(NUM_LIGHTS):
            for b in range(MAX_BOUNCES):
                for r in range(0, NUM_RAYS, 7):
                    p1, p2 = starts_np[l, b, r], ends_np[l, b, r]
                    if np.linalg.norm(p1 - p2) > 0.1:
                        pygame.draw.line(bloom, (255, 255, 180, 8), (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 1)
        self.screen.blit(bloom, (0, 0), special_flags=pygame.BLEND_ADD)

        for lens in self.lenses:
            pygame.draw.circle(self.screen, (15, 30, 45), (int(lens[0]), int(lens[1])), int(lens[2]))
            pygame.draw.circle(self.screen, (80, 160, 255), (int(lens[0]), int(lens[1])), int(lens[2]), 2)
        for lp in self.cur_lights:
            pygame.draw.circle(self.screen, (255, 255, 255), (int(lp[0]), int(lp[1])), 10)
            
        pos_np = np.array(self.pos); angle_np = np.array(self.angle)
        alive_np = np.array(self.alive); energy_np = np.array(self.energy); sens_np = np.array(self.sensor_vals)
        
        for i in range(MAX_POP):
            if alive_np[i]:
                cx, cy = int(pos_np[i, 0]), int(pos_np[i, 1])
                col_g = int(min(255, 50 + energy_np[i] * 2.0))
                pygame.draw.circle(self.screen, (10, 20, 50), (cx, cy), 20)
                pygame.draw.circle(self.screen, (0, col_g, 255), (cx, cy), 20, 2)
                for idx, sa in enumerate([-0.6, 0, 0.6]):
                    ex, ey = cx + math.cos(angle_np[i] + sa) * 16, cy + math.sin(angle_np[i] + sa) * 16
                    bright = int(min(255, 50 + sens_np[i, idx] * 200.0))
                    pygame.draw.circle(self.screen, (bright, bright, 200), (int(ex), int(ey)), 4)
                core_col = (255, int(150 + energy_np[i]), 100) if energy_np[i] > 80 else (0, 150, 255)
                pygame.draw.circle(self.screen, core_col, (cx, cy), 6)

        title_font = pygame.font.SysFont("Arial", 26, bold=True)
        self.screen.blit(title_font.render("LUMEN: JAX Translucent Edition", True, (255, 255, 255)), (30, 20))
        stats_font = pygame.font.SysFont("Courier", 18)
        self.screen.blit(stats_font.render(f"Population: {int(np.sum(alive_np))} | FPS: {int(self.clock.get_fps())}", True, (200, 220, 255)), (30, 55))
        pygame.display.flip()

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.running = False
            self.update(); self.draw(); self.clock.tick(FPS)

if __name__ == "__main__":
    print("Initializing JAX Simulation (Translucent XLA Warmup)...")
    sim = LumenJAX(); sim.run(); pygame.quit()
