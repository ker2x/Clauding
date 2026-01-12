import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import time
import math
import torch
import torch.nn.functional as F
import numpy as np
import pygame
import random

# --- Configuration ---
WIDTH, HEIGHT = 1200, 900
NUM_RAYS = 1024 # Optimized ray count
MAX_BOUNCES = 2
POP_SIZE = 12
FPS = 60

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

class LumenCreature:
    def __init__(self, x, y, dna=None):
        self.pos = torch.tensor([float(x), float(y)], device=device)
        self.vel = torch.zeros(2, device=device)
        self.angle = random.uniform(0, 2*math.pi)
        self.radius = 20.0
        self.energy = 50.0
        self.alive = True
        
        # DNA: Neural weights for the brain (3 sensors -> 2 motors)
        if dna is None:
            self.dna = torch.randn(8, device=device) * 1.5 
        else:
            self.dna = dna
        
        self.sensor_vals = [0.0, 0.0, 0.0]

    def brain(self, sensors):
        s = torch.tensor(sensors, device=device).float()
        w = self.dna[0:6].view(2, 3)
        b = self.dna[6:8]
        return torch.tanh(torch.mv(w, s) + b)

    def mutate(self):
        new_dna = self.dna + torch.randn_like(self.dna) * 0.15
        return LumenCreature(self.pos[0].item(), self.pos[1].item(), new_dna)

class LumenWorld:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF | pygame.HWACCEL)
        pygame.display.set_caption("LUMEN: Photonic Evolution (Optimized)")
        self.clock = pygame.time.Clock()
        
        self.lights = [
            torch.tensor([WIDTH*0.2, HEIGHT*0.2], device=device),
            torch.tensor([WIDTH*0.8, HEIGHT*0.2], device=device),
            torch.tensor([WIDTH*0.5, HEIGHT*0.8], device=device)
        ]
        
        self.lenses = torch.tensor([
            [WIDTH*0.5, HEIGHT*0.5, 90.0, 1.6],
            [WIDTH*0.25, HEIGHT*0.65, 65.0, 1.4],
            [WIDTH*0.75, HEIGHT*0.65, 65.0, 1.4]
        ], device=device).float()
        
        self.creatures = [LumenCreature(random.uniform(50, WIDTH-50), random.uniform(50, HEIGHT-50)) for _ in range(POP_SIZE)]
        self.running = True
        self.selected_light = None

    def intersect_circles_vect(self, P, D, circles):
        # P: [N, 2], D: [N, 2], circles: [M, 4]
        # Fully vectorized intersection: O(N * M) but one CUDA kernel
        N, M = P.shape[0], circles.shape[0]
        
        # Expand for broad-phase broadcast
        P_exp = P.unsqueeze(1) # [N, 1, 2]
        D_exp = D.unsqueeze(1) # [N, 1, 2]
        C = circles[:, :2].unsqueeze(0) # [1, M, 2]
        R = circles[:, 2].unsqueeze(0)  # [1, M]
        
        PC = P_exp - C # [N, M, 2]
        b = torch.sum(D_exp * PC, dim=-1) # [N, M]
        c = torch.sum(PC * PC, dim=-1) - R**2 # [N, M]
        
        delta = b**2 - c
        mask = delta > 0
        
        sqrt_delta = torch.sqrt(torch.clamp(delta, min=0))
        t1 = -b - sqrt_delta
        t2 = -b + sqrt_delta
        
        t = torch.where(t1 > 0.1, t1, t2)
        valid = mask & (t > 0.1)
        
        # We need the min T across circles for each ray
        # Fill non-valid with large distance
        t_valid = torch.where(valid, t, torch.tensor(2000.0, device=device))
        
        best_t, best_idx = torch.min(t_valid, dim=1)
        
        # If best_t is still 2000, we hit nothing
        hit_mask = best_t < 1999.0
        best_idx = torch.where(hit_mask, best_idx, torch.tensor(-1, device=device))
        
        return best_t, best_idx

    def cast_light_rays(self, origin, all_circles):
        angles = torch.linspace(0, 2*math.pi, NUM_RAYS, device=device)
        D = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        P = origin.view(1, 2).expand(NUM_RAYS, 2)
        
        segments = []
        c_hits = torch.zeros(NUM_RAYS, device=device, dtype=torch.long) - 1
        
        curr_P, curr_D = P, D
        for bounce in range(MAX_BOUNCES + 1):
            t, idx = self.intersect_circles_vect(curr_P, curr_D, all_circles)
            hit_P = curr_P + curr_D * t.unsqueeze(-1)
            
            # Record segments for drawing (subsample rays for drawing speed)
            mask_draw = (torch.arange(NUM_RAYS, device=device) % 6 == 0)
            segments.append((curr_P[mask_draw].detach().cpu().numpy(), hit_P[mask_draw].detach().cpu().numpy()))
            
            if bounce == 0:
                is_c = (idx >= 0) & (idx < len(self.creatures))
                c_hits = torch.where(is_c, idx, c_hits)
            
            is_lens = (idx >= len(self.creatures))
            if is_lens.any() and bounce < MAX_BOUNCES:
                mask = is_lens
                C_lens = all_circles[idx[mask], :2]
                ior = all_circles[idx[mask], 3].unsqueeze(-1)
                to_center = C_lens - hit_P[mask]
                to_center /= (torch.norm(to_center, dim=-1, keepdim=True) + 1e-6)
                
                curr_P = hit_P
                curr_D = curr_D.clone()
                curr_D[mask] = (curr_D[mask] * 0.6 + to_center * (ior - 1.0)).clamp(-1, 1)
                curr_D[mask] /= (torch.norm(curr_D[mask], dim=-1, keepdim=True) + 1e-6)
            else: break
                
        return segments, c_hits

    def handle_collisions_vect(self):
        if not self.creatures: return
        
        c_pos = torch.stack([c.pos for c in self.creatures]) # [N, 2]
        c_rad = torch.tensor([c.radius for c in self.creatures], device=device)
        
        # 1. Creature vs Creature
        dist_mat = torch.cdist(c_pos, c_pos) # [N, N]
        rad_mat = c_rad.unsqueeze(0) + c_rad.unsqueeze(1) # [N, N]
        
        overlap = rad_mat - dist_mat
        mask = (overlap > 0) & (torch.eye(len(self.creatures), device=device) == 0)
        
        if mask.any():
            diff = c_pos.unsqueeze(0) - c_pos.unsqueeze(1) # [N, N, 2]
            push_dir = diff / (dist_mat.unsqueeze(-1) + 1e-6)
            push_force = push_dir * (overlap.unsqueeze(-1) * 0.4)
            # Accumulate pushes
            total_push = (push_force * mask.unsqueeze(-1)).sum(dim=1)
            # Apply to creature objects
            for i, c in enumerate(self.creatures):
                c.pos += total_push[i]

        # 2. Creature vs Lens
        for lens in self.lenses:
            lp = lens[:2]
            lr = lens[2]
            diff = c_pos - lp
            dist = torch.norm(diff, dim=-1)
            min_dist = c_rad + lr
            mask_l = dist < min_dist
            if mask_l.any():
                overlap_l = min_dist - dist
                push_l = (diff / dist.unsqueeze(-1)) * overlap_l.unsqueeze(-1)
                # Apply only to colliding
                for i in range(len(self.creatures)):
                    if mask_l[i]:
                        self.creatures[i].pos += push_l[i]
                        self.creatures[i].vel *= 0.5

    def update(self):
        # Raycasting
        creature_nodes = torch.stack([torch.cat([c.pos, torch.tensor([c.radius, 0.0], device=device)]) for c in self.creatures])
        all_circles = torch.cat([creature_nodes, self.lenses], dim=0)

        all_segments = []
        total_feeding_gain = torch.zeros(len(self.creatures), device=device)
        
        for lp in self.lights:
            segs, c_hits = self.cast_light_rays(lp, all_circles)
            all_segments.append(segs)
            valid_mask = c_hits >= 0
            if valid_mask.any():
                total_feeding_gain.index_add_(0, c_hits[valid_mask], torch.ones_like(c_hits[valid_mask]).float() * 0.06)
            
        # Neural/Biology
        new_creatures = []
        c_pos_tensor = torch.stack([c.pos for c in self.creatures])
        dist_mat = torch.cdist(c_pos_tensor, c_pos_tensor)
        crowding = (dist_mat < 100.0).float().sum(dim=1) - 1.0

        for i, c in enumerate(self.creatures):
            # Combined Light Sensors
            eye_vals = []
            for sa in [-0.6, 0, 0.6]:
                spos = c.pos + torch.tensor([math.cos(c.angle + sa), math.sin(c.angle + sa)], device=device) * c.radius * 1.5
                l_val = 0.0
                for lp in self.lights:
                    l_val += 1500.0 / (torch.norm(lp - spos) + 1.0)
                if i < len(total_feeding_gain) and total_feeding_gain[i] > 1:
                    l_val *= 1.3
                eye_vals.append(l_val * 0.01)
            
            c.sensor_vals = eye_vals
            brain_out = c.brain(eye_vals)
            thrust = (brain_out[0].item() + 1.2) * 2.0
            turn = brain_out[1].item() * 0.18
            
            c.angle += turn
            c.energy -= 0.22 # Basal
            c.energy -= abs(thrust) * 0.06 # Move
            c.energy -= crowding[i].item() * 0.12 # Crowd
            c.energy += total_feeding_gain[i].item() # Eat
            c.energy = min(100, c.energy)
            
            if c.energy <= 0:
                c.alive = False
                continue
            
            c.vel = torch.tensor([math.cos(c.angle), math.sin(c.angle)], device=device) * thrust
            c.pos += c.vel
            c.pos[0] = torch.clamp(c.pos[0], 0, WIDTH)
            c.pos[1] = torch.clamp(c.pos[1], 0, HEIGHT)
            
            if c.energy > 95:
                c.energy = 40 
                if len(self.creatures) + len(new_creatures) < POP_SIZE * 5:
                    new_creatures.append(c.mutate())
        
        self.handle_collisions_vect()
        
        self.creatures = [c for c in self.creatures if c.alive]
        self.creatures.extend(new_creatures)
        if len(self.creatures) < 5:
            self.creatures.append(LumenCreature(random.uniform(0, WIDTH), random.uniform(0, HEIGHT)))
            
        return all_segments

    def draw(self, all_segments):
        self.screen.fill((2, 2, 8))
        
        # 1. Lenses
        for lens in self.lenses.cpu().numpy():
            lx, ly, lr, lior = lens
            pygame.draw.circle(self.screen, (15, 30, 45), (int(lx), int(ly)), int(lr))
            pygame.draw.circle(self.screen, (80, 160, 255), (int(lx), int(ly)), int(lr), 2)
            
        # 2. Light (Optimized draw pass)
        bloom = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for seg_list in all_segments:
            for p1, p2 in seg_list:
                for i in range(len(p1)):
                    pygame.draw.line(bloom, (255, 255, 180, 12), (int(p1[i][0]), int(p1[i][1])), (int(p2[i][0]), int(p2[i][1])), 1)
        self.screen.blit(bloom, (0, 0), special_flags=pygame.BLEND_ADD)
        
        # 3. Creatures
        for c in self.creatures:
            cx, cy = int(c.pos[0]), int(c.pos[1])
            col_g = int(min(255, 100 + c.energy * 1.5))
            pygame.draw.circle(self.screen, (10, 20, 40), (cx, cy), int(c.radius))
            pygame.draw.circle(self.screen, (0, col_g, 255), (cx, cy), int(c.radius), 2)
            # Eyes
            for idx, sa in enumerate([-0.6, 0, 0.6]):
                ex, ey = cx + math.cos(c.angle + sa) * c.radius * 0.8, cy + math.sin(c.angle + sa) * c.radius * 0.8
                bright = int(min(255, 50 + c.sensor_vals[idx] * 200.0))
                pygame.draw.circle(self.screen, (bright, bright, 200), (int(ex), int(ey)), 4)
            core_col = (255, int(150 + c.energy), 100) if c.energy > 80 else (0, 150, 255)
            pygame.draw.circle(self.screen, core_col, (cx, cy), int(c.radius * 0.3))

        # 4. Light Sources
        for lp in self.lights:
            lx, ly = int(lp[0]), int(lp[1])
            pygame.draw.circle(self.screen, (255, 255, 255), (lx, ly), 10)
        
        # UI
        title_font = pygame.font.SysFont("Arial", 26, bold=True)
        self.screen.blit(title_font.render("LUMEN: Photonic Evolution", True, (255, 255, 255)), (30, 20))
        stats_font = pygame.font.SysFont("Courier", 18)
        self.screen.blit(stats_font.render(f"Population: {len(self.creatures)} | FPS: {int(self.clock.get_fps())}", True, (200, 220, 255)), (30, 55))
        
        pygame.display.flip()

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    m_pos = torch.tensor(event.pos, device=device, dtype=torch.float32)
                    for i, lp in enumerate(self.lights):
                        if torch.norm(lp - m_pos) < 25: self.selected_light = i
                if event.type == pygame.MOUSEBUTTONUP: self.selected_light = None
                if event.type == pygame.MOUSEMOTION and self.selected_light is not None:
                    self.lights[self.selected_light] = torch.tensor(event.pos, device=device, dtype=torch.float32)

            all_segments = self.update()
            self.draw(all_segments)
            self.clock.tick(FPS)

if __name__ == "__main__":
    sim = LumenWorld()
    sim.run()
    pygame.quit()
