import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import time
import math
import torch
import numpy as np
import pygame
import random

# --- Configuration ---
WIDTH, HEIGHT = 1200, 600
POP_SIZE = 40
SIM_TIME = 7.0 # Faster epochs
FPS = 60
SUBSTEPS = 5 

# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

class XenobotsSim:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF)
        pygame.display.set_caption("XENOBOTS: Soft-Body Evolution (Optimized)")
        self.clock = pygame.time.Clock()
        
        # Physics Params: More energetic
        self.gravity = 1000.0  # Heavier
        self.ground_y = 550
        self.nodes_per_bot = 6
        self.links_per_bot = 15
        
        self.init_population()
        self.epoch = 0
        self.timer = 0
        self.running = True
        self.best_dist = 0.0
        self.fast_mode = False

    def init_population(self):
        self.pos = torch.zeros((POP_SIZE, self.nodes_per_bot, 2), device=device)
        self.vel = torch.zeros((POP_SIZE, self.nodes_per_bot, 2), device=device)
        self.dna = torch.randn((POP_SIZE, self.links_per_bot, 3), device=device)
        
        adj = []
        for i in range(self.nodes_per_bot):
            for j in range(i + 1, self.nodes_per_bot):
                adj.append([i, j])
        self.adj = torch.tensor(adj[:self.links_per_bot], device=device).long()
        
        self.base_lengths = torch.ones((POP_SIZE, self.links_per_bot), device=device) * 50.0
        self.reset_positions()

    def reset_positions(self):
        for p in range(POP_SIZE):
            cx, cy = 100 + random.uniform(-10, 10), self.ground_y - 100
            for n in range(self.nodes_per_bot):
                a = (n / self.nodes_per_bot) * 2 * math.pi
                self.pos[p, n, 0] = cx + math.cos(a) * 35
                self.pos[p, n, 1] = cy + math.sin(a) * 35
        self.vel.zero_()
        self.timer = 0

    def physics_step(self, dt):
        dt_sub = dt / SUBSTEPS
        
        for _ in range(SUBSTEPS):
            # 1. Faster Muscle Contraction
            t = self.timer
            # Increased frequency range for more excitement
            freq = 6.0 + torch.abs(self.dna[:, :, 0] * 8.0)
            phase = self.dna[:, :, 1] * math.pi
            amp = torch.sigmoid(self.dna[:, :, 2]) * 0.7
            
            target_lengths = self.base_lengths * (1.0 + amp * torch.sin(freq * t + phase))
            
            # 2. Stronger Springs
            p1 = self.pos[:, self.adj[:, 0]]
            p2 = self.pos[:, self.adj[:, 1]]
            diff = p1 - p2
            dist = torch.norm(diff, dim=-1, keepdim=True) + 1e-6
            unit = diff / dist
            
            k = 600.0 # Stiffer
            force_mag = k * (dist - target_lengths.unsqueeze(-1))
            force_vec = unit * force_mag
            
            node_forces = torch.zeros_like(self.pos)
            node_forces.index_add_(1, self.adj[:, 0], -force_vec)
            node_forces.index_add_(1, self.adj[:, 1], force_vec)
            
            # 3. Forces
            node_forces[:, :, 1] += self.gravity 
            
            # 4. Air & Ground Friction
            self.vel *= 0.985 # Less damping -> more bounce
            self.vel += node_forces * dt_sub
            self.pos += self.vel * dt_sub
            
            mask = self.pos[:, :, 1] > self.ground_y
            if mask.any():
                self.pos[mask, 1] = self.ground_y
                self.vel[mask, 0] *= 0.5 # High grip
                self.vel[mask, 1] = torch.clamp(self.vel[mask, 1], max=0)
            
        self.timer += dt

    def evolve(self):
        fitness = self.pos[:, :, 0].max(dim=1).values
        indices = torch.argsort(fitness, descending=True)
        self.best_dist = float(fitness[indices[0]])
        
        # Elite Selection (Top 6)
        elites = self.dna[indices[:6]]
        
        new_dna = torch.zeros_like(self.dna)
        new_dna[:6] = elites
        for i in range(6, POP_SIZE):
            parent = elites[random.randint(0, 5)]
            # Adaptive mutation
            m_scale = 0.2 if self.epoch < 10 else 0.1
            new_dna[i] = parent + torch.randn_like(parent) * m_scale
            
        self.dna = new_dna
        self.epoch += 1
        self.reset_positions()

    def draw(self):
        self.screen.fill((5, 5, 10))
        # Ground
        pygame.draw.rect(self.screen, (25, 25, 40), (0, self.ground_y, WIDTH, HEIGHT-self.ground_y))
        
        pos_np = self.pos.detach().cpu().numpy()
        fitness_np = pos_np[:, :, 0].max(axis=1)
        best_idx = np.argmax(fitness_np)
        
        # Camera
        target_cam = max(0, pos_np[best_idx, :, 0].mean() - WIDTH // 4)
        if not hasattr(self, 'camera_x'): self.camera_x = 0
        self.camera_x = self.camera_x * 0.9 + target_cam * 0.1
        
        # Optimized Draw Loop
        adj_np = self.adj.cpu().numpy()
        for p in range(POP_SIZE):
            p_pos = pos_np[p]
            is_best = (p == best_idx)
            
            # Viewport culling
            if (p_pos[:, 0].max() - self.camera_x) < -50 or (p_pos[:, 0].min() - self.camera_x) > WIDTH + 50:
                continue
            
            # Draw Skin
            color = (80, 255, 120, 180) if is_best else (100, 100, 180, 60)
            pts = [(int(pt[0] - self.camera_x), int(pt[1])) for pt in p_pos]
            if is_best:
                pygame.draw.polygon(self.screen, color, pts)
            else:
                pygame.draw.polygon(self.screen, color, pts, 1)

            # Draw Muscles
            for l in range(self.links_per_bot):
                i1, i2 = adj_np[l]
                start = (int(p_pos[i1, 0] - self.camera_x), int(p_pos[i1, 1]))
                end = (int(p_pos[i2, 0] - self.camera_x), int(p_pos[i2, 1]))
                
                m_color = (150, 255, 200) if is_best else (80, 80, 120)
                width = 3 if is_best else 1
                pygame.draw.line(self.screen, m_color, start, end, width)

        # UI
        f = pygame.font.SysFont("Arial", 22, bold=True)
        ui_txt = f"EP: {self.epoch} | TIME: {self.timer:.1f}s | BEST: {self.best_dist - 100:.1f} | {'TURBO' if self.fast_mode else 'NORMAL'}"
        self.screen.blit(f.render(ui_txt, True, (255, 255, 255)), (20, 20))
        pygame.display.flip()

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q: self.running = False
                    if event.key == pygame.K_SPACE: self.fast_mode = not self.fast_mode
                    if event.key == pygame.K_r: 
                        self.init_population()
                        self.epoch = 0

            # Evolution cycle
            if self.timer > SIM_TIME:
                self.evolve()

            # Physics
            iters = 5 if self.fast_mode else 1
            for _ in range(iters):
                self.physics_step(1/60)

            if not self.fast_mode or (pygame.time.get_ticks() % 10 == 0):
                self.draw()
                
            if not self.fast_mode:
                self.clock.tick(FPS)

if __name__ == "__main__":
    sim = XenobotsSim()
    sim.run()
    pygame.quit()
