import os
# Enable MPS fallback for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pygame
import sys

# Gemini flash code

# --- Configuration ---
WIDTH, HEIGHT = 1280, 720
NUM_PARTICLES = 131072  # 2^17
DENSITY_RES = 128       # Resolution for the neural field's "eyes"
FPS = 60

# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU (Warning: Performance will be limited)")

# --- Neural Components ---

class NeuralAura(nn.Module):
    """
    The 'Brain' of the simulation. 
    Maps (x, y) + density_context -> (force_x, force_y)
    """
    def __init__(self):
        super().__init__()
        # Positional encoding projection (Fourier features)
        # We use multiples of pi to favor periodic patterns for the torus space
        freq_init = torch.randint(1, 8, (2, 32)).float() * math.pi
        self.register_buffer("freqs", freq_init)
        
        self.net = nn.Sequential(
            nn.Linear(64 + 1, 128), # 64 fourier + 1 density context
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 2)
        )
        
        # Initialize with small weights to prevent chaos immediately
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)

    def forward(self, pos, density_val):
        # pos: [N, 2] in range [-1, 1]
        # density_val: [N, 1]
        
        # Fourier positional encoding
        ph = pos @ self.freqs
        pos_enc = torch.cat([torch.sin(ph), torch.cos(ph)], dim=-1) # [N, 64]
        
        # Combine with density context
        x = torch.cat([pos_enc, density_val], dim=-1)
        
        force = self.net(x)
        return force

# --- Simulation Engine ---

class Engine:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF)
        pygame.display.set_caption("AURA: Neural Particle Field")
        self.clock = pygame.time.Clock()
        
        # State tensors
        self.pos = (torch.rand(NUM_PARTICLES, 2, device=device) * 2 - 1) # [-1, 1]
        self.vel = torch.zeros(NUM_PARTICLES, 2, device=device)
        
        # Colors: HSL-like but in RGB tensors
        # Random initial colors
        self.colors = torch.rand(NUM_PARTICLES, 3, device=device)
        
        # Neural Field
        self.model = NeuralAura().to(device)
        self.model.eval()

        # Density Map (used for self-organization logic)
        self.density_map = torch.zeros(DENSITY_RES, DENSITY_RES, device=device)

        self.running = True
        self.paused = False
        self.mutation_rate = 0.05

        # Perform initial mutation to avoid boring flat field
        self.mutate()
        print("Initial neural field mutation applied")
        
        # Interaction
        self.mouse_pos = torch.zeros(1, 2, device=device)
        self.mouse_strength = 0.0

    def mutate(self):
        """Slightly perturb the neural field to evolve patterns"""
        with torch.no_grad():
            for param in self.model.parameters():
                param.add_(torch.randn_like(param) * self.mutation_rate)

    def update(self):
        if self.paused: return

        with torch.no_grad():
            # 1. Compute Density Context
            grid_pos = ((self.pos + 1) * 0.5 * (DENSITY_RES - 1)).long()
            grid_pos = grid_pos.clamp(0, DENSITY_RES - 1)
            
            self.density_map.zero_()
            self.density_map.index_put_(
                (grid_pos[:, 1], grid_pos[:, 0]), 
                torch.ones(NUM_PARTICLES, device=device), 
                accumulate=True
            )
            
            # Density Gradient for 'Pressure' (Repulse from high density)
            # A simple 3x3 box blur/gradient would be nice, but even simpler:
            # Just use the local density value to push away from center of mass-ish.
            density_norm = torch.log1p(self.density_map) / 5.0
            sampled_density = density_norm[grid_pos[:, 1], grid_pos[:, 0]].unsqueeze(-1)
            
            # 2. Compute Neural Forces
            forces = self.model(self.pos, torch.sin(sampled_density * 4.0)) # Make density response oscillating/wild
            
            # --- NEW: Neural Drift (Passive Evolution) ---
            if not self.paused:
                for param in self.model.parameters():
                    param.data.add_(torch.randn_like(param) * 0.0002)
            
            # 3. Physics & Interaction
            dist_to_mouse = self.pos - self.mouse_pos
            r2 = torch.sum(dist_to_mouse**2, dim=-1, keepdim=True) + 0.01
            mouse_force = (dist_to_mouse / r2) * self.mouse_strength * -0.02
            
            # --- NEW: Density Pressure (Anti-Clustering) ---
            # Push particles slightly if they are in a super-dense spot
            pressure_force = torch.randn_like(self.pos) * (sampled_density > 0.7).float() * 0.01
            
            self.vel *= 0.94 
            self.vel += forces * 0.006
            self.vel += mouse_force
            self.vel += pressure_force
            self.vel += torch.randn_like(self.vel) * 0.0015 # More jitter
            
            # Position update
            self.pos += self.vel
            
            # 5. Boundary Conditions: Torus Wrap-Around
            # Map [-1, 1] to [0, 2], wrap with modulo, then map back to [-1, 1]
            self.pos = (self.pos + 1) % 2 - 1

            # 6. Color evolution based on velocity/density
            speed = torch.norm(self.vel, dim=-1, keepdim=True)
            self.colors[:, 0] = (self.colors[:, 0] * 0.99 + (speed.squeeze() * 5.0).clamp(0, 1) * 0.01) # Red shifts with speed
            self.colors[:, 1] = (self.colors[:, 1] * 0.99 + (sampled_density.squeeze() * 0.5).clamp(0, 1) * 0.01) # Green shifts with density
            self.colors[:, 2] = (1.0 - self.colors[:, 0] - self.colors[:, 1]).clamp(0.2, 1.0)

    def draw(self):
        # 1. GPU-Accelerated Splatting
        with torch.no_grad():
            # Map positions to screen-space grid indices
            # Using a slightly lower resolution for the 'detailed' map to keep it fast, 
            # but high enough to look like particles.
            DET_RES_W, DET_RES_H = WIDTH // 2, HEIGHT // 2
            
            grid_pos = ((self.pos + 1) * 0.5 * torch.tensor([DET_RES_W-1, DET_RES_H-1], device=device)).long()
            grid_pos[:, 0] = grid_pos[:, 0].clamp(0, DET_RES_W - 1)
            grid_pos[:, 1] = grid_pos[:, 1].clamp(0, DET_RES_H - 1)
            
            # Create a 3-channel 'image' on GPU
            # Channel 0: Red, Channel 1: Green, Channel 2: Blue
            # We'll weigh them by particle colors
            img_gpu = torch.zeros(3, DET_RES_H, DET_RES_W, device=device)
            
            indices = grid_pos[:, 1] * DET_RES_W + grid_pos[:, 0]
            
            # Scatter colors into the channels
            for c in range(3):
                img_gpu[c].view(-1).index_add_(0, indices, self.colors[:, c])
            
            # 2. Dynamic Post-Processing on GPU
            # Log-scale for better dynamic range (visualizing clusters and individuals)
            img_gpu = torch.log1p(img_gpu * 2.0)
            img_gpu = img_gpu / (img_gpu.max() + 1e-6)
            
            # Convert to CPU uint8
            img_cpu = (img_gpu * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            # Dimensions are [3, H, W] -> [W, H, 3] for Pygame
            img_final = img_cpu.transpose(2, 1, 0)

        # 3. Blit to Screen
        # Use a temporary surface for the upscaled GPU result
        full_img_surf = pygame.Surface((DET_RES_W, DET_RES_H))
        pygame.surfarray.blit_array(full_img_surf, img_final)
        
        # Scale to window size
        scaled_surf = pygame.transform.smoothscale(full_img_surf, (WIDTH, HEIGHT))
        self.screen.blit(scaled_surf, (0, 0))
        
        # 4. Info Layer
        font = pygame.font.SysFont("Arial", 18, bold=True)
        fps = self.clock.get_fps()
        info = f"Aura | Particles: {NUM_PARTICLES} | Device: {device} | FPS: {fps:.1f}"
        controls = "SPACE: Mutate NN | R: Reset | MOUSE: Influence | Q: Quit"
        
        # Draw background for text readability
        pygame.draw.rect(self.screen, (0, 0, 0, 150), (5, 5, 450, 55))
        
        ts1 = font.render(info, True, (255, 255, 255))
        ts2 = font.render(controls, True, (180, 180, 220))
        self.screen.blit(ts1, (15, 12))
        self.screen.blit(ts2, (15, 34))
        
        pygame.display.flip()

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.running = False
                    if event.key == pygame.K_SPACE:
                        self.mutate()
                        print("Neural field mutated!")
                    if event.key == pygame.K_r:
                        self.pos = (torch.rand(NUM_PARTICLES, 2, device=device) * 2 - 1)
                        self.vel.zero_()
                        print("Reset!")
                    if event.key == pygame.K_p:
                        self.paused = not self.paused

            # Interaction
            m_pos = pygame.mouse.get_pos()
            self.mouse_pos[0, 0] = (m_pos[0] / WIDTH) * 2 - 1
            self.mouse_pos[0, 1] = (m_pos[1] / HEIGHT) * 2 - 1
            
            m_buttons = pygame.mouse.get_pressed()
            if m_buttons[0]: self.mouse_strength = 1.0 # Attract
            elif m_buttons[2]: self.mouse_strength = -1.0 # Repel
            else: self.mouse_strength = 0.0

            self.update()
            self.draw()
            self.clock.tick(FPS)

if __name__ == "__main__":
    # Prevent multi-threading issues with torch/pygame if any
    try:
        engine = Engine()
        engine.run()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pygame.quit()
