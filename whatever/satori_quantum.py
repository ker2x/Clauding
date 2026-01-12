import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import time
import math
import torch
import torch.nn.functional as F
import torch.fft
import numpy as np
import pygame

# --- Configuration ---
WIDTH, HEIGHT = 1024, 768
GRID_RES = 128  # Even lower res for MAX stability
NUM_PARTICLES = 16384
FPS = 60

# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

class SatoriQuantum:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF)
        pygame.display.set_caption("SATORI: Quantum Pilot-Wave (v4 High-Stability)")
        self.clock = pygame.time.Clock()
        
        # Grid constants
        # Using a very small K scale to prevent aliasing
        kx = torch.fft.fftfreq(GRID_RES, device=device) * 2.0 * math.pi
        KX, KY = torch.meshgrid(kx, kx, indexing='ij')
        self.K2 = KX**2 + KY**2
        self.dt = 0.001

        self.reset_wave()
        self.running = True

    def reset_wave(self):
        # Wave Function
        self.psi = torch.zeros((GRID_RES, GRID_RES), dtype=torch.complex64, device=device)
        self.potential = torch.zeros((GRID_RES, GRID_RES), dtype=torch.float32, device=device)
        
        y, x = torch.meshgrid(torch.linspace(-1, 1, GRID_RES, device=device), 
                              torch.linspace(-1, 1, GRID_RES, device=device), indexing='ij')
        
        # Stable Wave Packet
        gauss = torch.exp(-(x**2 + (y+0.6)**2) / 0.15)
        phase = y * 10.0
        self.psi = torch.complex(gauss * torch.cos(phase), gauss * torch.sin(phase))
        
        # Initial particles
        self.pos = (torch.rand(NUM_PARTICLES, 2, device=device) * 0.4) + torch.tensor([0.0, -0.6], device=device)
        self.vel = torch.zeros(NUM_PARTICLES, 2, device=device)

    def update_wave(self):
        # 1. Potential Step
        p_phase = self.potential * self.dt * 1000.0
        self.psi *= torch.complex(torch.cos(-p_phase), torch.sin(-p_phase))
        
        # 2. Kinetic Step (Free propagation)
        psi_k = torch.fft.fft2(self.psi)
        k_phase = self.K2 * self.dt * 20.0 # Reduced scaling
        psi_k *= torch.complex(torch.cos(-k_phase), torch.sin(-k_phase))
        self.psi = torch.fft.ifft2(psi_k)
        
        # 3. NO MASK - Let it reflect/wrap naturally
        
        # 4. Mandatory Normalization
        absq = self.psi.real**2 + self.psi.imag**2
        norm = torch.sqrt(torch.sum(absq))
        if norm > 1e-10:
            self.psi /= (norm + 1e-10)

    def update_particles(self):
        with torch.no_grad():
            # Guide particles by wave phase gradient
            # Interpolate psi at particle positions
            coords = self.pos.view(1, 1, -1, 2)
            psi_ri = torch.stack([self.psi.real, self.psi.imag], dim=1).unsqueeze(0)
            s_psi = F.grid_sample(psi_ri, coords, align_corners=True, mode='bilinear').squeeze()
            
            p_r, p_i = s_psi[0], s_psi[1]
            rho = p_r**2 + p_i**2 + 1e-6
            
            # Simple Central Difference Gradients
            gx_r = (torch.roll(self.psi.real, -1, 1) - torch.roll(self.psi.real, 1, 1)) * 10.0
            gx_i = (torch.roll(self.psi.imag, -1, 1) - torch.roll(self.psi.imag, 1, 1)) * 10.0
            gy_r = (torch.roll(self.psi.real, -1, 0) - torch.roll(self.psi.real, 1, 0)) * 10.0
            gy_i = (torch.roll(self.psi.imag, -1, 0) - torch.roll(self.psi.imag, 1, 0)) * 10.0
            
            g_ri = torch.stack([gx_r, gx_i, gy_r, gy_i], dim=1).unsqueeze(0)
            s_grad = F.grid_sample(g_ri, coords, align_corners=True, mode='bilinear').squeeze()
            
            # Bohmian Formula: v = Im(grad_psi / psi)
            vx = (s_grad[1] * p_r - s_grad[0] * p_i) / rho
            vy = (s_grad[3] * p_r - s_grad[2] * p_i) / rho
            
            # Smooth velocity
            target_vel = torch.stack([vx, vy], dim=-1) * 0.02
            self.vel = (self.vel * 0.8) + target_vel.clamp(-0.1, 0.1)
            
            self.pos += self.vel
            self.pos = (self.pos + 1) % 2 - 1

    def draw(self):
        with torch.no_grad():
            mag = torch.sqrt(self.psi.real**2 + self.psi.imag**2)
            phase = torch.atan2(self.psi.imag, self.psi.real)
            
            # HIGH GAIN RENDERING
            # Make sure even small waves are visible
            mag_vis = (mag / (mag.mean() + 1e-8)) * 0.5
            mag_vis = torch.tanh(mag_vis)

            r = (torch.cos(phase) * 0.5 + 0.5) * mag_vis
            g = (torch.cos(phase + 2.0) * 0.5 + 0.5) * mag_vis
            b = (torch.cos(phase + 4.0) * 0.5 + 0.5) * mag_vis
            
            field_rgb = torch.stack([r, g, b], dim=-1)
            # Add Potential (Yellow/White)
            field_rgb[..., 0:2] += self.potential.unsqueeze(-1) * 0.5
            field_rgb = field_rgb.clamp(0, 1)
            
            field_cpu = (field_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
            surf = pygame.surfarray.make_surface(field_cpu.swapaxes(0, 1))
            self.screen.blit(pygame.transform.scale(surf, (WIDTH, HEIGHT)), (0, 0))
            
            # Particles
            pos_np = self.pos.detach().cpu().numpy()
            screen_x = ((pos_np[:, 0] + 1) * 0.5 * WIDTH).astype(np.int32)
            screen_y = ((pos_np[:, 1] + 1) * 0.5 * HEIGHT).astype(np.int32)
            
            valid = (screen_x > 0) & (screen_x < WIDTH-1) & (screen_y > 0) & (screen_y < HEIGHT-1)
            # Use pixel array for fast particle rendering
            px = pygame.PixelArray(self.screen)
            for i in range(len(screen_x)):
                if valid[i]:
                    px[screen_x[i], screen_y[i]] = (255, 255, 255)
            del px

        # UI
        font = pygame.font.SysFont("Arial", 16, bold=True)
        pygame.draw.rect(self.screen, (0, 0, 0, 180), (10, HEIGHT-40, 600, 30))
        self.screen.blit(font.render(f"Satori v4 | MOUSE: Draw Barriers | SPACE: Reset | FPS: {self.clock.get_fps():.0f}", True, (255,255,255)), (20, HEIGHT - 35))
        
        pygame.display.flip()

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: self.reset_wave()
        
        # Direct Coordinate Drawing
        if pygame.mouse.get_pressed()[0]:
            mx, my = pygame.mouse.get_pos()
            gx, gy = int((mx/WIDTH)*GRID_RES), int((my/HEIGHT)*GRID_RES)
            if 0 <= gx < GRID_RES and 0 <= gy < GRID_RES:
                # Stamp a permanent barrier
                for i in range(-3, 4):
                    for j in range(-3, 4):
                        if 0 <= gx+i < GRID_RES and 0 <= gy+j < GRID_RES:
                            self.potential[gy+j, gx+i] = 1.0

    def run(self):
        while self.running:
            self.handle_input()
            self.update_wave()
            self.update_particles()
            self.draw()
            self.clock.tick(FPS)

if __name__ == "__main__":
    sim = SatoriQuantum()
    sim.run()
    pygame.quit()
