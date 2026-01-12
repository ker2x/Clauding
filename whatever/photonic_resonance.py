import pygame
import numpy as np
import math

WIDTH, HEIGHT = 1400, 900
GRID_RES = 1024  # High resolution for fine detailing
FPS = 60
DT = 0.002  # Smaller timestep for stability

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF)
pygame.display.set_caption("Wave Billiard - 2D Wave Equation")
clock = pygame.time.Clock()

# ============================================================================
# WAVE SIMULATOR (2D Wave Equation with boundary)
# ============================================================================

class WaveBilliard:
    """2D wave equation in a domain with boundaries"""
    def __init__(self, width=GRID_RES, height=GRID_RES):
        self.w, self.h = width, height
        self.u = np.zeros((height, width), dtype=np.float32)  # Wave amplitude
        self.u_prev = np.zeros((height, width), dtype=np.float32)
        self.boundary = np.ones((height, width), dtype=np.float32)  # 1 = open, 0 = wall

        # Domain: circular by default
        self._init_circular_domain()

        self.time = 0

    def _init_circular_domain(self, center_x=0.5, center_y=0.5, radius=0.4):
        """Create circular billiard"""
        cy, cx = np.ogrid[:self.h, :self.w]
        cy = cy / self.h - center_y
        cx = cx / self.w - center_x
        dist = np.sqrt(cx**2 + cy**2)
        self.boundary = (dist < radius).astype(np.float32)

    def _init_elliptical_domain(self, cx=0.5, cy=0.5, a=0.45, b=0.3):
        """Create elliptical billiard"""
        y, x = np.ogrid[:self.h, :self.w]
        y = y / self.h - cy
        x = x / self.w - cx
        dist = (x**2 / (a**2) + y**2 / (b**2))
        self.boundary = (dist < 1.0).astype(np.float32)

    def _init_stadium_domain(self):
        """Create stadium (rectangle with semicircles)"""
        y, x = np.ogrid[:self.h, :self.w]
        y = y / self.h - 0.5
        x = x / self.w - 0.5

        # Rectangle part
        rect = (np.abs(x) < 0.3) & (np.abs(y) < 0.4)

        # Semicircles
        left = (x < -0.3) & ((x + 0.3)**2 + y**2 < 0.4**2)
        right = (x > 0.3) & ((x - 0.3)**2 + y**2 < 0.4**2)

        self.boundary = (rect | left | right).astype(np.float32)

    def step(self):
        """Wave equation with stable propagation"""
        # Compute Laplacian
        u_padded = np.pad(self.u, 1, mode='constant', constant_values=0)
        lap = (u_padded[1:-1, 2:] + u_padded[1:-1, :-2] +
               u_padded[2:, 1:-1] + u_padded[:-2, 1:-1] - 4*u_padded[1:-1, 1:-1])

        # Stable wave speed (CFL condition: must be < 1/sqrt(2) ≈ 0.707)
        c_squared_dt_sq = 0.3  # Stable, visible propagation
        u_new = (2*self.u - self.u_prev + c_squared_dt_sq * lap)

        # Apply boundary: zero Dirichlet BC
        u_new[self.boundary == 0] = 0

        # Gentle damping to prevent blow-up
        # REMOVED DAMPING as requested. 
        # u_new *= 0.999

        self.u_prev = self.u.copy()
        self.u = u_new
        self.time += DT

    def emit_source(self, grid_x, grid_y, amplitude=2.0, frequency=8):
        """Add oscillating point source with hard driver"""
        gx = int(grid_x * self.w)
        gy = int(grid_y * self.h)

        if 0 <= gx < self.w and 0 <= gy < self.h and self.boundary[gy, gx] > 0:
            # Hard driver: Overwrite wave value to prevent infinite build-up
            # Use a tiny radius (point source) to allow natural propagation
            # A large forced area fights the wave equation
            r = 3 # influence radius
            x0, x1 = max(0, gx-r), min(self.w, gx+r)
            y0, y1 = max(0, gy-r), min(self.h, gy+r)
            
            y, x = np.ogrid[y0:y1, x0:x1]
            dist_sq = (x - gx)**2 + (y - gy)**2
            
            # Sinusoidal driver (faster frequency for better ripples)
            # Use 20 instead of 8 to get more visible periodic waves
            freq_eff = 20
            source = amplitude * np.exp(-dist_sq / (2.0**2)) * math.sin(freq_eff * self.time)
            
            # Use a mask to only overwrite near the center
            mask = np.exp(-dist_sq / (2.0**2))
            self.u[y0:y1, x0:x1] = self.u[y0:y1, x0:x1] * (1 - mask) + source * mask

    def render(self):
        """Convert wave to RGB"""
        # Use absolute value
        u_abs = np.abs(self.u)

        # Log-scaling was okay, but let's try extreme gamma compression for visibility
        # Gamma < 1 boosts dark values significantly
        
        # Magma/Inferno Palette (Nils Berglund Style)
        # Visualization of Wave Energy/Amplitude Envelope
        
        # 1. Compute Intensity (Amplitude)
        # Use log scale to compress dynamic range and boost lows
        # This makes weak interference patterns visible as orange/purple instead of black
        intensity = np.log1p(u_abs * 5.0) / np.log1p(10.0)
        val = np.clip(intensity, 0, 1.0)
        
        # 3. Create the Gradient (approximating Magma)
        # Shifted thresholds to be more colorful
        #   0.0 - 0.2: Deep Purple
        #   0.2 - 0.5: Purple -> Red/Orange
        #   0.5 - 0.8: Orange -> Yellow
        #   0.8 - 1.0: Yellow -> White
        
        # Vectorized Piecewise Interpolation
        r = np.zeros_like(val)
        g = np.zeros_like(val)
        b = np.zeros_like(val)
        
        # Segment 1: Deep Purple (0.0 - 0.2)
        mask1 = (val < 0.2)
        if np.any(mask1):
            t = val[mask1] / 0.2
            r[mask1] = 0.2 * t
            g[mask1] = 0.0
            b[mask1] = 0.4 * t
            
        # Segment 2: Purple to Orange (0.2 - 0.5)
        mask2 = (val >= 0.2) & (val < 0.5)
        if np.any(mask2):
            t = (val[mask2] - 0.2) / 0.3
            r[mask2] = 0.2 + 0.8 * t
            g[mask2] = 0.4 * t # Up to orange/red
            b[mask2] = 0.4 - 0.4 * t # Fade out blue
            
        # Segment 3: Orange to Yellow (0.5 - 0.8)
        mask3 = (val >= 0.5) & (val < 0.8)
        if np.any(mask3):
            t = (val[mask3] - 0.5) / 0.3
            r[mask3] = 1.0
            g[mask3] = 0.4 + 0.6 * t
            b[mask3] = 0.0 + 0.2 * t
            
        # Segment 4: Yellow to White (0.8 - 1.0)
        mask4 = (val >= 0.8)
        if np.any(mask4):
            t = (val[mask4] - 0.8) / 0.2
            r[mask4] = 1.0
            g[mask4] = 1.0
            b[mask4] = 0.2 + 0.8 * t

        # Apply boundary mask
        r *= self.boundary
        g *= self.boundary
        b *= self.boundary

        # Add boundary walls (dark grey)
        r[self.boundary == 0] = 0.1
        g[self.boundary == 0] = 0.1
        b[self.boundary == 0] = 0.1

        rgb = np.stack([r, g, b], axis=-1)
        return np.clip(rgb, 0, 1)

# ============================================================================
# MAIN
# ============================================================================

def main():
    billiard = WaveBilliard(GRID_RES, GRID_RES)

    # Choose domain
    domain_mode = 0  # 0=circular, 1=elliptical, 2=stadium
    billiard._init_circular_domain()

    source_pos = None # REMOVED
    sources = [] # List of (x, y) tuples
    running = True
    paused = False
    source_enabled = True
    frame_counter = 0

    # Cached render
    rendered = None

    print("Controls:")
    print("CLICK: Place light source | L: Toggle light ON/OFF | SPACE: Pause | C/E/S: Domains | X: Clear | Q: Quit")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_c:
                    billiard._init_circular_domain()
                    billiard.u = np.zeros((GRID_RES, GRID_RES), dtype=np.float32)
                    billiard.u_prev = np.zeros((GRID_RES, GRID_RES), dtype=np.float32)
                    print("Switched to circular domain")
                if event.key == pygame.K_e:
                    billiard._init_elliptical_domain()
                    billiard.u = np.zeros((GRID_RES, GRID_RES), dtype=np.float32)
                    billiard.u_prev = np.zeros((GRID_RES, GRID_RES), dtype=np.float32)
                    print("Switched to elliptical domain")
                if event.key == pygame.K_s:
                    billiard._init_stadium_domain()
                    billiard.u = np.zeros((GRID_RES, GRID_RES), dtype=np.float32)
                    billiard.u_prev = np.zeros((GRID_RES, GRID_RES), dtype=np.float32)
                    print("Switched to stadium domain")
                if event.key == pygame.K_x:
                    billiard.u = np.zeros((GRID_RES, GRID_RES), dtype=np.float32)
                    billiard.u_prev = np.zeros((GRID_RES, GRID_RES), dtype=np.float32)
                    sources = []
                    print("Cleared all waves and sources")
                if event.key == pygame.K_l:
                    source_enabled = not source_enabled
                    print(f"Light source: {'ON' if source_enabled else 'OFF'}")

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                # Add new source position
                new_source = (mx / WIDTH, my / HEIGHT)
                sources.append(new_source)
                print(f"Added source at ({new_source[0]:.2f}, {new_source[1]:.2f}) - Total: {len(sources)}")

        if not paused:
            # Update wave
            billiard.step()

            # Add sources if enabled
            if source_enabled:
                for sx, sy in sources:
                     billiard.emit_source(sx, sy, amplitude=1.5, frequency=8)

                # Debug every 60 frames
                frame_counter += 1
                if frame_counter % 60 == 0:
                    print(f"Frame {frame_counter}: source_enabled={source_enabled}")

        # Render every frame
        rendered = billiard.render()

        # Display
        screen.fill((5, 5, 10))

        if rendered is not None:
            # Convert to pygame surface
            rgb_uint8 = (rendered * 255).astype(np.uint8)
            rgb_transposed = np.transpose(rgb_uint8, (1, 0, 2))

            surf = pygame.surfarray.make_surface(rgb_transposed)
            surf_scaled = pygame.transform.scale(surf, (WIDTH, HEIGHT))
            screen.blit(surf_scaled, (0, 0))

        # UI
        font = pygame.font.SysFont("Arial", 14, bold=True)
        info = [
            f"Wave Billiard | Time: {billiard.time:.2f}s | Light: {'ON' if source_enabled else 'OFF'} | {'PAUSED' if paused else 'RUNNING'}",
            "CLICK: Place source | L: Toggle | C/E/S: Domains | SPACE: Pause | X: Clear | Q: Quit"
        ]
        for i, text in enumerate(info):
            txt_surf = font.render(text, True, (100, 255, 100))
            screen.blit(txt_surf, (10, 10 + i * 25))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
