import pygame
import jax
import jax.numpy as jnp
import numpy as np
import time

# Function to enable 64-bit precision if needed (usually 32-bit is faster and sufficient for visuals)
jax.config.update("jax_enable_x64", False)

WIDTH, HEIGHT = 1400, 800
GRID_RES = 2048
FPS = 60
DT = 0.004

# ============================================================================
# JAX KERNELS (JIT Compiled)
# ============================================================================

@jax.jit
def update_wave(u, u_prev, boundary, source_mask, source_val):
    """
    Update wave equation:
    u_new = 2*u - u_prev + c^2 * dt^2 * Laplacian(u)
    """
    # 1. Compute Laplacian with padding
    u_padded = jnp.pad(u, 1, mode='constant', constant_values=0)
    lap = (u_padded[1:-1, 2:] + u_padded[1:-1, :-2] +
           u_padded[2:, 1:-1] + u_padded[:-2, 1:-1] - 4 * u_padded[1:-1, 1:-1])

    c_squared_dt_sq = 0.3
    u_new = 2 * u - u_prev + c_squared_dt_sq * lap

    # 2. Apply Boundary (Dirichlet BC)
    u_new = u_new * boundary

    # 3. Apply Hard Source Driver
    # Where mask > 0, blend towards source_val.
    # u_new = u_new * (1 - mask) + source_val * mask
    u_new = u_new * (1.0 - source_mask) + source_val * source_mask

    # 4. No Damping (as requested)
    
    return u_new

@jax.jit
def render_magma(u, boundary):
    """
    Render wave amplitude using Magma/Inferno palette.
    """
    u_abs = jnp.abs(u)
    
    # 1. Compute Intensity (Log scale)
    intensity = jnp.log1p(u_abs * 5.0) / jnp.log1p(10.0)
    val = jnp.clip(intensity, 0, 1.0)
    
    # 2. Create Gradient (Magma-like)
    #   0.0 - 0.2: Deep Purple
    #   0.2 - 0.5: Purple -> Red/Orange
    #   0.5 - 0.8: Orange -> Yellow
    #   0.8 - 1.0: Yellow -> White
    
    # Use jnp.select/where for vectorized branching
    
    # Segment 1: Deep Purple (0.0 - 0.2)
    t1 = val / 0.2
    r1 = 0.2 * t1
    g1 = 0.0
    b1 = 0.4 * t1
    
    # Segment 2: Purple to Orange (0.2 - 0.5)
    t2 = (val - 0.2) / 0.3
    r2 = 0.2 + 0.8 * t2
    g2 = 0.4 * t2
    b2 = 0.4 - 0.4 * t2
    
    # Segment 3: Orange to Yellow (0.5 - 0.8)
    t3 = (val - 0.5) / 0.3
    r3 = 1.0
    g3 = 0.4 + 0.6 * t3
    b3 = 0.0 + 0.2 * t3
    
    # Segment 4: Yellow to White (0.8 - 1.0)
    t4 = (val - 0.8) / 0.2
    r4 = 1.0
    g4 = 1.0
    b4 = 0.2 + 0.8 * t4
    
    r = jnp.zeros_like(val)
    g = jnp.zeros_like(val)
    b = jnp.zeros_like(val)
    
    # Apply conditions in order (last true wins in select, or we can use nested where)
    # Using simple boolean masking multiplication for clarity since ranges are disjoint-ish
    # But jnp.select or np.piecewise logic is cleaner with jnp.where
    
    # We will use select for clean logic
    cond1 = val < 0.2
    cond2 = (val >= 0.2) & (val < 0.5)
    cond3 = (val >= 0.5) & (val < 0.8)
    cond4 = val >= 0.8
    
    r = jnp.zeros_like(val)
    r = jnp.where(cond1, r1, r)
    r = jnp.where(cond2, r2, r)
    r = jnp.where(cond3, r3, r)
    r = jnp.where(cond4, r4, r)
    
    g = jnp.zeros_like(val)
    g = jnp.where(cond1, g1, g)
    g = jnp.where(cond2, g2, g)
    g = jnp.where(cond3, g3, g)
    g = jnp.where(cond4, g4, g)
    
    b = jnp.zeros_like(val)
    b = jnp.where(cond1, b1, b)
    b = jnp.where(cond2, b2, b)
    b = jnp.where(cond3, b3, b)
    b = jnp.where(cond4, b4, b)

    # Apply boundary mask
    r *= boundary
    g *= boundary
    b *= boundary

    # Add boundary walls (dark grey)
    boundary_inv = 1.0 - boundary
    r += boundary_inv * 0.1
    g += boundary_inv * 0.1
    b += boundary_inv * 0.1

    rgb = jnp.stack([r, g, b], axis=-1)
    rgb = jnp.clip(rgb, 0, 1)
    return (rgb * 255).astype(jnp.uint8)

# ============================================================================
# MAIN SIMULATION CLASS
# ============================================================================

# ============================================================================
# MAIN SIMULATION CLASS
# ============================================================================

class WaveBilliardJAX:
    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.u = jnp.zeros((self.h, self.w), dtype=jnp.float32)
        self.u_prev = jnp.zeros((self.h, self.w), dtype=jnp.float32)
        self.boundary = jnp.ones((self.h, self.w), dtype=jnp.float32)
        self.grid_x, self.grid_y = jnp.meshgrid(jnp.arange(self.w), jnp.arange(self.h))
        self.time = 0.0
        
        # Sources state
        self.source_static_mask = jnp.zeros((self.h, self.w), dtype=jnp.float32)
        self.sources = [] # Check (x, y) tuples
        self.cached_mask_needs_update = False
        
        self.init_circular_domain()

    def init_circular_domain(self):
        cx, cy = self.w // 2, self.h // 2
        radius = min(cx, cy) - 20
        dist_sq = (self.grid_x - cx)**2 + (self.grid_y - cy)**2
        boundary_np = np.zeros((self.h, self.w), dtype=np.float32)
        boundary_np[dist_sq <= radius**2] = 1.0
        boundary_np[cy-30:cy+30, :cx] = 1.0 # Slit
        self.boundary = jnp.array(boundary_np)
        self.u = jnp.zeros_like(self.u) # Reset waves

    def init_elliptical_domain(self):
        cx, cy = self.w // 2, self.h // 2
        # Aspect corrected ellipse
        a = self.w // 2 - 20
        b = self.h // 2 - 20 # Max height usage
        dist_sq = ((self.grid_x - cx) / a)**2 + ((self.grid_y - cy) / b)**2
        boundary_np = np.zeros((self.h, self.w), dtype=np.float32)
        boundary_np[dist_sq <= 1.0] = 1.0
        self.boundary = jnp.array(boundary_np)
        self.u = jnp.zeros_like(self.u)

    def init_stadium_domain(self):
        cx, cy = self.w // 2, self.h // 2
        rect_w = self.w // 3
        radius = self.h // 3
        # Left circle
        dist_left = (self.grid_x - (cx - rect_w // 2))**2 + (self.grid_y - cy)**2
        # Right circle
        dist_right = (self.grid_x - (cx + rect_w // 2))**2 + (self.grid_y - cy)**2
        
        boundary_np = np.zeros((self.h, self.w), dtype=np.float32)
        
        # Circle caps
        boundary_np[dist_left <= radius**2] = 1.0
        boundary_np[dist_right <= radius**2] = 1.0
        # Rectangle body
        boundary_np[(self.grid_y >= cy - radius) & (self.grid_y <= cy + radius) & 
                    (self.grid_x >= cx - rect_w // 2) & (self.grid_x <= cx + rect_w // 2)] = 1.0
                    
        self.boundary = jnp.array(boundary_np)
        self.u = jnp.zeros_like(self.u)

        self.u = jnp.zeros_like(self.u)

    def init_rectangular_domain(self):
        cx, cy = self.w // 2, self.h // 2
        w, h = self.w // 2, self.h // 2
        mask = (self.grid_x >= cx - w//2) & (self.grid_x <= cx + w//2) & \
               (self.grid_y >= cy - h//2) & (self.grid_y <= cy + h//2)
        boundary_np = np.zeros((self.h, self.w), dtype=np.float32)
        boundary_np[mask] = 1.0
        self.boundary = jnp.array(boundary_np)
        self.u = jnp.zeros_like(self.u)

    def init_triangular_domain(self):
        # Equilateral Triangle
        h_tri = int(self.h * 0.8)
        side = int(h_tri * 2 / np.sqrt(3))
        # Center the bounding box vertically
        # Centroid is h/3 from bottom. Top is 2h/3 from centroid.
        # Bounding box mid is centroid - h/6.
        # So we want cy - h/6 = H/2  =>  cy = H/2 + h/6
        cx, cy = self.w // 2, self.h // 2 + h_tri // 6
        
        # Vertex 1 (Top)
        v1 = (cx, cy - 2 * h_tri // 3)
        # Vertex 2 (Bottom Left)
        v2 = (cx - side // 2, cy + h_tri // 3)
        # Vertex 3 (Bottom Right)
        v3 = (cx + side // 2, cy + h_tri // 3)
        
        # Barycentric coordinates or half-plane method
        # Line 1: v1 -> v2
        d1 = (v2[0] - v1[0]) * (self.grid_y - v1[1]) - (v2[1] - v1[1]) * (self.grid_x - v1[0])
        # Line 2: v2 -> v3
        d2 = (v3[0] - v2[0]) * (self.grid_y - v2[1]) - (v3[1] - v2[1]) * (self.grid_x - v2[0])
        # Line 3: v3 -> v1
        d3 = (v1[0] - v3[0]) * (self.grid_y - v3[1]) - (v1[1] - v3[1]) * (self.grid_x - v3[0])
        
        # Inside if all signs match.
        # With current vertex order (Top -> Left -> Right) and Y-down coords,
        # the cross products for "Inside" are all Negative (or all Positive if flipped).
        # We can just check the sign at the centroid to be sure, or rely on geometry.
        # My analysis says Negative for this winding.
        boundary_np = np.zeros((self.h, self.w), dtype=np.float32)
        mask = (d1 <= 0) & (d2 <= 0) & (d3 <= 0) 
        boundary_np[mask] = 1.0
        self.boundary = jnp.array(boundary_np)
        self.u = jnp.zeros_like(self.u)

    def init_mushroom_domain(self):
        # Classic Mushroom Billiard (Chaos)
        cx, cy = self.w // 2, self.h // 2
        r_cap = self.h // 4
        w_stem = r_cap 
        h_stem = self.h // 3
        
        # Cap (Semi-circle)
        dist_sq = (self.grid_x - cx)**2 + (self.grid_y - (cy - h_stem // 2))**2
        mask_cap = (dist_sq <= r_cap**2) & (self.grid_y <= (cy - h_stem // 2))
        
        # Stem
        mask_stem = (self.grid_x >= cx - w_stem // 2) & (self.grid_x <= cx + w_stem // 2) & \
                    (self.grid_y >= cy - h_stem // 2) & (self.grid_y <= cy + h_stem // 2)
                    
        boundary_np = np.zeros((self.h, self.w), dtype=np.float32)
        boundary_np[mask_cap | mask_stem] = 1.0
        self.boundary = jnp.array(boundary_np)
        self.u = jnp.zeros_like(self.u)

    def init_cannon_domain(self):
        # "Beam Cannon" - Designed for resonance and directional output
        # 1. Chamber: Large circle at left
        cx_cham, cy = int(self.w * 0.2), self.h // 2
        r_cham = int(self.h * 0.25)
        
        # 2. Barrel: Long narrow channel
        barrel_w = int(self.w * 0.4)
        barrel_h = int(self.h * 0.05) # Narrow
        barrel_x = cx_cham + r_cham // 2 
        
        # 3. Funnel/Coupler: Taper from Chamber to Barrel
        # Simple triangle connecting the two? Or just overlap.
        # Let's just abut them.
        
        # 4. Target Field: Open area at the end
        field_x = barrel_x + barrel_w
        
        # Create Masks
        
        # Chamber
        dist_cham = (self.grid_x - cx_cham)**2 + (self.grid_y - cy)**2
        mask_cham = (dist_cham <= r_cham**2)
        
        # Barrel
        mask_barrel = (self.grid_x >= cx_cham) & (self.grid_x <= field_x) & \
                      (self.grid_y >= cy - barrel_h // 2) & (self.grid_y <= cy + barrel_h // 2)

        # Target Field (Big rectangle at end)
        mask_field = (self.grid_x >= field_x) & \
                     (self.grid_y >= cy - r_cham) & (self.grid_y <= cy + r_cham)
        
        # Reflectors/Obstacles in Field?
        # Let's add a "Target" bar
        target_x = field_x + 100
        mask_target = (self.grid_x >= target_x) & (self.grid_x <= target_x + 20) & \
                      (self.grid_y >= cy - 50) & (self.grid_y <= cy + 50)
                      
        # Combine: Chamber | Barrel | Field - Target
        # Actually we want "Medium" = 1. So 
        mask_total = (mask_cham | mask_barrel | mask_field) & (~mask_target)
        
        boundary_np = np.zeros((self.h, self.w), dtype=np.float32)
        boundary_np[mask_total] = 1.0
        self.boundary = jnp.array(boundary_np)
        self.u = jnp.zeros_like(self.u)

    def init_resonator_domain(self):
        # Adjusted Cassegrain Antenna (Feed -> Secondary -> Primary -> Out)
        # Based on visual reference:
        # 1. Waveguide (Feed) enters from left, shoots Right.
        # 2. Hits Small Convex Mirror (Secondary) which reflects LEFT.
        # 3. Hits Large Concave Mirror (Primary) which reflects RIGHT (Collimated).
        
        cy = self.h // 2            # center y
        
        # 2. Secondary Mirror: Small Convex on Right, Facing Left.
        # Placed in front of the waveguide.
        # Let's say waveguide ends at 0.15w (120px for 800w, 210 for 1400w).
        # Secondary should be at roughly 0.3w?
        cx_sec = int(self.w * 0.45)
        r_sec = int(self.h * 0.15) # Smaller radius for tighter curve
        
        # Convex facing Right (requested) = Center of Curvature on Left.
        center_sec_x = cx_sec - r_sec
        dist_sec = (self.grid_x - center_sec_x)**2 + (self.grid_y - cy)**2
        
        # Shell for Secondary
        # We want the Right hemisphere (x >= center) so the bulge faces Right.
        mask_sec = (dist_sec >= r_sec**2) & (dist_sec <= (r_sec + 10)**2) & \
                   (self.grid_x >= center_sec_x) # Right half
                   
        # 3. Primary Mirror: Large Parabolic Dish on Left, Opening Right.
        # It needs to catch the reflection from Secondary.
        # So it should be BEHIND the Secondary, and maybe even behind the Waveguide tip?
        # Reference shows it wrapping around the waveguide.
        # Vertex at 0.1w?
        f_prim = self.h * 0.5 # Focal length
        vertex_prim = int(self.w * 0.05)
        
        x_curve = (self.grid_y - cy)**2 / (4 * f_prim) + vertex_prim
        mask_prim = (self.grid_x <= x_curve + 10) & (self.grid_x >= x_curve) & \
                    (self.grid_x <= self.w * 0.8)

        # Logic: Obstacles = (Prim | Sec)
        boundary_np = np.ones((self.h, self.w), dtype=np.float32)
        
        obstacles = mask_prim | mask_sec
        
        # No waveguide channel needed anymore
        
        boundary_np[obstacles] = 0.0
        
        self.boundary = jnp.array(boundary_np)
        self.u = jnp.zeros_like(self.u)

    def init_cardioid_domain(self):
        # Cardioid: r = 2a * (1 - cos(theta))
        # Cusp at origin (pointing right).
        
        # Center the shape (cusp) somewhat to the right so the bulk is in center
        cx, cy = int(self.w * 0.6), self.h // 2
        a = 150 # Size parameter
        
        dx = self.grid_x - cx
        dy = self.grid_y - cy
        rho = jnp.sqrt(dx**2 + dy**2)
        phi = jnp.arctan2(dy, dx)
        
        # Equation
        r_bound = 2 * a * (1 - jnp.cos(phi))
        
        # Interior mask
        mask = rho <= r_bound
        
        boundary_np = np.zeros((self.h, self.w), dtype=np.float32)
        boundary_np[mask] = 1.0 # Air inside
        
        self.boundary = jnp.array(boundary_np)
        self.u = jnp.zeros_like(self.u)

    def add_source(self, nx, ny):
        self.sources.append((nx, ny))
        self.cached_mask_needs_update = True

    def clear_sources(self):
        self.sources = []
        self.cached_mask_needs_update = True
        self.u = jnp.zeros_like(self.u)
        self.u_prev = jnp.zeros_like(self.u_prev)

    def _update_source_mask(self):
        # Reconstruct the static mask for all sources
        if not self.sources:
            self.source_static_mask = jnp.zeros((self.h, self.w), dtype=jnp.float32)
            return

        mask_composite = np.zeros((self.h, self.w), dtype=np.float32)
        # We'll do this in numpy on CPU since it happens rarely (on click)
        gx, gy = np.meshgrid(np.arange(self.w), np.arange(self.h))
        
        for (nx, ny) in self.sources:
            px = int(nx * self.w)
            py = int(ny * self.h)
            
            # Distance squared for Gaussian profile
            dist_sq = (gx - px)**2 + (gy - py)**2
            # Radius 3 Gaussian
            mask = np.exp(-dist_sq / (3.0**2))
            mask_composite += mask
            
        self.source_static_mask = jnp.array(np.clip(mask_composite, 0, 1))

    def step(self, source_enabled=True):
        if self.cached_mask_needs_update:
            self._update_source_mask()
            self.cached_mask_needs_update = False

        # Calculate source value for this timestep
        source_val = 0.0
        if source_enabled and self.sources:
            freq = 20.0
            source_val = 1.5 * jnp.sin(freq * self.time)
        
        # Run JAX kernel
        # We pass the static mask and flexible source value
        # But wait, if sources have different frequencies/phases this is limited.
        # The original code had freq=20 for all.
        # But each source applied it locally.
        # Here we apply the SAME source_val to ALL masked areas.
        # This matches the original logic where they all emit sin(freq*t).
        
        u_new = update_wave(self.u, self.u_prev, self.boundary, self.source_static_mask, source_val)
        
        self.u_prev = self.u
        self.u = u_new
        self.time += DT

    def get_render_array(self):
        # Run JAX kernel
        rgb_jax = render_magma(self.u, self.boundary)
        # Block until ready and convert to numpy
        return np.array(rgb_jax)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF)
    pygame.display.set_caption("Wave Billiard (JAX Accelerated)")
    clock = pygame.time.Clock()

    print("Initializing JAX simulation...")
    # Use native resolution matching window for 1:1 pixel mapping
    sim = WaveBilliardJAX(WIDTH, HEIGHT)
    
    # NO initial source as requested
    # sim.add_source(0.25, 0.5)

    running = True
    paused = False
    source_enabled = True
    
    # Surface for rendering (match simulation size)
    surf = pygame.Surface((WIDTH, HEIGHT))
    
    # Font for legend
    font = pygame.font.SysFont("monospace", 16)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_x:
                    sim.clear_sources()
                    print("Cleared all.")
                if event.key == pygame.K_l:
                    source_enabled = not source_enabled
                    print(f"Light: {source_enabled}")
                # Domain Controls
                if event.key == pygame.K_c:
                    sim.init_circular_domain()
                    print("Switched to Circular domain")
                if event.key == pygame.K_e:
                    sim.init_elliptical_domain()
                    print("Switched to Elliptical domain")
                if event.key == pygame.K_s:
                    sim.init_stadium_domain()
                    sim.clear_sources() # Clear on switch!
                    print("Switched to Stadium domain")
                # New Domains
                if event.key == pygame.K_r:
                    sim.init_rectangular_domain()
                    sim.clear_sources()
                    print("Switched to Rectangular domain")
                if event.key == pygame.K_t:
                    sim.init_triangular_domain()
                    sim.clear_sources()
                    print("Switched to Triangular domain")
                if event.key == pygame.K_m:
                    sim.init_mushroom_domain()
                    sim.clear_sources()
                    print("Switched to Mushroom domain")
                if event.key == pygame.K_b:
                    sim.init_cannon_domain()
                    sim.clear_sources()
                    print("Switched to Beam Cannon domain")
                if event.key == pygame.K_o:
                    sim.init_resonator_domain()
                    sim.clear_sources()
                    print("Switched to Optical Resonator domain")
                if event.key == pygame.K_h:
                    sim.init_cardioid_domain()
                    sim.clear_sources()
                    print("Switched to Cardioid domain")
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                nx = mx / WIDTH
                ny = my / HEIGHT
                sim.add_source(nx, ny)
                print(f"Added source at {nx:.2f}, {ny:.2f}")

        if not paused:
            sim.step(source_enabled)

        # Render
        # Ideally we don't pull back every frame if FPS is bottlenecked by transfer
        # But for 1024x1024 it should be fine (~3MB/frame)
        
        rgb_array = sim.get_render_array()
        
        # Pygame surface update
        pygame.surfarray.blit_array(surf, jnp.swapaxes(rgb_array, 0, 1)) # swap for pygame
        
        # Scale to screen
        pygame.transform.scale(surf, (WIDTH, HEIGHT), screen)
        
        # UI Legend
        # (Looking at previous code, it seems it didn't have on-screen text, only console print.
        # But user asked to "put it back", so I will add it properly.)
        legend_lines = [
            f"Wave Billiard (JAX) | FPS: {clock.get_fps():.1f} | Time: {sim.time:.2f}s",
            "CLICK: New Source | L: Toggle Light | X: Clear | SPACE: Pause | Q: Quit",
            f"Wave Billiard (JAX) | FPS: {clock.get_fps():.1f} | Time: {sim.time:.2f}s",
            "CLICK: New Source | L: Toggle Light | X: Clear | SPACE: Pause | Q: Quit",
            f"Domains: C=Circle... | B=BEAM | O=OPTICAL | H=CARDIOID | Sources: {len(sim.sources)}"
        ]
        
        for i, line in enumerate(legend_lines):
            text_surf = font.render(line, True, (0, 255, 128)) # Electric green
            screen.blit(text_surf, (10, 10 + i * 20))
        
        # UI
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
