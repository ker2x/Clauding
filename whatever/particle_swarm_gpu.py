import pygame
import numpy as np
import moderngl as mgl
import random
import math
from collections import deque

# Window setup
WIDTH, HEIGHT = 1200, 800
FPS = 60
PARTICLE_COUNT = 3000

# Colors for groups (6 groups)
COLORS = [
    (255, 100, 100),    # Red
    (100, 255, 100),    # Green
    (100, 100, 255),    # Blue
    (255, 255, 100),    # Yellow
    (255, 100, 255),    # Magenta
    (100, 255, 255),    # Cyan
]

NUM_GROUPS = len(COLORS)

# Interaction matrix
INTERACTION_MATRIX = np.array([
    [-0.15,  0.5,   0.3,   0.2,   0.4,  -0.3 ],
    [-0.4,  -0.15, -0.5,   0.4,  -0.3,   0.5 ],
    [ 0.3,  -0.45, -0.15, -0.35, -0.4,   0.55],
    [ 0.15,  0.35, -0.3,  -0.15,  0.25, -0.2 ],
    [ 0.35, -0.25, -0.35,  0.2,  -0.15, -0.5 ],
    [-0.25,  0.45,  0.5,  -0.15, -0.45, -0.15],
], dtype=np.float32)

class ParticleSwarmGPU:
    def __init__(self):
        # Pygame setup
        pygame.init()
        self.display = pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption("Particle Swarm GPU (Compute Shaders)")
        self.clock = pygame.time.Clock()
        self.running = True

        # Modern GL context (auto-detect version)
        self.ctx = mgl.create_context()
        print(f"OpenGL Version: {self.ctx.version_code}")

        # Check if compute shaders are available
        if self.ctx.version_code < 430:
            print("WARNING: Compute shaders require OpenGL 4.3+")
            print("Falling back to CPU physics with GPU rendering")

        self.ctx.enable(mgl.BLEND)
        self.ctx.blend_func = (mgl.SRC_ALPHA, mgl.ONE)

        print(f"OpenGL Version: {self.ctx.version_code}")
        print(f"Max Compute Work Group Size: {self.ctx.max_compute_work_group_size}")

        self.time = 0
        self.paused = False

        # Particle data (position, velocity, group)
        self.particle_data = self._init_particles()
        self.num_particles = PARTICLE_COUNT

        # GPU buffers
        self._create_buffers()

        # Shaders
        self._create_shaders()

        # Trails (CPU side)
        self.trails = [deque(maxlen=15) for _ in range(PARTICLE_COUNT)]
        self.trail_update_counter = 0

    def _init_particles(self):
        """Initialize particle data"""
        particles = []
        particles_per_group = PARTICLE_COUNT // NUM_GROUPS

        for group_id in range(NUM_GROUPS):
            cx = (WIDTH / (NUM_GROUPS + 1)) * (group_id + 1)
            cy = HEIGHT / 2

            for _ in range(particles_per_group):
                x = cx + random.gauss(0, 40)
                y = cy + random.gauss(0, 40)
                x = max(10, min(WIDTH - 10, x))
                y = max(10, min(HEIGHT - 10, y))

                vx = random.uniform(-1, 1)
                vy = random.uniform(-1, 1)

                particles.append([x, y, vx, vy, group_id, 0])  # x, y, vx, vy, group, padding

        return np.array(particles, dtype=np.float32)

    def _create_buffers(self):
        """Create GPU buffers"""
        # Particle SSBO (Shader Storage Buffer Object)
        self.particle_buffer = self.ctx.buffer(self.particle_data)

        # Interaction matrix UBO
        self.interaction_buffer = self.ctx.buffer(INTERACTION_MATRIX)

        # Indirect dispatch buffer (not needed for basic version)

    def _create_shaders(self):
        """Create compute shader for physics and rendering shader"""
        # Compute shader for particle physics
        compute_shader = """
        #version 430

        layout(std430, binding = 0) buffer particle_data {
            vec4 particles[];
        };

        uniform mat2x3 interaction_matrix;
        uniform uint num_particles;
        uniform float dt;
        uniform float width;
        uniform float height;

        layout (local_size_x = 256) in;

        void main() {
            uint idx = gl_GlobalInvocationID.x;
            if (idx >= num_particles) return;

            vec4 p = particles[idx];  // x, y, vx, vy
            float px = p.x;
            float py = p.y;
            float vx = p.z;
            float vy = p.w;
            int group = int(particles[idx + num_particles].x);

            // Compute forces from all other particles
            vec2 force = vec2(0.0);

            for (uint j = 0; j < num_particles; j++) {
                if (j == idx) continue;

                vec4 other = particles[j];
                float ox = other.x;
                float oy = other.y;
                int ogroup = int(particles[j + num_particles].x);

                float dx = ox - px;
                float dy = oy - py;
                float dist_sq = dx*dx + dy*dy + 1e-6;
                float dist = sqrt(dist_sq);

                float force_mag = 0.0;

                if (dist < 30.0) {
                    force_mag = -1.0 / (dist + 1.0);
                } else if (dist < 150.0) {
                    force_mag = 1.0 / (dist + 1.0);
                } else {
                    force_mag = 0.2 / (dist + 1.0);
                }

                // Apply interaction matrix
                float interaction = interaction_matrix[group * 6 + ogroup];
                force_mag *= interaction * 3.0;

                force += force_mag * normalize(vec2(dx, dy));
            }

            // Update velocity
            vx = (vx + force.x * dt * 0.5) * 0.92;
            vy = (vy + force.y * dt * 0.5) * 0.92;

            // Update position
            px += vx * dt;
            py += vy * dt;

            // Boundary conditions
            float margin = 10.0;
            if (px < margin) { px = margin; vx = -vx; }
            if (px > width - margin) { px = width - margin; vx = -vx; }
            if (py < margin) { py = margin; vy = -vy; }
            if (py > height - margin) { py = height - margin; vy = -vy; }

            // Write back
            particles[idx] = vec4(px, py, vx, vy);
        }
        """

        # Compile compute shader
        self.compute_program = self.ctx.compute_shader(compute_shader)

    def update(self):
        """Update particles on GPU"""
        if self.paused:
            return

        # Bind buffers
        self.particle_buffer.bind_to_uniform_block(0)
        self.interaction_buffer.bind_to_uniform_block(1)

        # Set uniforms
        self.compute_program['num_particles'] = self.num_particles
        self.compute_program['dt'] = 0.016
        self.compute_program['width'] = float(WIDTH)
        self.compute_program['height'] = float(HEIGHT)

        # Dispatch compute shader
        # Use work groups of 256 threads
        work_groups = (self.num_particles + 255) // 256
        self.compute_program.run(group_x=work_groups)

        # Read back particle data for trails (every 2 frames to save bandwidth)
        self.trail_update_counter += 1
        if self.trail_update_counter % 2 == 0:
            self.particle_data = np.frombuffer(self.particle_buffer.read(), dtype=np.float32)
            for i in range(min(self.num_particles, len(self.trails))):
                self.trails[i].append((self.particle_data[i*6], self.particle_data[i*6+1]))

        self.time += 1

    def draw(self):
        """Render particles"""
        self.ctx.clear(0.06, 0.06, 0.08)

        # Read particle positions for rendering (copy from GPU)
        self.particle_data = np.frombuffer(self.particle_buffer.read(), dtype=np.float32)

        # Draw trails first (faint)
        for i in range(min(self.num_particles, len(self.trails))):
            trail = list(self.trails[i])
            if len(trail) > 1:
                group = int(self.particle_data[i*6 + 4])
                color = COLORS[group]
                dimmed = tuple(c // 2 for c in color)

                # Draw trail as lines
                for j in range(len(trail) - 1):
                    x1, y1 = trail[j]
                    x2, y2 = trail[j + 1]
                    # Normalize to GL coordinates (-1 to 1)
                    glx1 = (x1 / WIDTH) * 2 - 1
                    gly1 = 1 - (y1 / HEIGHT) * 2
                    glx2 = (x2 / WIDTH) * 2 - 1
                    gly2 = 1 - (y2 / HEIGHT) * 2

        # Draw particles as points
        # Create vertex buffer for current positions
        vertex_data = np.zeros(self.num_particles * 7, dtype=np.float32)
        for i in range(self.num_particles):
            x = self.particle_data[i*6]
            y = self.particle_data[i*6 + 1]
            group = int(self.particle_data[i*6 + 4])
            color = COLORS[group]

            # Normalize to GL coordinates
            glx = (x / WIDTH) * 2 - 1
            gly = 1 - (y / HEIGHT) * 2

            vertex_data[i*7] = glx
            vertex_data[i*7 + 1] = gly
            vertex_data[i*7 + 2] = 0.0
            vertex_data[i*7 + 3] = color[0] / 255.0
            vertex_data[i*7 + 4] = color[1] / 255.0
            vertex_data[i*7 + 5] = color[2] / 255.0
            vertex_data[i*7 + 6] = 1.0

        # Create vertex buffer and render
        vbo = self.ctx.buffer(vertex_data)
        vao = self.ctx.simple_vertex_array(None, vbo)

        # Simple point rendering
        self.ctx.disable(mgl.CULL_FACE)
        vao.render(mgl.POINTS)

        # Swap buffers
        pygame.display.flip()

    def handle_events(self):
        """Handle input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.running = False
                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.particle_data = self._init_particles()
                    self.particle_buffer.write(self.particle_data)
                    self.time = 0

    def run(self):
        """Main loop"""
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)

        pygame.quit()

if __name__ == "__main__":
    swarm = ParticleSwarmGPU()
    swarm.run()
