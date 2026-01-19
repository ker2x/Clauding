struct Particle {
    pos: vec2f,
    vel: vec2f,
    color: vec4f,
    life: f32,
    p_type: u32, // 'type' is a reserved keyword in some contexts, safe to use p_type
}

struct SimParams {
    delta_time: f32,
    time: f32,
    screen_width: f32,
    screen_height: f32,
    player_x: f32,
    player_y: f32,
    mouse_x: f32,
    mouse_y: f32,
    mouse_down: u32,
}

@group(0) @binding(0) var<uniform> params: SimParams;
@group(1) @binding(0) var<storage, read_write> particlesSim: array<Particle>;
@group(2) @binding(0) var<storage, read> particlesRender: array<Particle>;

@compute @workgroup_size(64)
fn simulate(@builtin(global_invocation_id) global_id: vec3u) {
    let index = global_id.x;
    if (index >= arrayLength(&particlesSim)) {
        return;
    }

    var particle = particlesSim[index];

    var force = vec2f(0.0, 0.0);
    
    // 1. Swarm Behavior: Seek Player
    let player_pos = vec2f(params.player_x, params.player_y);
    let to_player = player_pos - particle.pos;
    let dist_player_sq = dot(to_player, to_player);
    let dist_player = sqrt(dist_player_sq);
    
    if (dist_player > 1.0) {
        // Seek force
        force += normalize(to_player) * 150.0;
    }
    
    // 2. Weapon: Shockwave (Repel from Mouse/Player?)
    // Game: Move with WASD. Click to create a Repulsion field at Mouse Cursor?
    // Let's do: Repel from Mouse Cursor (Simulate a bomb)
    let mouse_pos = vec2f(params.mouse_x, params.mouse_y);
    let to_mouse = mouse_pos - particle.pos;
    let dist_mouse = length(to_mouse);
    
    if (params.mouse_down == 1u) {
        // Shockwave active
        if (dist_mouse < 300.0) {
            // Strong repulsion
            let strength = 20000.0 / (dist_mouse + 1.0);
            force -= normalize(to_mouse) * strength;
        }
    }
    
    // Apply Force
    particle.vel += force * params.delta_time;
    
    // Drag (Damping) - critical for swarm stability
    particle.vel *= 0.96;

    // Movement
    particle.pos += particle.vel * params.delta_time;

    // Boundary Bouncing
    if (particle.pos.x < 0.0) { particle.pos.x = 0.0; particle.vel.x *= -1.0; }
    if (particle.pos.x > params.screen_width) { particle.pos.x = params.screen_width; particle.vel.x *= -1.0; }
    if (particle.pos.y < 0.0) { particle.pos.y = 0.0; particle.vel.y *= -1.0; }
    if (particle.pos.y > params.screen_height) { particle.pos.y = params.screen_height; particle.vel.y *= -1.0; }

    particlesSim[index] = particle;
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
}

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VertexOutput {
    let particle = particlesRender[instanceIndex];
    let corner = vertexIndex % 6u;
    
    var pos = particle.pos;
    // Size based on velocity
    let speed = length(particle.vel);
    let size = 3.0 + speed * 0.01; 
    
    // Quad expansion
    var offset = vec2f(0.0, 0.0);
    if (corner == 0u || corner == 3u) { offset = vec2f(-1.0, -1.0); }
    else if (corner == 1u) { offset = vec2f(1.0, -1.0); }
    else if (corner == 2u || corner == 4u) { offset = vec2f(1.0, 1.0); }
    else if (corner == 5u) { offset = vec2f(-1.0, 1.0); }
    
    pos += offset * size;
    
    let clip_x = (pos.x / params.screen_width) * 2.0 - 1.0;
    let clip_y = (pos.y / params.screen_height) * 2.0 - 1.0;
    
    var out: VertexOutput;
    out.position = vec4f(clip_x, -clip_y, 0.0, 1.0);
    
    // Color: Velocity heat map
    // Slow = Blue/Cyan, Fast = Red/Orange/Pink
    let t = clamp(speed / 500.0, 0.0, 1.0);
    let color_slow = vec4f(0.0, 0.8, 1.0, 0.5); // Cyan
    let color_fast = vec4f(1.0, 0.2, 0.5, 0.8); // Pink/Red
    out.color = mix(color_slow, color_fast, t);
    
    return out;
}

@fragment
fn fs_main(@location(0) color: vec4f) -> @location(0) vec4f {
    return color;
}
