const NUM_PARTICLES = 100000;
// Particle struct size: 2f(pos)+2f(vel)+4f(color)+1f(life)+1u(type) 
// = 8 + 8 + 16 + 4 + 4 = 40 bytes. 
// Padding needed? Struct in WGSL has specific alignment.
// vec2f is 8 bytes, vec4f is 16.
// Layout:
// pos: 0-8
// vel: 8-16
// color: 16-32
// life: 32-36
// p_type: 36-40
// Total 40 bytes. But vec4<f32> needs 16-byte alignment.
// pos(0), vel(8) ok. color(16) aligned. life(32), type(36).
// Array stride must be multiple of 16? No, multiple of largest alignment?
// WGSL alignments: vec4f=16. So struct size likely padded to 48.
const PARTICLE_SIZE = 48; // Rounded up to nearest 16 for safety/simplicity in array

export async function init() {
    if (!navigator.gpu) {
        alert("WebGPU not supported on this browser.");
        return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        alert("No appropriate GPUAdapter found.");
        return;
    }

    const device = await adapter.requestDevice();
    const canvas = document.getElementById("gpu-canvas");
    const context = canvas.getContext("webgpu");

    // Handle resizing
    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    window.addEventListener("resize", resize);
    resize();

    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device: device,
        format: format,
        alphaMode: "premultiplied",
    });

    // Load shaders
    const shaderCode = await (await fetch("./shaders.wgsl")).text();
    const shaderModule = device.createShaderModule({
        label: "Game Shaders",
        code: shaderCode,
    });

    // Buffers
    const particlesBufferSize = NUM_PARTICLES * PARTICLE_SIZE;
    const particlesBuffer = device.createBuffer({
        label: "Particles Buffer",
        size: particlesBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const initData = new Float32Array(particlesBufferSize / 4);
    for (let i = 0; i < NUM_PARTICLES; i++) {
        const offset = i * (PARTICLE_SIZE / 4);
        // Pos
        initData[offset] = Math.random() * window.innerWidth;
        initData[offset + 1] = Math.random() * window.innerHeight;
        // Vel
        initData[offset + 2] = (Math.random() - 0.5) * 50;
        initData[offset + 3] = (Math.random() - 0.5) * 50;
        // Color
        initData[offset + 4] = Math.random(); // R
        initData[offset + 5] = Math.random(); // G
        initData[offset + 6] = 1.0;           // B
        initData[offset + 7] = 1.0;           // A
        // Life
        initData[offset + 8] = 100.0;
        // Type (uint view needed for this one, but Float32Array write works if careful or use DataView)
        // Here we just write float, reinterpreting in shader might be messy unless we use a separate init loop or DataView.
        // Easier: use ArrayBuffer and multiple views.
    }

    // Proper Buffer Initialization
    const cpuBuffer = new ArrayBuffer(particlesBufferSize);
    const f32View = new Float32Array(cpuBuffer);
    const u32View = new Uint32Array(cpuBuffer);

    for (let i = 0; i < NUM_PARTICLES; i++) {
        // Stride in float32/uint32 indices (4 bytes each)
        const base = i * (PARTICLE_SIZE / 4);

        // pos
        f32View[base + 0] = Math.random() * window.innerWidth;
        f32View[base + 1] = Math.random() * window.innerHeight; // Ensure full height coverage
        // vel
        f32View[base + 2] = (Math.random() - 0.5) * 100;
        f32View[base + 3] = (Math.random() - 0.5) * 100;
        // color (RGBA)
        f32View[base + 4] = 0.0; // R
        f32View[base + 5] = Math.random() * 0.5 + 0.5; // G (Cyans/Greens)
        f32View[base + 6] = 1.0; // B
        f32View[base + 7] = 1.0; // A
        // life
        f32View[base + 8] = 100.0;
        // type
        u32View[base + 9] = 1; // 1 = Enemy (for test)
        // padding: base+10, base+11 (if 48 bytes = 12 floats)
    }

    device.queue.writeBuffer(particlesBuffer, 0, cpuBuffer);

    // Uniform Buffer (SimParams)
    // f32: delta_time, time, width, height, p_x, p_y, m_x, m_y, u32: m_down, pad, pad, pad
    const paramsBufferSize = 4 * 12; // 48 bytes
    const paramsBuffer = device.createBuffer({
        label: "Params Buffer",
        size: paramsBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const paramsValues = new ArrayBuffer(paramsBufferSize);
    const paramsF32 = new Float32Array(paramsValues);
    const paramsU32 = new Uint32Array(paramsValues);


    // Bind Group Layouts (Split to avoid usage conflicts)
    const bindGroupLayout0 = device.createBindGroupLayout({
        entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX, buffer: { type: "uniform" } }],
    });
    const bindGroupLayout1 = device.createBindGroupLayout({
        entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }],
    });
    const bindGroupLayout2 = device.createBindGroupLayout({
        entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } }],
    });

    const bindGroup0 = device.createBindGroup({
        layout: bindGroupLayout0,
        entries: [{ binding: 0, resource: { buffer: paramsBuffer } }],
    });
    const bindGroup1 = device.createBindGroup({
        layout: bindGroupLayout1,
        entries: [{ binding: 0, resource: { buffer: particlesBuffer } }],
    });
    const bindGroup2 = device.createBindGroup({
        layout: bindGroupLayout2,
        entries: [{ binding: 0, resource: { buffer: particlesBuffer } }],
    });

    // Empty Bind Group for gaps (Group 1 in Render)
    const bindGroupLayoutEmpty = device.createBindGroupLayout({ entries: [] });
    const bindGroupEmpty = device.createBindGroup({
        layout: bindGroupLayoutEmpty,
        entries: [],
    });

    const computePipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout0, bindGroupLayout1],
    });

    const renderPipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout0, bindGroupLayoutEmpty, bindGroupLayout2],
    });

    // Pipelines
    const computePipeline = device.createComputePipeline({
        label: "Simulation Pipeline",
        layout: computePipelineLayout,
        compute: {
            module: shaderModule,
            entryPoint: "simulate",
        },
    });

    const renderPipeline = device.createRenderPipeline({
        label: "Render Pipeline",
        layout: renderPipelineLayout,
        vertex: {
            module: shaderModule,
            entryPoint: "vs_main",
        },
        fragment: {
            module: shaderModule,
            entryPoint: "fs_main",
            targets: [{
                format: format, blend: {
                    color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                    alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' },
                }
            }], // Additive blending for neon glow
        },
        primitive: {
            topology: "triangle-list",
        },
    });

    let lastTime = performance.now();

    // Input Handling
    canvas.addEventListener("mousemove", (e) => {
        paramsF32[6] = e.clientX;
        paramsF32[7] = e.clientY;
    });
    canvas.addEventListener("mousedown", () => { paramsU32[8] = 1; });
    canvas.addEventListener("mouseup", () => { paramsU32[8] = 0; });

    const keys = {};
    window.addEventListener("keydown", (e) => { keys[e.code] = true; });
    window.addEventListener("keyup", (e) => { keys[e.code] = false; });

    let playerX = canvas.width / 2;
    let playerY = canvas.height / 2;
    paramsF32[4] = playerX;
    paramsF32[5] = playerY;

    function frame() {
        const now = performance.now();
        const dt = (now - lastTime) / 1000;
        lastTime = now;

        // Update User Input
        const speed = 300 * dt;
        if (keys["KeyW"]) playerY -= speed;
        if (keys["KeyS"]) playerY += speed;
        if (keys["KeyA"]) playerX -= speed;
        if (keys["KeyD"]) playerX += speed;

        // Clamp player
        playerX = Math.max(0, Math.min(canvas.width, playerX));
        playerY = Math.max(0, Math.min(canvas.height, playerY));

        paramsF32[0] = dt;
        paramsF32[1] = now / 1000;
        paramsF32[2] = canvas.width;
        paramsF32[3] = canvas.height;
        paramsF32[4] = playerX;
        paramsF32[5] = playerY;
        // paramsF32[6/7] updated by event listener

        device.queue.writeBuffer(paramsBuffer, 0, paramsValues);

        const commandEncoder = device.createCommandEncoder();

        // 1. Compute Pass
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup0); // Uniforms
        passEncoder.setBindGroup(1, bindGroup1); // Particles RW
        const workgroupSize = 64;
        passEncoder.dispatchWorkgroups(Math.ceil(NUM_PARTICLES / workgroupSize));
        passEncoder.end();

        // 2. Render Pass
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: context.getCurrentTexture().createView(),
                loadOp: "clear",
                clearValue: { r: 0.05, g: 0.05, b: 0.05, a: 1.0 }, // Dark bg
                storeOp: "store",
            }],
        });
        renderPass.setPipeline(renderPipeline);
        renderPass.setBindGroup(0, bindGroup0); // Uniforms
        renderPass.setBindGroup(1, bindGroupEmpty); // Empty (skip)
        renderPass.setBindGroup(2, bindGroup2); // Particles RO
        // Draw 6 vertices per particle (quad)
        renderPass.draw(6, NUM_PARTICLES);
        renderPass.end();

        device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
}

init();
