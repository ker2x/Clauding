const PARTICLE_COUNT = 1000000;
const WORKGROUP_SIZE = 64;

// WGSL Shaders
const computeShaderCode = `
struct Particle {
    pos : vec2<f32>,
    vel : vec2<f32>,
};

struct SimParams {
    deltaT : f32,
    rule1Distance : f32,
    rule2Distance : f32,
    rule3Distance : f32,
    rule1Scale : f32,
    rule2Scale : f32,
    rule3Scale : f32,
};

@group(0) @binding(0) var<storage, read_write> particlesA : array<Particle>;
@group(0) @binding(1) var<storage, read_write> particlesB : array<Particle>;
@group(0) @binding(2) var<uniform> simParams : SimParams;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
    let index : u32 = GlobalInvocationID.x;
    if (index >= ${PARTICLE_COUNT}) {
        return;
    }

    var vPos : vec2<f32> = particlesA[index].pos;
    var vVel : vec2<f32> = particlesA[index].vel;

    // Simple physics: move by velocity
    vPos = vPos + vVel * simParams.deltaT;

    // Bounce off walls
    if (vPos.x < -1.0) { vPos.x = -1.0; vVel.x = -vVel.x; }
    if (vPos.x > 1.0) { vPos.x = 1.0; vVel.x = -vVel.x; }
    if (vPos.y < -1.0) { vPos.y = -1.0; vVel.y = -vVel.y; }
    if (vPos.y > 1.0) { vPos.y = 1.0; vVel.y = -vVel.y; }
    
    // Attractor logic (center pull)
    let center = vec2<f32>(0.0, 0.0);
    let dist = distance(vPos, center);
    let dir = normalize(center - vPos);
    
    // Gravity well
    vVel = vVel + dir * 0.0001;
    
    // Speed limit
    if (length(vVel) > 0.01) {
        vVel = normalize(vVel) * 0.01;
    }

    particlesB[index].pos = vPos;
    particlesB[index].vel = vVel;
}
`;

const vertexShaderCode = `
struct VertexOutput {
    @builtin(position) Position : vec4<f32>,
    @location(0) color : vec4<f32>,
};

struct Particle {
    pos : vec2<f32>,
    vel : vec2<f32>,
};

@group(0) @binding(0) var<storage, read> particles : array<Particle>;

@vertex
fn main(@builtin(vertex_index) VertexIndex : u32) -> VertexOutput {
    var particle = particles[VertexIndex];
    
    // Point size logic would normally be needed for point primitives if not supported by topology
    // But PointList is valid for WebGPU.
    // However, to make them visible, 'point-list' paints single pixels (usually).
    // For a million particles, single pixels are fine.
    
    var output : VertexOutput;
    output.Position = vec4<f32>(particle.pos.x, particle.pos.y, 0.0, 1.0);
    
    // Color based on velocity
    let speed = length(particle.vel);
    let nSpeed = smoothstep(0.0, 0.01, speed);
    
    // Nice gradient: blue -> purple -> orange
    let colorA = vec3<f32>(0.1, 0.4, 1.0);
    let colorB = vec3<f32>(1.0, 0.5, 0.0);
    
    output.color = vec4<f32>(mix(colorA, colorB, nSpeed), 0.8);
    return output;
}
`;

const fragmentShaderCode = `
@fragment
fn main(@location(0) color : vec4<f32>) -> @location(0) vec4<f32> {
    return color;
}
`;

async function init() {
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

    const canvas = document.getElementById("canvas");
    const context = canvas.getContext("webgpu");

    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    
    // Resize handling
    const observer = new ResizeObserver(entries => {
        for (const entry of entries) {
            const width = entry.contentBoxSize[0].inlineSize;
            const height = entry.contentBoxSize[0].blockSize;
            canvas.width = Math.max(1, Math.min(width, device.limits.maxTextureDimension2D));
            canvas.height = Math.max(1, Math.min(height, device.limits.maxTextureDimension2D));
            context.configure({
                device: device,
                format: presentationFormat,
                alphaMode: "premultiplied",
            });
        }
    });
    observer.observe(canvas);

    // Initial config
    context.configure({
        device: device,
        format: presentationFormat,
        alphaMode: "premultiplied",
    });

    // --- Resources ---

    // Create particles
    // Each particle is 2x vec2<f32> = 4 floats = 16 bytes.
    // 1M particles = 16MB.
    const particleUnitSize = 4 * 4; 
    const particleBufferSize = PARTICLE_COUNT * particleUnitSize;

    const initialParticleData = new Float32Array(PARTICLE_COUNT * 4);
    for (let i = 0; i < PARTICLE_COUNT; i++) {
        initialParticleData[i * 4 + 0] = (Math.random() * 2 - 1); // posX
        initialParticleData[i * 4 + 1] = (Math.random() * 2 - 1); // posY
        initialParticleData[i * 4 + 2] = (Math.random() * 2 - 1) * 0.001; // velX
        initialParticleData[i * 4 + 3] = (Math.random() * 2 - 1) * 0.001; // velY
    }

    const particleBufferA = device.createBuffer({
        size: particleBufferSize,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE,
        mappedAtCreation: true,
    });
    new Float32Array(particleBufferA.getMappedRange()).set(initialParticleData);
    particleBufferA.unmap();

    const particleBufferB = device.createBuffer({
        size: particleBufferSize,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE,
    });


    // Sim Params Uniform
    const simParamsBufferSize = 7 * 4; // 7 floats (aligned to 16 bytes boundary in blocks though? scalar is simple)
    // Actually standard uniform buffer alignment rules apply.
    // struct SimParams {
    //  deltaT : f32, // offset 0
    //  rule1Distance : f32, // offset 4
    //  ...
    // };
    // It's all f32s, so 4-byte alignment, tightly packed is okayish but often std140 is safer.
    // However, let's keep it simple. If we use writeBuffer, it's just raw bytes.
    
    const simParamsBuffer = device.createBuffer({
        size: 32, // Padding to 8 floats just to be safe (32 bytes)
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });


    // --- Pipelines ---

    // Compute Pipeline
    const computeModule = device.createShaderModule({ code: computeShaderCode });
    const computeBindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
    });

    const computePipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] }),
        compute: {
            module: computeModule,
            entryPoint: "main",
        },
    });

    // Render Pipeline
    const vertexModule = device.createShaderModule({ code: vertexShaderCode });
    const fragmentModule = device.createShaderModule({ code: fragmentShaderCode });
    
    const renderBindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } }, // Reading the output of compute
        ],
    });

    const renderPipeline = device.createRenderPipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [renderBindGroupLayout] }),
        vertex: {
            module: vertexModule,
            entryPoint: "main",
        },
        fragment: {
            module: fragmentModule,
            entryPoint: "main",
            targets: [{ format: presentationFormat }],
        },
        primitive: {
            topology: "point-list",
        },
    });

    // Bind Groups
    // We need 2 bind groups for ping-ponging in compute
    // A -> B
    const computeBindGroup0 = device.createBindGroup({
        layout: computeBindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: particleBufferA } },
            { binding: 1, resource: { buffer: particleBufferB } },
            { binding: 2, resource: { buffer: simParamsBuffer } },
        ],
    });

    // B -> A
    const computeBindGroup1 = device.createBindGroup({
        layout: computeBindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: particleBufferB } },
            { binding: 1, resource: { buffer: particleBufferA } },
            { binding: 2, resource: { buffer: simParamsBuffer } },
        ],
    });

    // Render Bind Groups
    const renderBindGroupA = device.createBindGroup({
        layout: renderBindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: particleBufferA } },
        ],
    });

    const renderBindGroupB = device.createBindGroup({
        layout: renderBindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: particleBufferB } },
        ],
    });


    // --- Frame Loop ---
    let frameCount = 0;
    let t = 0;
    let frame = 0;
    const fpsElem = document.getElementById("fps");
    let lastTime = 0;

    function frameLoop(timestamp) {
        if (!lastTime) lastTime = timestamp;
        const deltaTime = (timestamp - lastTime) / 1000;
        lastTime = timestamp;

        // Update FPS every second
        frameCount++;
        if (Math.floor(timestamp / 1000) > Math.floor((timestamp - deltaTime * 1000) / 1000)) {
            fpsElem.innerText = `FPS: ${frameCount}`;
            frameCount = 0;
        }

        // Update Sim Params
        const simParams = new Float32Array([
            0.01 + Math.sin(timestamp * 0.001) * 0.005, // dynamic DeltaT?
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0 // padding
        ]);
        device.queue.writeBuffer(simParamsBuffer, 0, simParams);

        const commandEncoder = device.createCommandEncoder();

        // 1. Compute Pass
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(computePipeline);
        // Ping-pong choice
        const bindGroup = (frame % 2 === 0) ? computeBindGroup0 : computeBindGroup1;
        computePass.setBindGroup(0, bindGroup);
        const workgroupCount = Math.ceil(PARTICLE_COUNT / WORKGROUP_SIZE);
        computePass.dispatchWorkgroups(workgroupCount);
        computePass.end();

        // 2. Render Pass
        const textureView = context.getCurrentTexture().createView();
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: textureView,
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                loadOp: "clear",
                storeOp: "store",
            }],
        });
        renderPass.setPipeline(renderPipeline);
        // If we computed A->B, then B has the latest positions, so we render B
        const renderBG = (frame % 2 === 0) ? renderBindGroupB : renderBindGroupA;
        renderPass.setBindGroup(0, renderBG);
        renderPass.draw(PARTICLE_COUNT);
        renderPass.end();

        device.queue.submit([commandEncoder.finish()]);

        frame++;
        requestAnimationFrame(frameLoop);
    }

    requestAnimationFrame(frameLoop);
}

init();
