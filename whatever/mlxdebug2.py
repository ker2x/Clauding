import time
import gc
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# --- Configuration ---
INPUT_DIM = 1024
HIDDEN_DIM = 4096
OUTPUT_DIM = 1024
BATCH_SIZE = 64
STEPS_PER_SESSION = 200

class LargeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
        ]

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x

def run_session(session_id, offset_mb):
    # 1. Memory Spacer: Force address shift
    spacer_size = int(offset_mb * 1024 * 1024 / 4) # float32 size
    spacer = None
    if spacer_size > 0:
        spacer = mx.zeros((spacer_size,))
        mx.eval(spacer)

    print(f"\n[Session {session_id}] Memory Offset: {offset_mb} MB")

    # 2. Init Model
    model = LargeModel()
    optimizer = optim.AdamW(learning_rate=1e-4)
    
    def loss_fn(m, x, y):
        return nn.losses.mse_loss(m(x), y, reduction="mean")

    # NO COMPILE - Pure Eager Execution
    def step(x, y):
        loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
        optimizer.update(model, grads)
        return loss

    # 3. Pre-generate Data (Remove data loader variance)
    # We generate on GPU and eval immediately so they sit in memory
    data_x = mx.random.normal((STEPS_PER_SESSION, BATCH_SIZE, INPUT_DIM))
    data_y = mx.random.normal((STEPS_PER_SESSION, BATCH_SIZE, OUTPUT_DIM))
    mx.eval(data_x, data_y)

    # 4. Warmup (Run 5 steps to stabilize allocator)
    for i in range(5):
        step(data_x[i], data_y[i])
        # Sync optimizer state to ensure it's initialized
        mx.eval(optimizer.state)

    # 5. Timed Run
    start_time = time.perf_counter()
    
    for i in range(5, STEPS_PER_SESSION):
        loss = step(data_x[i], data_y[i])
        
        # Sync every 10 steps to keep queue moving without stalling CPU
        if i % 10 == 0:
             mx.eval(loss, optimizer.state)
    
    # Final Hard Sync
    mx.eval(optimizer.state, loss)
    end_time = time.perf_counter()

    # 6. Calc Stats
    total_time = end_time - start_time
    steps_counted = STEPS_PER_SESSION - 5
    avg_ms = (total_time / steps_counted) * 1000
    
    print(f"   -> Avg Speed: {avg_ms:.2f} ms/step")
    
    # Cleanup to ensure next session is independent
    del model, optimizer, spacer, data_x, data_y
    mx.clear_cache()
    gc.collect()
    
    return avg_ms

def main():
    print("--- MLX Memory Layout Stress Test (No Compile) ---")
    
    results = []
    # Offsets shift the starting memory address
    offsets = [0, 128, 256, 512]
    
    for i, offset in enumerate(offsets):
        ms = run_session(i+1, offset)
        results.append(ms)

    print("\n--- Summary ---")
    baseline = min(results)
    for i, r in enumerate(results):
        diff = (r / baseline)
        tag = "FAST" if diff < 1.15 else f"SLOW ({diff:.2f}x)"
        print(f"Offset {offsets[i]}MB: {r:.2f} ms | {tag}")

if __name__ == "__main__":
    main()
