import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# --- 1. Diagnostic Tool ---
class MLXBenchmark:
    """
    Wraps the training loop to track execution time and memory usage.
    """
    def __init__(self, log_interval=50, threshold=1.5):
        self.log_interval = log_interval
        self.threshold = threshold
        self.step_times = []
        self.start_time = None
        
    def __enter__(self):
        # Force a sync before starting the timer so we measure ONLY this step
        mx.eval(mx.random.uniform(shape=(1,))) 
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def mark_step(self, step, *tensors_to_sync):
        # FORCE SYNC: This makes the CPU wait for the GPU to finish.
        mx.eval(*tensors_to_sync)
        
        # Measure
        current_time = time.perf_counter()
        dt = current_time - self.start_time
        self.step_times.append(dt)
        self.start_time = current_time

        # Analyze
        if step > 10 and step % self.log_interval == 0:
            avg = sum(self.step_times[-self.log_interval:]) / self.log_interval
            peak = max(self.step_times[-self.log_interval:])
            variance = peak / avg if avg > 0 else 0
            
            status = f"Step {step}: Avg {avg*1000:.2f}ms"
            
            if variance > self.threshold:
                # Use updated API calls to avoid deprecation warnings
                active_mem = mx.get_active_memory() / 1e6
                cache_mem = mx.get_cache_memory() / 1e6
                
                print(f"\n[⚠️ VARIANCE DETECTED] {status} | Peak: {peak*1000:.2f}ms ({variance:.2f}x slower)")
                print(f"   Active Memory: {active_mem:.2f} MB")
                print(f"   Cache Memory:  {cache_mem:.2f} MB")
            else:
                print(status, end="\r")

# --- 2. Model Definition ---
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ]

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x

def generate_synthetic_batch(batch_size, input_dim, output_dim):
    X = mx.random.normal((batch_size, input_dim))
    y = mx.random.randint(0, output_dim, (batch_size,))
    return X, y

def loss_fn(model, X, y):
    logits = model(X)
    return nn.losses.cross_entropy(logits, y, reduction="mean")

# --- 3. Pure Python Step (No Compile) ---
def step(model, optimizer, X, y):
    loss, grads = nn.value_and_grad(model, loss_fn)(model, X, y)
    optimizer.update(model, grads)
    return loss

# --- 4. Main Execution ---
def main():
    # Hyperparameters
    BATCH_SIZE = 128
    INPUT_DIM = 512
    HIDDEN_DIM = 1024
    OUTPUT_DIM = 10
    NUM_STEPS = 10000  # Increased duration
    CACHE_CLEAR_INTERVAL = 500 # The Heartbeat Fix
    
    print(f"--- Starting MLX Stress Test (Long Run + Heartbeat) ---")
    print(f"Goal: Detect random execution spikes > 1.5x average.")
    print(f"Heartbeat: Clearing cache every {CACHE_CLEAR_INTERVAL} steps.\n")

    model = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    optimizer = optim.AdamW(learning_rate=1e-3)
    
    benchmark = MLXBenchmark(log_interval=50, threshold=1.5)

    try:
        for i in range(NUM_STEPS):
            X, y = generate_synthetic_batch(BATCH_SIZE, INPUT_DIM, OUTPUT_DIM)
            
            with benchmark:
                loss = step(model, optimizer, X, y)
                benchmark.mark_step(i, loss, optimizer.state)

            # THE FIX: Periodic Cache Clearing
            # Prevents allocator fragmentation and forces a clean slate
            if i > 0 and i % CACHE_CLEAR_INTERVAL == 0:
                mx.clear_cache()

    except KeyboardInterrupt:
        print("\nTest stopped by user.")

    print("\n--- Test Complete ---")
    print(f"Final Peak Memory: {mx.get_peak_memory() / 1e6:.2f} MB")

if __name__ == "__main__":
    main()
