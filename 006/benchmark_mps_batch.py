
import time
import torch
import numpy as np
from sac import SACAgent, ReplayBuffer

def benchmark_device(device_name, batch_sizes=[256, 512, 1024, 2048], n_steps=100):
    print(f"\nBenchmarking {device_name}...")
    device = torch.device(device_name)
    
    state_dim = 71
    action_dim = 2
    
    results = {}
    
    for batch_size in batch_sizes:
        # Initialize agent
        agent = SACAgent(state_dim, action_dim, device=device)
        
        # Initialize buffer
        buffer = ReplayBuffer(capacity=batch_size * 5, state_shape=state_dim, action_dim=action_dim, device=device)
        
        # Fill buffer
        for _ in range(batch_size * 2):
            state = np.random.randn(state_dim).astype(np.float32)
            action = np.random.randn(action_dim).astype(np.float32)
            next_state = np.random.randn(state_dim).astype(np.float32)
            buffer.push(state, action, 0.0, next_state, False, False)
            
        # Warmup
        agent.update(buffer, batch_size)
        
        # Benchmark update (Training)
        start_time = time.time()
        for _ in range(n_steps):
            agent.update(buffer, batch_size)
        if device_name == 'mps':
            torch.mps.synchronize()
        elif device_name == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
        
        avg_train = (end_time - start_time) / n_steps * 1000
        results[batch_size] = avg_train
        print(f"  Batch {batch_size}: {avg_train:.3f} ms/step")
        
    return results

if __name__ == "__main__":
    print(f"MPS Available: {torch.backends.mps.is_available()}")
    
    cpu_results = benchmark_device('cpu')
    
    if torch.backends.mps.is_available():
        mps_results = benchmark_device('mps')
        
        print("\nComparison (MPS vs CPU Training Speed):")
        print(f"{'Batch':<10} | {'CPU (ms)':<10} | {'MPS (ms)':<10} | {'Speedup':<10}")
        print("-" * 46)
        for bs in cpu_results:
            cpu_time = cpu_results[bs]
            mps_time = mps_results[bs]
            speedup = cpu_time / mps_time
            print(f"{bs:<10} | {cpu_time:<10.2f} | {mps_time:<10.2f} | {speedup:<10.2f}x")
