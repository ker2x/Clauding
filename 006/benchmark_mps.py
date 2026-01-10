
import time
import torch
import numpy as np
from sac import SACAgent, ReplayBuffer
from training_utils import get_device

def benchmark_device(device_name, batch_size=256, n_steps=100):
    print(f"\nBenchmarking {device_name}...")
    device = torch.device(device_name)
    
    state_dim = 71
    action_dim = 2
    
    # Initialize agent
    agent = SACAgent(state_dim, action_dim, device=device)
    
    # Initialize buffer
    buffer = ReplayBuffer(capacity=2000, state_shape=state_dim, action_dim=action_dim, device=device)
    
    # Fill buffer with dummy data
    for _ in range(batch_size * 2):
        state = np.random.randn(state_dim).astype(np.float32)
        action = np.random.randn(action_dim).astype(np.float32)
        next_state = np.random.randn(state_dim).astype(np.float32)
        buffer.push(state, action, 0.0, next_state, False, False)
        
    # Warmup
    dummy_state = np.random.randn(state_dim).astype(np.float32)
    agent.select_action(dummy_state)
    agent.update(buffer, batch_size)
    
    # Benchmark select_action (Rollout)
    start_time = time.time()
    for _ in range(n_steps):
        agent.select_action(dummy_state)
    if device_name == 'mps':
        torch.mps.synchronize()
    elif device_name == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    avg_inference = (end_time - start_time) / n_steps * 1000
    print(f"  Select Action (Inference): {avg_inference:.3f} ms/step")
    
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
    print(f"  Update (Training): {avg_train:.3f} ms/step")
    
    return avg_inference, avg_train

if __name__ == "__main__":
    print(f"MPS Available: {torch.backends.mps.is_available()}")
    
    cpu_inf, cpu_train = benchmark_device('cpu')
    
    if torch.backends.mps.is_available():
        mps_inf, mps_train = benchmark_device('mps')
        
        print("\nComparison (MPS vs CPU):")
        print(f"  Inference Speedup: {cpu_inf / mps_inf:.2f}x (Higher is better)")
        print(f"  Training Speedup:  {cpu_train / mps_train:.2f}x (Higher is better)")
        
        if mps_inf > cpu_inf:
            print("\nObservation: MPS inference is SLOWER than CPU (expected for small batches).")
        else:
            print("\nObservation: MPS inference is FASTER than CPU.")
            
        if mps_train < cpu_train:
            print("Observation: MPS training is FASTER than CPU (expected for large batches).")
        else:
            print("Observation: MPS training is SLOWER than CPU.")
