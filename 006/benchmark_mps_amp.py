
import time
import torch
import numpy as np
from sac import SACAgent, ReplayBuffer

def benchmark_amp(batch_size=1024, n_steps=100):
    if not torch.backends.mps.is_available():
        print("MPS not available, skipping.")
        return

    device = torch.device('mps')
    print(f"\nBenchmarking AMP on MPS (Batch Size: {batch_size})...")
    
    state_dim = 71
    action_dim = 2
    
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
    
    # 1. Benchmark FP32 (Standard)
    print("Running FP32 (Standard)...")
    start_time = time.time()
    for _ in range(n_steps):
        agent.update(buffer, batch_size)
    torch.mps.synchronize()
    end_time = time.time()
    fp32_time = (end_time - start_time) / n_steps * 1000
    print(f"  FP32: {fp32_time:.3f} ms/step")
    
    # 2. Benchmark AMP
    print("Running AMP (Mixed Precision)...")
    scaler = torch.amp.GradScaler('mps')
    
    actor = agent.actor
    critic_1 = agent.critic_1
    critic_2 = agent.critic_2
    actor_optimizer = agent.actor_optimizer
    critic_1_optimizer = agent.critic_1_optimizer
    critic_2_optimizer = agent.critic_2_optimizer
    
    # Sample once to reuse
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    
    start_time = time.time()
    for i in range(n_steps):
        try:
            # CRITIC UPDATE
            with torch.amp.autocast(device_type='mps', dtype=torch.float16):
                with torch.no_grad():
                    next_actions_samp, next_log_probs = agent._sample_action(next_states)
                    target_q1 = agent.critic_target_1(next_states, next_actions_samp)
                    target_q2 = agent.critic_target_2(next_states, next_actions_samp)
                    target_q = torch.min(target_q1, target_q2) - agent.alpha.detach() * next_log_probs
                    target_q = rewards + (1 - dones) * agent.gamma * target_q
                
                current_q1 = critic_1(states, actions)
                current_q2 = critic_2(states, actions)
                critic_1_loss = torch.nn.functional.mse_loss(current_q1, target_q)
                critic_2_loss = torch.nn.functional.mse_loss(current_q2, target_q)
            
            critic_1_optimizer.zero_grad(set_to_none=True)
            scaler.scale(critic_1_loss).backward()
            scaler.step(critic_1_optimizer)
            
            critic_2_optimizer.zero_grad(set_to_none=True)
            scaler.scale(critic_2_loss).backward()
            scaler.step(critic_2_optimizer)
            
            scaler.update()
            
            # ACTOR UPDATE
            with torch.amp.autocast(device_type='mps', dtype=torch.float16):
                new_actions, log_probs = agent._sample_action(states)
                q1 = critic_1(states, new_actions)
                q2 = critic_2(states, new_actions)
                q = torch.min(q1, q2)
                actor_loss = (agent.alpha.detach() * log_probs - q).mean()
                
            actor_optimizer.zero_grad(set_to_none=True)
            scaler.scale(actor_loss).backward()
            scaler.step(actor_optimizer)
            scaler.update()
            
        except Exception as e:
            print(f"Error at step {i}: {e}")
            raise e
        
    torch.mps.synchronize()
    end_time = time.time()
    amp_time = (end_time - start_time) / n_steps * 1000
    print(f"  AMP:  {amp_time:.3f} ms/step")
    
    print(f"\nSpeedup: {fp32_time / amp_time:.2f}x")

if __name__ == "__main__":
    try:
        benchmark_amp()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
