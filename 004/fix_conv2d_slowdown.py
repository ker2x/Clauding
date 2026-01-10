"""
Potential fixes for intermittent torch.conv2d slowdown in visual mode.

This script contains various strategies to diagnose and fix the issue.
Try these fixes one at a time to see which one helps.
"""

import torch

# ============================================================================
# FIX 1: Disable CuDNN autotuner (most common cause of intermittent slowdowns)
# ============================================================================
# The CuDNN autotuner tries different convolution algorithms and picks the fastest.
# However, this benchmarking can cause intermittent slowdowns.
# Add this at the top of train.py:

def fix_1_disable_cudnn_benchmark():
    """Disable CuDNN benchmark to prevent algorithm retuning."""
    torch.backends.cudnn.benchmark = False
    print("✓ CuDNN benchmark disabled")


# ============================================================================
# FIX 2: Force deterministic operations
# ============================================================================
# Non-deterministic operations can sometimes cause performance variability.
# Add this at the top of train.py:

def fix_2_force_deterministic():
    """Force deterministic operations."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("✓ Deterministic mode enabled")


# ============================================================================
# FIX 3: Clear GPU cache periodically
# ============================================================================
# GPU memory fragmentation can cause slowdowns.
# Add this in the training loop (e.g., every 100 episodes):

def fix_3_clear_cache():
    """Clear GPU cache to prevent memory fragmentation."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ CUDA cache cleared")
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        print("✓ MPS cache cleared")


# ============================================================================
# FIX 4: Pre-compile networks with dummy forward pass
# ============================================================================
# First forward pass through network can trigger JIT compilation.
# Add this after creating the agent in train.py:

def fix_4_warmup_networks(agent, state_shape, device):
    """Warm up networks with dummy forward passes."""
    agent.actor.eval()
    agent.critic_1.eval()
    agent.critic_2.eval()

    with torch.no_grad():
        dummy_state = torch.randn(1, *state_shape).to(device)
        dummy_action = torch.randn(1, agent.action_dim).to(device)

        # Warm up actor
        _ = agent.actor(dummy_state)

        # Warm up critics
        _ = agent.critic_1(dummy_state, dummy_action)
        _ = agent.critic_2(dummy_state, dummy_action)

        # Synchronize to ensure compilation is done
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()

    agent.actor.train()
    agent.critic_1.train()
    agent.critic_2.train()

    print("✓ Networks warmed up")


# ============================================================================
# FIX 5: Pin memory for data loading
# ============================================================================
# This speeds up CPU->GPU transfers but doesn't directly fix conv2d.
# Modify ReplayBuffer.sample() to use pinned memory (already on GPU).


# ============================================================================
# FIX 6: Use channels_last memory format
# ============================================================================
# channels_last can be faster for convolutions on some hardware.
# Add this after creating networks:

def fix_6_channels_last(agent):
    """Convert networks to channels_last memory format."""
    if hasattr(agent.actor, 'conv1'):  # Visual mode
        agent.actor = agent.actor.to(memory_format=torch.channels_last)
        agent.critic_1 = agent.critic_1.to(memory_format=torch.channels_last)
        agent.critic_2 = agent.critic_2.to(memory_format=torch.channels_last)
        agent.critic_target_1 = agent.critic_target_1.to(memory_format=torch.channels_last)
        agent.critic_target_2 = agent.critic_target_2.to(memory_format=torch.channels_last)
        print("✓ Networks converted to channels_last format")


# ============================================================================
# FIX 7: Use torch.compile() (PyTorch 2.0+)
# ============================================================================
# torch.compile can significantly speed up models.
# Add this after creating networks (requires PyTorch 2.0+):

def fix_7_compile_networks(agent):
    """Compile networks with torch.compile for better performance."""
    if hasattr(torch, 'compile'):
        agent.actor = torch.compile(agent.actor)
        agent.critic_1 = torch.compile(agent.critic_1)
        agent.critic_2 = torch.compile(agent.critic_2)
        print("✓ Networks compiled with torch.compile")
    else:
        print("✗ torch.compile not available (requires PyTorch 2.0+)")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def apply_all_fixes(agent, state_shape, device):
    """Apply all fixes to potentially resolve slowdown issue."""
    print("\nApplying fixes for intermittent conv2d slowdown...")
    print("="*70)

    # Fix 1: Disable CuDNN benchmark (most important!)
    fix_1_disable_cudnn_benchmark()

    # Fix 2: Force deterministic
    # fix_2_force_deterministic()  # Optional: can reduce performance slightly

    # Fix 3: Clear cache (call this periodically in training loop)
    fix_3_clear_cache()

    # Fix 4: Warm up networks
    fix_4_warmup_networks(agent, state_shape, device)

    # Fix 6: Use channels_last (optional, may help on some GPUs)
    # fix_6_channels_last(agent)

    # Fix 7: Compile networks (optional, PyTorch 2.0+ only)
    # fix_7_compile_networks(agent)

    print("="*70)
    print("Fixes applied! Monitor training to see if slowdowns persist.\n")


if __name__ == "__main__":
    print(__doc__)
    print("\nTo use these fixes:")
    print("1. Import this module in train.py:")
    print("   from fix_conv2d_slowdown import apply_all_fixes, fix_3_clear_cache")
    print("\n2. After creating agent, apply fixes:")
    print("   apply_all_fixes(agent, state_shape, device)")
    print("\n3. In training loop, periodically clear cache:")
    print("   if (episode + 1) % 100 == 0:")
    print("       fix_3_clear_cache()")
    print("\nMost important fix: Disable CuDNN benchmark (Fix 1)")
