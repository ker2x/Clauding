"""
Test device switching between CPU (self-play) and MPS (training).
"""

import torch
from config import Config
from checkers.network.resnet import CheckersNetwork

print("=" * 60)
print("Device Switching Test")
print("=" * 60)

# Show configuration
print(f"\nConfiguration:")
print(f"  Training device: {Config.DEVICE}")
print(f"  Self-play device: {Config.SELFPLAY_DEVICE}")

# Get devices
training_device = Config.get_device()
selfplay_device = Config.get_selfplay_device()

print(f"\nResolved devices:")
print(f"  Training: {training_device}")
print(f"  Self-play: {selfplay_device}")

# Create network
print(f"\nCreating network...")
network = CheckersNetwork(
    num_filters=Config.NUM_FILTERS,
    num_res_blocks=Config.NUM_RES_BLOCKS,
    policy_size=Config.POLICY_SIZE
)

# Test device switching
print(f"\n1. Initial state: CPU")
print(f"   Network is on: {next(network.parameters()).device}")

print(f"\n2. Moving to training device ({training_device})...")
network.to(training_device)
print(f"   Network is on: {next(network.parameters()).device}")

print(f"\n3. Moving to self-play device ({selfplay_device})...")
network.to(selfplay_device)
print(f"   Network is on: {next(network.parameters()).device}")

print(f"\n4. Moving back to training device ({training_device})...")
network.to(training_device)
print(f"   Network is on: {next(network.parameters()).device}")

# Test forward pass on each device
print(f"\n5. Testing forward pass on self-play device...")
network.to(selfplay_device)
network.eval()
dummy_input = torch.randn(1, 8, 10, 10).to(selfplay_device)
with torch.no_grad():
    policy, value = network(dummy_input)
print(f"   ✓ Forward pass works on {selfplay_device}")
print(f"   Output shapes: policy={policy.shape}, value={value.shape}")

print(f"\n6. Testing forward pass on training device...")
network.to(training_device)
dummy_input = dummy_input.to(training_device)
with torch.no_grad():
    policy, value = network(dummy_input)
print(f"   ✓ Forward pass works on {training_device}")
print(f"   Output shapes: policy={policy.shape}, value={value.shape}")

print("\n" + "=" * 60)
print("✓ Device switching test passed!")
print("=" * 60)

print("\nSummary:")
print("  • Self-play will use CPU (faster for MCTS with 100 simulations)")
print("  • Training will use MPS (faster for large batch gradients)")
print("  • Network seamlessly switches between devices during training loop")
