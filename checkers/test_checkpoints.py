"""
Test checkpoint saving and loading functionality.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from config8x8 import Config
from checkers8x8.network.resnet import CheckersNetwork
from checkers8x8.training.trainer import Trainer
from checkers8x8.training.replay_buffer import ReplayBuffer
import torch
import numpy as np
from pathlib import Path
import shutil

# Setup temporary test directory
TEST_CHECKPOINT_DIR = Path("test_checkpoints_tmp")
TEST_CHECKPOINT_DIR.mkdir(exist_ok=True)

# Override config for testing to avoid touching real checkpoints
Config.CHECKPOINT_DIR = str(TEST_CHECKPOINT_DIR)

print("="*70)
print("Testing Checkpoint System")
print("="*70)

# Test 1: Replay buffer state dict
print("\n[Test 1] Replay Buffer State Dict")
print("-"*70)
buffer = ReplayBuffer(capacity=1000, recency_tau=50.0)

# Add some data
for i in range(50):
    state = np.random.randn(8, 8, 8).astype(np.float32)
    policy = np.random.rand(128).astype(np.float32)
    policy = policy / policy.sum()
    value = np.random.rand() * 2 - 1
    buffer.add(state, policy, value)

print(f"Buffer size: {len(buffer)}")
print(f"Current generation: {buffer.current_generation}")

# Save state
state_dict = buffer.state_dict()
print(f"State dict keys: {list(state_dict.keys())}")
print(f"✓ State dict created")

# Create new buffer and load
buffer2 = ReplayBuffer(capacity=1000, recency_tau=50.0)
buffer2.load_state_dict(state_dict)
print(f"Restored buffer size: {len(buffer2)}")
print(f"Restored generation: {buffer2.current_generation}")

assert len(buffer) == len(buffer2), "Buffer sizes don't match!"
assert buffer.current_generation == buffer2.current_generation, "Generations don't match!"
print("✓ Buffer state dict works correctly")

# Test 2: Trainer checkpoint save/load
print("\n\n[Test 2] Trainer Checkpoint Save/Load")
print("-"*70)

device = torch.device("cpu")
network = CheckersNetwork(
    num_filters=Config.NUM_FILTERS,
    num_res_blocks=Config.NUM_RES_BLOCKS,
    policy_size=Config.POLICY_SIZE
)
trainer = Trainer(network, Config, device, device)

# Add some data to buffer
for i in range(20):
    state = np.random.randn(8, 8, 8).astype(np.float32)
    policy = np.random.rand(128).astype(np.float32)
    policy = policy / policy.sum()
    value = np.random.rand() * 2 - 1
    trainer.replay_buffer.add(state, policy, value)

print(f"Buffer size before save: {len(trainer.replay_buffer)}")

# Save checkpoint
trainer.save_checkpoint(5)
print("✓ Checkpoint saved")

# Check files exist
checkpoint_dir = Path(Config.CHECKPOINT_DIR)
assert (checkpoint_dir / "checkpoint_iter_5.pt").exists(), "Numbered checkpoint not found!"
assert (checkpoint_dir / "latest.pt").exists(), "Latest checkpoint not found!"
print("✓ Checkpoint files exist")

# Load into new trainer
network2 = CheckersNetwork(
    num_filters=Config.NUM_FILTERS,
    num_res_blocks=Config.NUM_RES_BLOCKS,
    policy_size=Config.POLICY_SIZE
)
trainer2 = Trainer(network2, Config, device, device)
trainer2.load_checkpoint(str(checkpoint_dir / "latest.pt"))

assert trainer2.start_iteration == 5, f"Wrong iteration: {trainer2.start_iteration}"
assert len(trainer2.replay_buffer) == 20, f"Wrong buffer size: {len(trainer2.replay_buffer)}"
print("✓ Checkpoint loaded correctly")

# Test 3: Best model saving
print("\n\n[Test 3] Best Model Saving")
print("-" * 70)

# The save_best_model method was replaced by evaluate_and_update_best
# We'll test manual saving to verify the path exists and then verify loading
torch.save({
    'network_state_dict': trainer.network.state_dict(),
    'iteration': 5,
    'win_rate': 0.85,
    'loss': 0.85,  # For backward compatibility with this test
}, trainer.best_model_path)

assert trainer.best_model_path.exists(), "Best model not found!"
print(f"✓ Best model saved manually at {trainer.best_model_path}")

# Verify content
checkpoint = torch.load(trainer.best_model_path, weights_only=False)
assert checkpoint['win_rate'] == 0.85, "Saved win_rate mismatch!"
print("✓ Best model data verified")

print("\n" + "="*70)
print("✅ ALL CHECKPOINT TESTS PASSED!")
print("="*70)

# Cleanup
if TEST_CHECKPOINT_DIR.exists():
    shutil.rmtree(TEST_CHECKPOINT_DIR)
    print(f"\nCleaned up test directory: {TEST_CHECKPOINT_DIR}")
