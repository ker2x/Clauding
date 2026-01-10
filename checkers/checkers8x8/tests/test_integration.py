"""
Integration test for 8x8 Checkers system.

Tests the complete pipeline with minimal settings.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import numpy as np

from config8x8 import Config
from checkers8x8.engine.game import CheckersGame
from checkers8x8.engine.action_encoder import encode_action, decode_action, NUM_ACTIONS
from checkers8x8.network.resnet import CheckersNetwork
from checkers8x8.mcts.mcts import MCTS
from checkers8x8.training.replay_buffer import ReplayBuffer
from checkers8x8.training.self_play import SelfPlayGame


def test_game_engine():
    """Test game engine basics."""
    print("\n1. Testing Game Engine...")

    game = CheckersGame()
    assert not game.is_terminal(), "Initial position should not be terminal"

    legal_actions = game.get_legal_actions()
    assert len(legal_actions) > 0, "Initial position should have legal moves"

    # Make a move
    action = legal_actions[0]
    success = game.make_action(action)
    assert success, "Legal action should succeed"

    print(f"  ✓ Game engine works ({len(legal_actions)} legal moves)")


def test_action_encoding():
    """Test fixed action space encoding."""
    print("\n2. Testing Action Encoding...")

    # Test all actions encode/decode correctly
    for action in range(NUM_ACTIONS):
        from_square, direction = decode_action(action)
        reconstructed = encode_action(from_square, direction)
        assert action == reconstructed, f"Encoding mismatch at action {action}"

    print(f"  ✓ All {NUM_ACTIONS} actions encode/decode correctly")


def test_network():
    """Test network forward pass."""
    print("\n3. Testing Neural Network...")

    network = CheckersNetwork(num_filters=64, num_res_blocks=2, policy_size=128)

    # Test forward pass
    dummy_input = torch.randn(4, 8, 8, 8)
    policy_logits, value = network(dummy_input)

    assert policy_logits.shape == (4, 128), f"Policy shape mismatch: {policy_logits.shape}"
    assert value.shape == (4, 1), f"Value shape mismatch: {value.shape}"

    print(f"  ✓ Network forward pass works")


def test_mcts():
    """Test MCTS search."""
    print("\n4. Testing MCTS...")

    network = CheckersNetwork(num_filters=64, num_res_blocks=2, policy_size=128)
    network.eval()

    game = CheckersGame()

    mcts = MCTS(
        network=network,
        c_puct=1.0,
        num_simulations=10,  # Minimal for testing
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25
    )

    policy = mcts.search(game, add_noise=True)

    assert policy.shape == (128,), f"Policy shape mismatch: {policy.shape}"
    assert abs(policy.sum() - 1.0) < 0.01, f"Policy should sum to 1, got {policy.sum()}"

    print(f"  ✓ MCTS search works ({np.count_nonzero(policy)} actions)")


def test_self_play():
    """Test self-play game generation."""
    print("\n5. Testing Self-Play...")

    # Temporary minimal config
    class TestConfig:
        C_PUCT = 1.0
        MCTS_SIMS_SELFPLAY = 10
        DIRICHLET_ALPHA = 0.3
        DIRICHLET_EPSILON = 0.25
        TEMPERATURE_THRESHOLD = 5
        TEMPERATURE = 1.0
        MAX_GAME_LENGTH = 200

    network = CheckersNetwork(num_filters=64, num_res_blocks=2, policy_size=128)
    network.eval()

    self_play = SelfPlayGame(network, TestConfig, torch.device("cpu"))

    states, policies, values = self_play.play_game()

    assert len(states) > 0, "Should generate at least one state"
    assert len(states) == len(policies) == len(values), "Lengths should match"

    print(f"  ✓ Self-play works ({len(states)} examples generated)")


def test_replay_buffer():
    """Test replay buffer sampling."""
    print("\n6. Testing Replay Buffer...")

    buffer = ReplayBuffer(capacity=100, recency_tau=50.0)

    # Add some examples
    for i in range(50):
        state = np.random.randn(8, 8, 8).astype(np.float32)
        policy = np.random.rand(128).astype(np.float32)
        policy = policy / policy.sum()
        value = np.random.rand() * 2 - 1

        buffer.add(state, policy, value)

    # Sample batch
    states, policies, values = buffer.sample(batch_size=16)

    assert states.shape == (16, 8, 8, 8), f"States shape mismatch: {states.shape}"
    assert policies.shape == (16, 128), f"Policies shape mismatch: {policies.shape}"
    assert values.shape == (16,), f"Values shape mismatch: {values.shape}"

    print(f"  ✓ Replay buffer works ({len(buffer)} samples)")


def test_training_step():
    """Test one training step."""
    print("\n7. Testing Training Step...")

    network = CheckersNetwork(num_filters=64, num_res_blocks=2, policy_size=128)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

    # Create dummy batch
    states = torch.randn(16, 8, 8, 8)
    target_policies = torch.randn(16, 128)
    target_policies = torch.softmax(target_policies, dim=1)
    target_values = torch.randn(16, 1)

    # Forward pass
    pred_policies, pred_values = network(states)

    # Compute loss
    policy_loss = -torch.sum(target_policies * torch.log_softmax(pred_policies, dim=1)) / 16
    value_loss = torch.nn.functional.mse_loss(pred_values, target_values)
    total_loss = policy_loss + value_loss

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print(f"  ✓ Training step works (loss: {total_loss.item():.4f})")


def main():
    print("=" * 70)
    print("8x8 Checkers Integration Test")
    print("=" * 70)

    test_game_engine()
    test_action_encoding()
    test_network()
    test_mcts()
    test_self_play()
    test_replay_buffer()
    test_training_step()

    print("\n" + "=" * 70)
    print("✅ ALL INTEGRATION TESTS PASSED!")
    print("=" * 70)
    print("\nThe system is ready for full training!")


if __name__ == "__main__":
    main()
