"""
Integration test: full pipeline (game -> network -> MCTS -> self-play -> training).
"""

import sys
import os
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from going.engine.game import GoGame
from going.engine.action_encoder import NUM_ACTIONS, PASS_ACTION
from going.network.resnet import GoNetwork, count_parameters
from going.mcts.mcts import MCTS
from going.training.replay_buffer import ReplayBuffer
from going.training.self_play import SelfPlayGame
from config import Config


def test_game_engine():
    """Test game engine basics."""
    print("  Game engine...", end=" ", flush=True)

    game = GoGame()
    assert game.current_player == 1  # Black
    assert not game.is_terminal()

    legal = game.get_legal_actions()
    assert len(legal) == 82  # All 81 positions + pass
    assert PASS_ACTION in legal

    # Make a move
    game.make_action(40)  # Center
    assert game.current_player == 2  # White
    assert game.board[40] == 1  # Black stone

    # Make another
    game.make_action(41)
    assert game.current_player == 1
    assert game.board[41] == 2  # White stone

    print("OK")


def test_neural_input():
    """Test neural input shape and content."""
    print("  Neural input...", end=" ", flush=True)

    game = GoGame()
    state = game.to_neural_input()
    assert state.shape == (17, 9, 9)
    assert state.dtype == np.float32

    # Play some moves and check
    game.make_action(40)
    game.make_action(41)
    state2 = game.to_neural_input()
    assert state2.shape == (17, 9, 9)

    # Should have non-zero values now
    assert np.sum(state2[:16]) > 0

    print("OK")


def test_network():
    """Test neural network forward pass."""
    print("  Network...", end=" ", flush=True)

    net = GoNetwork(num_filters=32, num_res_blocks=2, policy_size=82, input_planes=17)
    params = count_parameters(net)
    assert params > 0
    print(f"({params:,} params) ", end="", flush=True)

    # Forward pass
    batch = torch.randn(4, 17, 9, 9)
    policy, value = net(batch)
    assert policy.shape == (4, 82)
    assert value.shape == (4, 1)

    # Prediction with masking
    state = torch.randn(17, 9, 9)
    legal = [0, 1, 40, 81]
    probs, val = net.predict(state, legal, torch.device("cpu"))
    assert probs.shape == (82,)
    assert abs(probs.sum() - 1.0) < 1e-5
    assert all(probs[i] > 0 for i in legal)
    assert all(probs[i] == 0 for i in range(82) if i not in legal)

    print("OK")


def test_mcts():
    """Test MCTS search."""
    print("  MCTS...", end=" ", flush=True)

    net = GoNetwork(num_filters=32, num_res_blocks=2, policy_size=82, input_planes=17)
    net.eval()

    game = GoGame()
    mcts = MCTS(
        network=net, c_puct=1.0, num_simulations=10,
        dirichlet_alpha=0.03, dirichlet_epsilon=0.25
    )

    policy = mcts.search(game, add_noise=True)
    assert policy.shape == (82,)
    assert abs(policy.sum() - 1.0) < 1e-5

    best = mcts.get_best_action()
    assert 0 <= best <= 81

    # Play the action
    game.make_action(best)
    assert not game.is_terminal() or game.consecutive_passes >= 2

    print("OK")


def test_replay_buffer():
    """Test replay buffer."""
    print("  Replay buffer...", end=" ", flush=True)

    buf = ReplayBuffer(capacity=100, recency_tau=50.0)
    assert len(buf) == 0

    # Add examples
    for i in range(50):
        state = np.random.randn(17, 9, 9).astype(np.float32)
        policy = np.random.rand(82).astype(np.float32)
        policy /= policy.sum()
        value = np.random.uniform(-1, 1)
        buf.add(state, policy, value)

    assert len(buf) == 50

    # Sample
    states, policies, values = buf.sample(16)
    assert states.shape == (16, 17, 9, 9)
    assert policies.shape == (16, 82)
    assert values.shape == (16,)

    # Test checkpoint
    sd = buf.state_dict()
    buf2 = ReplayBuffer(capacity=100)
    buf2.load_state_dict(sd)
    assert len(buf2) == 50

    print("OK")


def test_self_play():
    """Test self-play game generation."""
    print("  Self-play (1 game, 5 sims)...", end=" ", flush=True)

    net = GoNetwork(num_filters=32, num_res_blocks=2, policy_size=82, input_planes=17)
    net.eval()

    # Use minimal config for speed
    class TestConfig:
        C_PUCT = 1.0
        MCTS_SIMS_SELFPLAY = 5
        MCTS_BATCH_SIZE = 4
        DIRICHLET_ALPHA = 0.03
        DIRICHLET_EPSILON = 0.25
        TEMPERATURE_THRESHOLD = 10
        TEMPERATURE = 1.0
        MAX_GAME_LENGTH = 50
        KOMI = 7.5

    sp = SelfPlayGame(net, TestConfig, torch.device("cpu"))
    states, policies, values = sp.play_game()

    assert len(states) > 0
    assert len(states) == len(policies) == len(values)
    assert states[0].shape == (17, 9, 9)
    assert policies[0].shape == (82,)

    print(f"({len(states)} moves) OK")


def test_training_step():
    """Test a single training step."""
    print("  Training step...", end=" ", flush=True)

    net = GoNetwork(num_filters=32, num_res_blocks=2, policy_size=82, input_planes=17)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # Create fake batch
    states = torch.randn(8, 17, 9, 9)
    policies = torch.rand(8, 82)
    policies = policies / policies.sum(dim=1, keepdim=True)
    values = torch.randn(8, 1)

    # Forward
    net.train()
    pred_policy, pred_value = net(states)

    # Loss
    import torch.nn.functional as F
    policy_loss = -torch.sum(policies * F.log_softmax(pred_policy, dim=1)) / 8
    value_loss = F.mse_loss(pred_value, values)
    total_loss = policy_loss + value_loss

    # Backward
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    assert total_loss.item() > 0
    print(f"(loss={total_loss.item():.4f}) OK")


def test_gtp_engine():
    """Test GTP command handling."""
    print("  GTP engine...", end=" ", flush=True)

    from going.gtp.engine import GTPEngine

    net = GoNetwork(num_filters=32, num_res_blocks=2, policy_size=82, input_planes=17)
    net.eval()

    class TestConfig:
        C_PUCT = 1.0
        MCTS_SIMS_EVAL = 5
        MCTS_BATCH_SIZE = 4
        DIRICHLET_ALPHA = 0.03
        DIRICHLET_EPSILON = 0.0
        KOMI = 7.5

    engine = GTPEngine(net, TestConfig, torch.device("cpu"))

    # Test basic commands
    ok, resp = engine.handle_command("protocol_version")
    assert ok and resp == "2"

    ok, resp = engine.handle_command("name")
    assert ok and resp == "going"

    ok, resp = engine.handle_command("boardsize 9")
    assert ok

    ok, resp = engine.handle_command("clear_board")
    assert ok

    ok, resp = engine.handle_command("komi 6.5")
    assert ok

    ok, resp = engine.handle_command("play black E5")
    assert ok

    ok, resp = engine.handle_command("genmove white")
    assert ok
    assert resp != ""

    ok, resp = engine.handle_command("showboard")
    assert ok

    ok, resp = engine.handle_command("known_command play")
    assert ok and resp == "true"

    ok, resp = engine.handle_command("list_commands")
    assert ok

    print("OK")


def main():
    print("=" * 60)
    print("Go Integration Tests")
    print("=" * 60)

    test_game_engine()
    test_neural_input()
    test_network()
    test_mcts()
    test_replay_buffer()
    test_self_play()
    test_training_step()
    test_gtp_engine()

    print("\n" + "=" * 60)
    print("All integration tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
