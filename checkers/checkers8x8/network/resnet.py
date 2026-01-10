"""
ResNet architecture for 8x8 Checkers with Fixed Action Space.

Input: (batch, 8, 8, 8) - 8 feature planes
Output: (batch, 128) policy logits, (batch, 1) value
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with batch normalization."""

    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # Skip connection
        out = F.relu(out)
        return out


class CheckersNetwork(nn.Module):
    """
    ResNet for 8x8 Checkers with Fixed Action Space.

    Architecture:
        Input (8, 8, 8)
        → Initial Conv (128 filters)
        → N Residual Blocks
        → Policy Head (128 actions)
        → Value Head (1 value)
    """

    def __init__(self, num_filters: int = 128, num_res_blocks: int = 6, policy_size: int = 128):
        super().__init__()

        self.num_filters = num_filters
        self.num_res_blocks = num_res_blocks
        self.policy_size = policy_size

        # Initial convolution (8 input planes → num_filters)
        self.initial_conv = nn.Conv2d(8, num_filters, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm2d(num_filters)

        # Residual tower
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])

        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, policy_size)

        # Value head
        self.value_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 8, 8, 8)

        Returns:
            (policy_logits, value): Policy logits (batch, 128), value (batch, 1)
        """
        # Initial convolution
        out = self.initial_conv(x)
        out = self.initial_bn(out)
        out = F.relu(out)

        # Residual tower
        for block in self.residual_blocks:
            out = block(out)

        # Policy head
        policy = self.policy_conv(out)
        policy = self.policy_bn(policy)
        policy = F.relu(policy)
        policy = policy.view(policy.size(0), -1)  # Flatten
        policy = self.policy_fc(policy)

        # Value head
        value = self.value_conv(out)
        value = self.value_bn(value)
        value = F.relu(value)
        value = value.view(value.size(0), -1)  # Flatten
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # Output in [-1, 1]

        return policy, value

    def predict(self, state, legal_actions, device):
        """
        Predict policy and value for a state with legal action masking.

        Args:
            state: Game state tensor (8, 8, 8) or (batch, 8, 8, 8)
            legal_actions: List of legal action indices
            device: PyTorch device

        Returns:
            (policy_probs, value): Softmaxed policy (128,), value scalar
        """
        # Ensure batch dimension
        if state.ndim == 3:
            state = state.unsqueeze(0)

        state = state.to(device)

        with torch.no_grad():
            policy_logits, value = self.forward(state)

            # Mask illegal actions
            mask = torch.full((1, self.policy_size), float('-inf'), device=device)
            for action in legal_actions:
                mask[0, action] = 0.0

            masked_logits = policy_logits + mask

            # Softmax to get probabilities
            policy_probs = F.softmax(masked_logits, dim=1)[0].cpu().numpy()
            value = value[0, 0].item()

        return policy_probs, value


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Testing
if __name__ == "__main__":
    print("Testing 8x8 Checkers ResNet")
    print("=" * 60)

    # Create network
    net = CheckersNetwork(num_filters=128, num_res_blocks=6, policy_size=128)
    print(f"Network created")
    print(f"Parameters: {count_parameters(net):,}")

    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 8, 8, 8)

    policy_logits, value = net(dummy_input)

    print(f"\nForward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Policy shape: {policy_logits.shape}")
    print(f"  Value shape: {value.shape}")

    assert policy_logits.shape == (batch_size, 128), f"Policy shape mismatch"
    assert value.shape == (batch_size, 1), f"Value shape mismatch"

    # Test prediction with legal actions
    print(f"\nPrediction with legal actions:")
    single_state = torch.randn(8, 8, 8)
    legal_actions = [0, 1, 5, 10, 20]

    policy_probs, value_pred = net.predict(single_state, legal_actions, torch.device("cpu"))

    print(f"  Policy probs shape: {policy_probs.shape}")
    print(f"  Policy probs sum: {policy_probs.sum():.4f}")
    print(f"  Value: {value_pred:.4f}")
    print(f"  Non-zero policy entries: {(policy_probs > 0).sum()}")

    # Check that only legal actions have non-zero probability
    for i in range(128):
        if i in legal_actions:
            assert policy_probs[i] > 0, f"Legal action {i} has zero prob"
        else:
            assert policy_probs[i] == 0, f"Illegal action {i} has non-zero prob"

    print("\n✓ Network tests passed!")
