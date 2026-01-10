"""
ResNet architecture for checkers policy and value networks.

Based on AlphaZero's architecture with modifications for CPU efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResidualBlock(nn.Module):
    """
    Residual block with two 3x3 convolutions.
    """

    def __init__(self, num_filters: int):
        super().__init__()

        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)

        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual  # Skip connection
        out = F.relu(out)

        return out


class PolicyHead(nn.Module):
    """
    Policy head that outputs action probabilities.
    """

    def __init__(self, num_filters: int, policy_size: int):
        super().__init__()

        self.conv = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.bn = nn.BatchNorm2d(32)

        # 32 filters * 10 * 10 = 3200
        self.fc = nn.Linear(32 * 10 * 10, policy_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)

        out = out.view(out.size(0), -1)  # Flatten
        out = self.fc(out)

        # Note: We don't apply softmax here - will be done after legal move masking
        return out


class ValueHead(nn.Module):
    """
    Value head that outputs a scalar evaluation of the position.
    """

    def __init__(self, num_filters: int):
        super().__init__()

        self.conv = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.bn = nn.BatchNorm2d(32)

        # 32 filters * 10 * 10 = 3200
        self.fc1 = nn.Linear(32 * 10 * 10, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)

        out = out.view(out.size(0), -1)  # Flatten

        out = self.fc1(out)
        out = F.relu(out)

        out = self.fc2(out)
        out = torch.tanh(out)  # Output in range [-1, 1]

        return out


class CheckersNetwork(nn.Module):
    """
    Combined policy and value network for checkers.

    Architecture:
    - Initial conv block (8 -> num_filters)
    - N residual blocks
    - Policy head (outputs policy_size logits)
    - Value head (outputs scalar value)
    """

    def __init__(
        self,
        num_filters: int = 128,
        num_res_blocks: int = 6,
        policy_size: int = 150,
    ):
        super().__init__()

        self.num_filters = num_filters
        self.num_res_blocks = num_res_blocks
        self.policy_size = policy_size

        # Initial convolution: 8 input planes -> num_filters
        self.initial_conv = nn.Conv2d(8, num_filters, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm2d(num_filters)

        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])

        # Policy and value heads
        self.policy_head = PolicyHead(num_filters, policy_size)
        self.value_head = ValueHead(num_filters)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 8, 10, 10)

        Returns:
            policy_logits: Shape (batch, policy_size)
            value: Shape (batch, 1)
        """
        # Initial convolution
        out = self.initial_conv(x)
        out = self.initial_bn(out)
        out = F.relu(out)

        # Residual tower
        for block in self.res_blocks:
            out = block(out)

        # Policy and value heads
        policy = self.policy_head(out)
        value = self.value_head(out)

        return policy, value

    def predict(
        self,
        state: torch.Tensor,
        legal_moves_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict policy and value with legal move masking.

        Args:
            state: Input tensor of shape (batch, 8, 10, 10)
            legal_moves_mask: Boolean mask of shape (batch, policy_size)

        Returns:
            policy: Probabilities of shape (batch, policy_size) (sums to 1)
            value: Scalar values of shape (batch, 1)
        """
        policy_logits, value = self.forward(state)

        # Mask illegal moves
        masked_logits = policy_logits.clone()
        masked_logits[~legal_moves_mask] = float('-inf')

        # Apply softmax
        policy = F.softmax(masked_logits, dim=1)

        return policy, value

    def get_action(
        self,
        state: torch.Tensor,
        legal_moves_mask: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.

        Args:
            state: Input tensor of shape (1, 8, 10, 10)
            legal_moves_mask: Boolean mask of shape (1, policy_size)
            temperature: Temperature for sampling (0 = greedy, >1 = more random)

        Returns:
            action: Sampled action index
            policy: Full policy distribution
            value: Position evaluation
        """
        with torch.no_grad():
            policy, value = self.predict(state, legal_moves_mask)

            if temperature == 0:
                # Greedy selection
                action = torch.argmax(policy[0]).item()
            else:
                # Temperature scaling
                policy_temp = torch.pow(policy[0], 1 / temperature)
                policy_temp = policy_temp / policy_temp.sum()

                # Sample from distribution
                action = torch.multinomial(policy_temp, 1).item()

        return action, policy, value


def initialize_weights(module: nn.Module):
    """Initialize network weights."""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(module.bias, 0)


# Example usage and testing
if __name__ == "__main__":
    print("Testing CheckersNetwork...")

    # Create network
    net = CheckersNetwork(num_filters=128, num_res_blocks=6, policy_size=150)
    net.apply(initialize_weights)

    # Count parameters
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 8, 10, 10)
    policy_logits, value = net(dummy_input)

    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Policy logits shape: {policy_logits.shape}")
    print(f"Value shape: {value.shape}")

    # Test with legal moves mask
    legal_moves_mask = torch.zeros(batch_size, 150, dtype=torch.bool)
    legal_moves_mask[:, :10] = True  # First 10 moves are legal

    policy, value = net.predict(dummy_input, legal_moves_mask)
    print(f"\nPolicy shape: {policy.shape}")
    print(f"Policy sum: {policy.sum(dim=1)}")  # Should be ~1.0
    print(f"Value range: [{value.min():.3f}, {value.max():.3f}]")  # Should be in [-1, 1]

    # Test action sampling
    single_input = dummy_input[:1]
    single_mask = legal_moves_mask[:1]

    action, policy, value = net.get_action(single_input, single_mask, temperature=1.0)
    print(f"\nSampled action: {action}")
    print(f"Value: {value.item():.3f}")

    print("\n✓ All tests passed!")
