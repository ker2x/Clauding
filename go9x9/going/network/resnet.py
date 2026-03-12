"""
ResNet architecture for 9x9 Go.

Input: (batch, INPUT_PLANES, 9, 9)
Output: (batch, 82) policy logits, (batch, 1) value, (batch, 81) ownership
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
        out += residual
        out = F.relu(out)
        return out


class GlobalPoolBias(nn.Module):
    """
    Injects board-wide context into the residual stream.

    Global average pools the feature map to a vector, projects it,
    then broadcasts back as a per-channel additive bias.
    This lets the network see territory imbalance, group count, etc.
    """

    def __init__(self, num_filters: int):
        super().__init__()
        self.fc = nn.Linear(num_filters, num_filters)

    def forward(self, x):
        pooled = x.mean(dim=[2, 3])              # (B, C)
        bias = self.fc(F.relu(pooled))           # (B, C)
        return x + bias.unsqueeze(-1).unsqueeze(-1)


class GoNetwork(nn.Module):
    """
    ResNet for 9x9 Go with:
        - Global-pool-bias layers every global_pool_freq residual blocks
        - Ownership auxiliary head (training only)

    Architecture:
        Input (INPUT_PLANES, 9, 9)
        -> Initial Conv (num_filters)
        -> Residual tower (num_res_blocks blocks + GlobalPoolBias layers)
        -> Policy Head (82 actions)
        -> Value Head (1 value in [-1, 1])
        -> Ownership Head (81 intersections, sigmoid)
    """

    def __init__(self, num_filters: int = 128, num_res_blocks: int = 6,
                 policy_size: int = 82, input_planes: int = 5,  # 2 history × 2 + 1 color
                 global_pool_freq: int = 3):
        super().__init__()

        self.num_filters = num_filters
        self.num_res_blocks = num_res_blocks
        self.policy_size = policy_size

        # Initial convolution
        self.initial_conv = nn.Conv2d(input_planes, num_filters, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm2d(num_filters)

        # Residual tower with interleaved global-pool-bias layers
        self.tower = nn.ModuleList()
        for i in range(num_res_blocks):
            self.tower.append(ResidualBlock(num_filters))
            if (i + 1) % global_pool_freq == 0:
                self.tower.append(GlobalPoolBias(num_filters))

        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 9 * 9, policy_size)

        # Value head
        self.value_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 9 * 9, 128)
        self.value_fc2 = nn.Linear(128, 1)

        # Ownership auxiliary head
        self.ownership_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.ownership_bn = nn.BatchNorm2d(32)
        self.ownership_fc = nn.Linear(32 * 9 * 9, 81)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, INPUT_PLANES, 9, 9)

        Returns:
            (policy_logits, value, ownership):
                policy_logits: (batch, 82)
                value:         (batch, 1)
                ownership:     (batch, 81) sigmoid probabilities (current player's perspective)
        """
        out = self.initial_conv(x)
        out = self.initial_bn(out)
        out = F.relu(out)

        for module in self.tower:
            out = module(out)

        # Policy head
        policy = self.policy_conv(out)
        policy = self.policy_bn(policy)
        policy = F.relu(policy)
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)

        # Value head
        value = self.value_conv(out)
        value = self.value_bn(value)
        value = F.relu(value)
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        # Ownership head
        ownership = self.ownership_conv(out)
        ownership = self.ownership_bn(ownership)
        ownership = F.relu(ownership)
        ownership = ownership.view(ownership.size(0), -1)
        ownership = torch.sigmoid(self.ownership_fc(ownership))

        return policy, value, ownership

    def predict(self, state, legal_actions, device):
        """
        Predict policy and value for a state with legal action masking.

        Args:
            state: Game state tensor (INPUT_PLANES, 9, 9) or (batch, INPUT_PLANES, 9, 9)
            legal_actions: List of legal action indices
            device: PyTorch device

        Returns:
            (policy_probs, value): Softmaxed policy (82,), value scalar
        """
        if state.ndim == 3:
            state = state.unsqueeze(0)

        state = state.to(device)

        with torch.no_grad():
            policy_logits, value, _ = self.forward(state)

            mask = torch.full((1, self.policy_size), float('-inf'), device=device)
            for action in legal_actions:
                mask[0, action] = 0.0

            masked_logits = policy_logits + mask

            policy_probs = F.softmax(masked_logits, dim=1)[0].cpu().numpy()
            value = value[0, 0].item()

        return policy_probs, value


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
