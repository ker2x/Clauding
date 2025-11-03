"""
Double Deep Q-Network (DDQN) Agent for CarRacing-v3.

DDQN improves upon DQN by addressing the overestimation bias in Q-value updates.

Standard DQN:
    Q_target = r + γ * max_a' Q_target(s', a')
    Problem: The same network both selects and evaluates actions, leading to
    optimistic (overestimated) Q-values.

Double DQN:
    Q_target = r + γ * Q_target(s', argmax_a' Q_policy(s', a'))
    Solution: Use policy network to SELECT the best action, target network to
    EVALUATE that action. This decorrelation reduces overestimation.

Key components:
1. Policy Network: Actively trained, used for action selection
2. Target Network: Frozen copy updated periodically, provides stable targets
3. Replay Buffer: Stores experiences for training
4. Epsilon-Greedy: Balances exploration vs exploitation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from typing import Tuple, Optional


# Experience tuple stored in replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DQN(nn.Module):
    """
    Deep Q-Network for processing stacked frames and outputting Q-values.

    Architecture:
        Input: (batch, 4, 96, 96) - 4 stacked grayscale frames (native CarRacing resolution)
        Conv layers: Extract spatial features
        FC layers: Combine features and output Q-values
        Output: (batch, n_actions) - Q-value for each action
    """

    def __init__(self, input_channels: int, n_actions: int):
        """
        Args:
            input_channels: Number of input channels (4 for stacked frames)
            n_actions: Number of discrete actions
        """
        super(DQN, self).__init__()

        # Convolutional layers (same as DQN Nature paper)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate size after conv layers
        # Input: 96×96 (native CarRacing resolution)
        # After conv1: (96-8)/4+1 = 23
        # After conv2: (23-4)/2+1 = 10
        # After conv3: (10-3)/1+1 = 8
        # Final: 64 channels × 8×8 = 4096
        conv_output_size = 64 * 8 * 8

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, n_actions)

        # Activation
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through network.

        Args:
            x: Input tensor (batch, 4, 96, 96)

        Returns:
            Q-values for each action (batch, n_actions)
        """
        # Convolutional layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class VectorDQN(nn.Module):
    """
    Deep Q-Network for processing vector state observations.

    Architecture:
        Input: (batch, 11) - vector state [x, y, vx, vy, angle, angular_vel, wheel_contacts[4], track_progress]
        FC layers: Process vector and output Q-values
        Output: (batch, n_actions) - Q-value for each action

    Much simpler and faster than CNN-based DQN.
    """

    def __init__(self, input_size: int, n_actions: int):
        """
        Args:
            input_size: Size of input vector (11 for CarRacing vector state)
            n_actions: Number of discrete actions
        """
        super(VectorDQN, self).__init__()

        # Fully connected layers
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, n_actions)

        # Activation
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through network.

        Args:
            x: Input tensor (batch, 11)

        Returns:
            Q-values for each action (batch, n_actions)
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)

        return x


class ReplayBuffer:
    """
    Experience Replay Buffer for storing and sampling experiences.

    The replay buffer is crucial for:
    1. Breaking temporal correlation between consecutive experiences
    2. Reusing experiences multiple times for data efficiency
    3. Stabilizing training by smoothing the distribution of experiences

    Implemented as a deque (double-ended queue) with fixed maximum size.
    When full, oldest experiences are automatically removed (FIFO).
    """

    def __init__(self, capacity: int):
        """
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Store an experience in the buffer.

        Args:
            state: Current state (4, 84, 84)
            action: Action taken
            reward: Reward received
            next_state: Next state (4, 84, 84)
            done: Whether episode ended
        """
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a random batch of experiences.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        experiences = random.sample(self.buffer, batch_size)

        # Unzip experiences into separate arrays
        states = np.array([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.array([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return current size of buffer."""
        return len(self.buffer)


class DDQNAgent:
    """
    Double Deep Q-Network Agent.

    Implements DDQN algorithm with epsilon-greedy exploration and experience replay.
    """

    def __init__(
        self,
        state_shape: Tuple,
        n_actions: int,
        learning_rate: float = 0.00025,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 1_000_000,
        buffer_size: int = 100_000,
        batch_size: int = 32,
        target_update_freq: int = 10_000,
        device: str = 'auto',
        state_mode: str = 'visual'
    ):
        """
        Initialize DDQN agent.

        Args:
            state_shape: Shape of state observation
                         - Visual/Synthetic: (channels, height, width) e.g., (4, 96, 96)
                         - Vector: (size,) e.g., (11,)
            n_actions: Number of discrete actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay_steps: Steps over which to decay epsilon
            buffer_size: Replay buffer capacity
            batch_size: Batch size for training
            target_update_freq: Steps between target network updates
            device: Device to use ('auto', 'cuda', 'mps', or 'cpu')
            state_mode: 'visual', 'synthetic' (both use CNN), or 'vector' (uses MLP)
        """
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.state_mode = state_mode

        # Device selection
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")
        print(f"Using state mode: {state_mode}")

        # Networks (choose based on state mode)
        if state_mode == 'vector':
            input_size = state_shape[0]  # e.g., 11 for CarRacing vector state
            self.policy_net = VectorDQN(input_size, n_actions).to(self.device)
            self.target_net = VectorDQN(input_size, n_actions).to(self.device)
        else:
            # Visual or synthetic mode (both use CNN for spatial features)
            input_channels = state_shape[0]  # e.g., 4 for stacked frames
            self.policy_net = DQN(input_channels, n_actions).to(self.device)
            self.target_net = DQN(input_channels, n_actions).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is always in eval mode

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.initial_learning_rate = learning_rate

        # Learning rate scheduler (reduces LR when training plateaus)
        # ReduceLROnPlateau: reduce LR by factor of 0.5 if no improvement for 100 episodes
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=100
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Training state
        self.steps_done = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state (channels, height, width)
            training: Whether in training mode (uses epsilon-greedy) or
                     evaluation mode (uses greedy policy)

        Returns:
            Action index
        """
        # During evaluation, always use greedy policy
        if not training:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device, dtype=torch.float32)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(dim=1).item()

        # Epsilon-greedy during training
        if random.random() < self.epsilon:
            # Explore: random action
            return random.randrange(self.n_actions)
        else:
            # Exploit: best action according to policy network
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device, dtype=torch.float32)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(dim=1).item()

    def update_epsilon(self):
        """
        Decay epsilon linearly based on steps.

        Epsilon decays from epsilon_start to epsilon_end over epsilon_decay_steps.
        """
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end)
            * self.steps_done / self.epsilon_decay_steps
        )

    def update_learning_rate(self, metric: float):
        """
        Update learning rate based on performance metric.

        Args:
            metric: Performance metric (e.g., average evaluation reward)
        """
        self.scheduler.step(metric)

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        """
        Perform one training step using Double DQN algorithm.

        Returns:
            Loss value if training occurred, None if buffer is too small
        """
        # Check if enough experiences in buffer
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors (using from_numpy for zero-copy when possible)
        states = torch.from_numpy(states).to(self.device, dtype=torch.float32)
        actions = torch.from_numpy(actions).to(self.device, dtype=torch.long)
        rewards = torch.from_numpy(rewards).to(self.device, dtype=torch.float32)
        next_states = torch.from_numpy(next_states).to(self.device, dtype=torch.float32)
        dones = torch.from_numpy(dones).to(self.device, dtype=torch.float32)

        # Current Q-values from policy network
        # Shape: (batch_size, n_actions)
        current_q_values = self.policy_net(states)
        # Select Q-values for taken actions
        # Shape: (batch_size,)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: Use policy network to SELECT action, target network to EVALUATE
        with torch.no_grad():
            # Policy network selects best action for next state
            next_q_values_policy = self.policy_net(next_states)
            next_actions = next_q_values_policy.argmax(dim=1)

            # Target network evaluates the selected action
            next_q_values_target = self.target_net(next_states)
            next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            # Compute target Q-values using Bellman equation
            # If done, target is just the reward (no future value)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss (MSE between current and target Q-values)
        loss = nn.functional.mse_loss(current_q_values, target_q_values)

        # Optimize the policy network
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update target network periodically
        # Note: steps_done is incremented in training loop, not here
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"Target network updated at step {self.steps_done}")

        return loss.item()

    def get_learning_rate(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def save(self, filepath: str):
        """
        Save agent state to file.

        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'steps_done': self.steps_done,
            'epsilon': self.epsilon,
            'state_mode': self.state_mode,
        }
        torch.save(checkpoint, filepath)
        print(f"Agent saved to {filepath}")

    def load(self, filepath: str, load_optimizer: bool = True):
        """
        Load agent state from file.

        Args:
            filepath: Path to checkpoint
            load_optimizer: Whether to load optimizer state (set False for evaluation)
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])

        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if load_optimizer and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.steps_done = checkpoint.get('steps_done', 0)
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)

        print(f"Agent loaded from {filepath}")
        print(f"Steps done: {self.steps_done}, Epsilon: {self.epsilon:.4f}")
        print(f"Learning rate: {self.get_learning_rate():.6f}")


if __name__ == "__main__":
    """Test DDQN agent initialization."""
    print("Testing DDQN agent...")

    # Create agent with CarRacing state shape
    state_shape = (4, 84, 84)  # 4 stacked frames
    n_actions = 9  # 3 steering × 3 gas/brake

    agent = DDQNAgent(
        state_shape=state_shape,
        n_actions=n_actions,
        learning_rate=0.00025,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_steps=1_000_000,
        buffer_size=100_000,
        batch_size=32,
        target_update_freq=10_000
    )

    print(f"\nPolicy Network:")
    print(agent.policy_net)

    print(f"\nAgent configuration:")
    print(f"State shape: {state_shape}")
    print(f"Number of actions: {n_actions}")
    print(f"Device: {agent.device}")
    print(f"Replay buffer capacity: {len(agent.replay_buffer.buffer)}")

    # Test action selection
    dummy_state = np.random.rand(*state_shape).astype(np.float32)
    action = agent.select_action(dummy_state)
    print(f"\nTest action selection: {action}")

    # Test storing experience
    agent.store_experience(dummy_state, action, 1.0, dummy_state, False)
    print(f"Replay buffer size after storing 1 experience: {len(agent.replay_buffer)}")

    print("\nDDQN agent test complete!")
