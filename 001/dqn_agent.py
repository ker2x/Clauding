"""
Deep Q-Network (DQN) Agent for Atari Breakout

This file implements a DQN agent, which is one of the foundational deep RL algorithms.
DQN combines Q-learning (a classic RL algorithm) with deep neural networks.

Key RL Concepts:
1. Q-Learning: Learn a Q-function Q(s,a) that estimates the expected return of taking action 'a' in state 's'
2. Deep Q-Network: Use a neural network to approximate Q(s,a)
3. Experience Replay: Store experiences and sample randomly to break correlation
4. Target Network: Use a separate network for stability during training
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple


# Experience tuple: stores one transition (state, action, reward, next_state, done)
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """
    Experience Replay Buffer

    In RL, sequential experiences are highly correlated. This can cause instability
    when training neural networks. Experience replay solves this by:
    1. Storing experiences in a buffer
    2. Sampling random batches for training (breaking correlation)
    3. Reusing experiences multiple times (sample efficiency)
    """

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store an experience in the buffer"""
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a random batch of experiences"""
        experiences = random.sample(self.buffer, batch_size)

        # Convert batch of experiences to batch of tensors
        # Explicitly convert to float32 for compatibility with MPS
        states = torch.from_numpy(np.array([e.state for e in experiences])).float()
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences])).float()
        dones = torch.FloatTensor([e.done for e in experiences])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    """
    Deep Q-Network Architecture

    This network takes a state (processed game frames) and outputs Q-values for each action.
    The architecture uses convolutional layers (like in computer vision) because
    game frames are images with spatial structure.
    """

    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()

        # Convolutional layers to process visual input
        # These extract features like edges, patterns, ball position, etc.
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Calculate size after convolutions (for 84x84 input)
        conv_output_size = 64 * 7 * 7  # This depends on input size and conv layers

        # Fully connected layers to output Q-values for each action
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        """
        Forward pass: state -> Q-values

        Args:
            x: Input state (batch_size, channels, height, width)
        Returns:
            Q-values for each action (batch_size, num_actions)
        """
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


class DQNAgent:
    """
    DQN Agent with Double DQN and other improvements

    This agent learns to play Atari games by:
    1. Observing states (game frames)
    2. Selecting actions using epsilon-greedy policy
    3. Learning from experiences using the Bellman equation
    4. Improving its policy over time
    """

    def __init__(
        self,
        state_shape,
        num_actions,
        learning_rate=0.00025,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=1000000,
        buffer_size=100000,
        batch_size=32,
        target_update_freq=10000,
        device=None  # Auto-detect best device
    ):
        """
        Initialize DQN Agent

        Args:
            state_shape: Shape of preprocessed state (channels, height, width)
            num_actions: Number of possible actions in the environment
            learning_rate: Learning rate for optimizer
            gamma: Discount factor (how much to value future rewards, 0-1)
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Steps over which to decay epsilon
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            target_update_freq: How often to update target network
            device: 'cuda', 'mps', 'cpu', or None (auto-detect)
        """
        # Auto-detect best device
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Epsilon-greedy parameters (for exploration vs exploitation)
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        # Create main network (online network) and target network
        # Target network provides stable Q-targets during training
        self.policy_net = DQN(state_shape[0], num_actions).to(self.device)
        self.target_net = DQN(state_shape[0], num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is not trained directly

        # Optimizer and loss function
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size)

        print(f"DQN Agent initialized on device: {self.device}")
        print(f"Policy network parameters: {sum(p.numel() for p in self.policy_net.parameters())}")
        if self.device == 'mps':
            print("Using Apple Silicon GPU acceleration!")
        elif self.device == 'cpu':
            print("Note: Training on CPU. This will be slower but more stable.")

    def get_epsilon(self):
        """
        Calculate current epsilon for epsilon-greedy policy

        Epsilon decays from epsilon_start to epsilon_end over epsilon_decay steps.
        This implements the exploration-exploitation tradeoff:
        - Early: Explore more (high epsilon = random actions)
        - Later: Exploit more (low epsilon = use learned policy)
        """
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-1. * self.steps_done / self.epsilon_decay)
        return epsilon

    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy

        Args:
            state: Current state (numpy array)
            training: If False, always use greedy policy (for evaluation)

        Returns:
            Action index (int)
        """
        # Only increment steps during training
        if training:
            self.steps_done += 1

        epsilon = self.get_epsilon() if training else 0.0

        # Epsilon-greedy: with probability epsilon, take random action (explore)
        if training and random.random() < epsilon:
            return random.randrange(self.num_actions)

        # Otherwise, take best action according to Q-network (exploit)
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(1).item()

    def train_step(self):
        """
        Perform one training step

        This implements the DQN learning algorithm:
        1. Sample a batch of experiences from replay buffer
        2. Compute Q-values for current states
        3. Compute target Q-values using target network
        4. Compute loss and update policy network

        Returns:
            Loss value (float) or None if not enough experiences
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch of experiences
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Compute Q(s, a) - the Q-value of the action that was taken
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Compute target Q-values using Bellman equation:
        # Q*(s, a) = r + gamma * max_a' Q(s', a')
        # If episode is done, target is just the reward
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss (Mean Squared Error between current and target Q-values)
        loss = nn.functional.mse_loss(current_q_values.squeeze(), target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """
        Update target network with policy network weights

        The target network is updated periodically to provide stable Q-targets.
        Without this, training can be unstable (chasing a moving target).
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        """Save model checkpoint"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done
        }, path)
        print(f"Model saved to {path}")

    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint['steps_done']
        print(f"Model loaded from {path}")
