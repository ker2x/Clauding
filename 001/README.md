# Deep Q-Network (DQN) for Atari Breakout

A complete educational implementation of Deep Q-Network (DQN) for learning to play Atari Breakout. This project is designed to help you learn Reinforcement Learning (RL) concepts through a practical, well-documented implementation.

## Table of Contents
- [What is Reinforcement Learning?](#what-is-reinforcement-learning)
- [Key RL Concepts](#key-rl-concepts)
  - [Q-Learning](#1-q-learning)
  - [Deep Q-Network (DQN)](#2-deep-q-network-dqn)
  - [Exploration vs Exploitation (Epsilon-Greedy)](#3-exploration-vs-exploitation-epsilon-greedy)
- [Package Dependencies Explained](#package-dependencies-explained)
- [How DQN Learning Works: The Core Algorithm](#how-dqn-learning-works-the-core-algorithm)
  - [What is a Q-Value?](#what-is-a-q-value)
  - [The Bellman Equation](#the-bellman-equation-foundation-of-q-learning)
  - [Loss Function and Training](#how-dqn-learns-the-loss-function)
  - [Reward Propagation](#the-reward-signal-where-learning-comes-from)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Understanding the Code](#understanding-the-code)
- [Training Details](#training-details)
- [Common Misconceptions & FAQs](#common-misconceptions--faqs)
- [Troubleshooting](#troubleshooting)
- [Learning Resources](#learning-resources)

## What is Reinforcement Learning?

Reinforcement Learning is a type of machine learning where an **agent** learns to make **decisions** by interacting with an **environment**. Unlike supervised learning (where you have labels), the agent learns from **rewards** and **punishments**.

### The RL Loop
```
Agent observes STATE → Selects ACTION → Environment gives REWARD
                ↑                                    ↓
                └────────── Agent learns ───────────┘
```

### Key Terms
- **State (s)**: What the agent observes (e.g., game screen)
- **Action (a)**: What the agent can do (e.g., move paddle left/right)
- **Reward (r)**: Feedback from environment (e.g., +1 for breaking brick, 0 otherwise)
- **Policy (π)**: The agent's strategy - maps states to actions
- **Value Function (V/Q)**: Estimates how good a state or action is

## Key RL Concepts

### 1. Q-Learning
Q-Learning learns a **Q-function**: Q(s, a) = expected total reward from taking action 'a' in state 's'

The **Bellman Equation** tells us how to update Q-values:
```
Q(s, a) = r + γ × max Q(s', a')
          ↑   ↑     ↑
       reward │   best future value
           discount factor (0-1)
```

### 2. Deep Q-Network (DQN)
DQN uses a **neural network** to approximate Q(s, a) instead of a table. This is crucial for complex environments like Atari where states are images.

**Key Innovations:**
1. **Experience Replay**: Store experiences and sample randomly
   - Breaks correlation between consecutive experiences
   - More sample efficient (reuse experiences)

2. **Target Network**: Use a separate network for computing targets
   - Provides stable learning targets
   - Updated periodically (not every step)

3. **Frame Stacking**: Stack 4 frames to capture motion
   - Single frame doesn't show velocity/direction
   - Agent can infer ball trajectory from frame sequence

### 3. Exploration vs Exploitation (Epsilon-Greedy)

One of the most important concepts in RL is balancing **exploration** vs **exploitation**:
- **Exploration**: Try random actions to discover new strategies (might find better ones!)
- **Exploitation**: Use learned knowledge to maximize reward (use what you know works)

#### The Epsilon Parameter (ε)

**Epsilon controls how random vs learned the agent's actions are:**

```
With probability ε:     Take RANDOM action (explore)
With probability 1-ε:   Take BEST action according to Q-network (exploit)
```

**Example:** If ε = 0.8 (80% random):
- 80% of actions are random exploration
- 20% of actions use the learned policy
- Agent is still mostly "searching" for good strategies

**Epsilon Decay Schedule:**
- **ε starts at 1.0 (100% random)** - Pure exploration at the beginning
- **ε decays over time** - Gradually shift from exploration to exploitation
- **ε ends at 0.01 (1% random)** - Mostly using learned policy, with tiny bit of exploration

**Why This Matters:**
- If you train for only 100-200 episodes, ε might still be 0.7-0.9 (70-90% random)
- The agent won't show its "true" learned ability until ε is low (~0.1-0.3)
- When you **watch** the agent play, ε is set to 0 (pure learned policy, no randomness)
- This is why a partially-trained agent might look "lazy" - its learned policy is only used 10-30% of the time during training!

**Typical Training Timeline (with default decay of 1M steps):**
| Steps Trained | Epsilon | Behavior |
|--------------|---------|----------|
| 0 | 1.00 (100%) | Completely random |
| 100k | 0.90 (90%) | Still mostly random |
| 250k | 0.78 (78%) | Starting to use learned actions |
| 500k | 0.61 (61%) | Balanced exploration/exploitation |
| 750k | 0.47 (47%) | More exploitation than exploration |
| 1M | 0.37 (37%) | Mostly using learned policy |
| 2M+ | 0.01 (1%) | Almost pure learned policy |

## Package Dependencies Explained

Understanding what each package does in this project:

### Core Libraries

**PyTorch (`torch`)**
- **Purpose**: Deep learning framework for building and training neural networks
- **What we use from torch**:
  - `torch.nn`: Neural network modules (Conv2d, Linear, ReLU, Sequential, etc.)
    - Building the DQN architecture (convolutional and fully-connected layers)
  - `torch.optim`: Optimization algorithms (Adam optimizer)
    - Updating network weights during training
  - `torch.Tensor`: Multi-dimensional arrays for storing data
    - Game frames, network weights, Q-values, gradients
  - **Device support**: CPU, CUDA (NVIDIA), MPS (Apple Silicon)
    - Your M3 Mac uses MPS for GPU acceleration!
- **Example usage in code**:
  ```python
  # Creating network layers
  self.conv = nn.Conv2d(4, 32, kernel_size=8, stride=4)

  # Computing forward pass
  q_values = self.policy_net(state)

  # Computing loss and updating
  loss = nn.functional.mse_loss(current_q, target_q)
  optimizer.backward(loss)
  ```

**TorchVision (`torchvision`) - NOT USED**
- **Purpose**: Computer vision library with pretrained models and image transforms
- **Why it's in requirements.txt**: Often installed with PyTorch as a companion package
- **Why we don't use it**: We use OpenCV (cv2) instead for image preprocessing
  - OpenCV is faster for our specific needs (grayscale conversion, resizing)
  - TorchVision is better for: pretrained ImageNet models, data augmentation pipelines
- **Note**: You could optionally use torchvision.transforms instead of OpenCV:
  ```python
  # Alternative approach (not used in this project):
  from torchvision import transforms
  transform = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Grayscale(),
      transforms.Resize((84, 84)),
      transforms.ToTensor()
  ])
  ```

**Gymnasium (`gymnasium`)**
- **Purpose**: Standard API for reinforcement learning environments
- **What we use it for**:
  - Provides the Atari Breakout game environment
  - Handles game logic, rendering, and state management
  - Returns observations (game frames), rewards, and done signals
  - Standardized interface: `reset()`, `step(action)`, `render()`

**ALE-py (`ale-py`)**
- **Purpose**: Arcade Learning Environment - Atari 2600 emulator
- **What we use it for**:
  - Backend that actually runs the Atari games
  - Provides ROMs (game files) for Breakout and other Atari games
  - Integrated with Gymnasium for RL research

### Data Processing

**NumPy (`numpy`)**
- **Purpose**: Numerical computing library
- **What we use it for**:
  - Array operations on game frames (image data)
  - Mathematical operations (calculating epsilon decay, statistics)
  - Converting between different data formats
  - Efficient storage and manipulation of experience replay buffer

**OpenCV (`cv2`)**
- **Purpose**: Computer vision library
- **What we use it for**:
  - Converting RGB frames to grayscale (color → black/white)
  - Resizing frames from 210×160 to 84×84 (downsampling)
  - Fast image processing operations

**Pillow (`PIL`)**
- **Purpose**: Image processing library
- **What we use it for**:
  - Additional image manipulation if needed
  - Saving/loading images for visualization

### Visualization & Monitoring

**Matplotlib (`matplotlib`)**
- **Purpose**: Plotting and visualization library
- **What we use it for**:
  - Creating training progress plots (rewards, loss, epsilon curves)
  - Visualizing preprocessed frames
  - Generating `logs/training_progress.png`

**TensorBoard (`tensorboard`)**
- **Purpose**: TensorFlow's visualization toolkit (works with PyTorch too)
- **What we use it for**:
  - Advanced training monitoring (optional)
  - Real-time visualization of metrics
  - Comparing different training runs

**tqdm (`tqdm`)**
- **Purpose**: Progress bar library
- **What we use it for**:
  - Showing progress during long training sessions
  - Visual feedback in terminal

## How DQN Learning Works: The Core Algorithm

This is the heart of how the agent learns to play Breakout. Understanding this is crucial!

### The Learning Cycle

```
1. Agent sees GAME FRAMES (state)
2. Agent picks ACTION (up/down/left/right)
3. Environment gives REWARD (+1 for brick, 0 otherwise)
4. Agent updates Q-NETWORK based on Bellman equation
5. Repeat
```

### What is a Q-Value?

**Q(s, a)** = "How good is action 'a' in state 's'?"

More precisely: **Expected total reward** if I take action 'a' in state 's' and play optimally afterward.

**Example:**
- Q(current_frame, "move_left") = 15.3 → Expected score of 15.3 if moving left
- Q(current_frame, "move_right") = 8.7 → Expected score of 8.7 if moving right
- Agent picks action with highest Q-value (move left!)

### The Bellman Equation: Foundation of Q-Learning

The optimal Q-value follows this recursive relationship:

```
Q*(s, a) = R(s,a) + γ × max[Q*(s', a')]
           ↑       ↑   ↑      ↑
      Q-value   reward discount  best Q-value
      we want              factor  in next state
```

**Breaking it down:**

1. **R(s,a)**: Immediate reward for taking action 'a' in state 's'
   - Example: +1 if you break a brick, 0 otherwise

2. **γ (gamma)**: Discount factor (default: 0.99)
   - How much we care about future rewards
   - γ=0.99 means "a reward tomorrow is worth 99% of a reward today"
   - γ=0 means "only care about immediate reward"
   - γ=1 means "future and present rewards equally important"

3. **max[Q*(s', a')]**: Best possible Q-value in the next state
   - After taking action 'a', we reach new state 's'
   - What's the best we can do from there?

4. **The equation says**: "Value of this action = immediate reward + discounted value of best future action"

### How DQN Learns: The Loss Function

Since we use a neural network to approximate Q(s,a), we need to train it. Here's how:

**Step 1: Get Training Data from Experience Replay**

Sample a batch of past experiences:
- (s₁, a₁, r₁, s'₁) - "I did action a₁ in state s₁, got reward r₁, ended up in state s'₁"
- (s₂, a₂, r₂, s'₂)
- ... (32 experiences in a batch)

**Step 2: Compute Current Q-Values**

```python
current_Q = policy_network(s₁).select(a₁)
# What does our current network think Q(s₁, a₁) is?
```

**Step 3: Compute Target Q-Values (using Bellman equation)**

```python
target_Q = r₁ + γ × max(target_network(s'₁))
           ↑     ↑          ↑
      immediate  discount   best Q-value
       reward    factor     in next state
                           (from target network)
```

**Why two networks?**
- **Policy network**: Being trained (constantly changing)
- **Target network**: Frozen copy, updated every 10k steps
- Using a frozen target prevents "chasing a moving target" (makes training stable!)

**Step 4: Compute Loss (Mean Squared Error)**

```python
loss = (current_Q - target_Q)²
```

We want current_Q to match target_Q. If they differ, we have error!

**Example:**
- current_Q = 10.0 (network thinks this action is worth 10 points)
- target_Q = 15.0 (Bellman equation says it should be worth 15 points)
- loss = (10 - 15)² = 25
- Network needs to increase its Q-value estimate!

**Step 5: Update Network with Gradient Descent**

```python
optimizer.backward(loss)  # Compute gradients
optimizer.step()          # Update network weights
```

This adjusts the network weights to make current_Q closer to target_Q.

### Why This Works: The Learning Process

**Initially (random network):**
- Q-values are garbage (random estimates)
- Agent explores randomly (high epsilon)

**After some training:**
- Agent sees: "I moved left, broke brick, got reward +1"
- Bellman equation: Q(moved_left) should be at least 1.0
- Network updates to increase Q(moved_left)
- Repeat 1000s of times

**Eventually:**
- Network learns: "Moving toward ball → higher Q-values"
- Network learns: "Positioning under ball → even higher Q-values"
- Network learns: "Certain patterns → likely to score"
- These patterns emerge from millions of experiences!

### The Reward Signal: Where Learning Comes From

In Atari Breakout:
```python
reward = +1  # When you break a brick
reward =  0  # Otherwise (ball in play, miss, etc.)
```

This simple signal is ALL the agent gets! From just +1/-0, it learns:
- To track the ball
- To position the paddle
- To aim for bricks
- To break all bricks efficiently

**Why it works:**
- Bellman equation propagates rewards backward through time
- Breaking brick at step 100 → increases Q-values at steps 95, 96, 97, 98, 99
- Network learns "actions leading to bricks" have high value
- This happens through millions of update steps!

### Putting It All Together

**Training Loop:**
```
For each episode:
    1. Reset game
    2. For each step:
        a. Pick action (epsilon-greedy)
        b. Execute action, get reward
        c. Store experience in replay buffer
        d. Sample batch from replay buffer
        e. Compute loss using Bellman equation
        f. Update policy network
        g. Every 10k steps: update target network
```

**After millions of steps:**
- Network has seen millions of (state, action, reward) tuples
- Bellman equation has propagated reward information backward
- Q-values now accurately predict: "which actions lead to high scores"
- Agent plays intelligently by picking actions with highest Q-values!

### Key Hyperparameters

**Learning Rate (0.00025)**: How big are network weight updates?
- Too high: Network oscillates, doesn't converge
- Too low: Learning is painfully slow
- 0.00025 is standard for DQN on Atari

**Gamma (0.99)**: How much do we value future rewards?
- Higher (0.99): Plan ahead, consider long-term consequences
- Lower (0.90): Prefer immediate rewards, more "greedy"
- 0.99 means 100 steps ahead is still worth ~37% of immediate reward

**Batch Size (32)**: How many experiences per update?
- Larger: More stable gradients, but slower
- Smaller: Faster updates, but noisier
- 32 is a good balance

**Target Network Update (10k steps)**: How often to update target?
- Too frequent: Training becomes unstable (chasing moving target)
- Too rare: Target becomes stale, learning slows
- 10k steps is empirically proven to work well

## Project Structure

```
.
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── dqn_agent.py                   # DQN agent implementation
├── preprocessing.py               # Atari frame preprocessing
├── train.py                       # Training script
├── train_fast_decay.py            # Alternative training with faster epsilon decay
├── watch_agent.py                 # Watch trained agent play
├── visualize_preprocessing.py     # Visualize preprocessing pipeline
├── test_setup.py                  # Verify installation and setup
├── inspect_checkpoint.py          # Inspect saved checkpoints (epsilon, steps, etc.)
├── checkpoints/                   # Saved model checkpoints
└── logs/                          # Training logs and plots
```

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Installation
```bash
python -c "import gymnasium; print(gymnasium.__version__)"
```

If you get ROM errors, the Atari ROMs should auto-install with `gymnasium[accept-rom-license]`.

## Quick Start

### 1. Verify Setup
Make sure everything is installed correctly:
```bash
python test_setup.py
```

This checks:
- All packages installed
- PyTorch configuration (CPU/GPU)
- Atari environment working
- DQN agent can be created

### 2. Visualize Preprocessing (Recommended)
Understand what the agent sees:
```bash
python visualize_preprocessing.py
```

This shows:
- Original vs preprocessed frames
- Frame stacking visualization
- Memory reduction benefits

### 3. Start Training
Basic training (will take several hours):
```bash
python train.py
```

Training with custom settings:
```bash
python train.py --episodes 2000 --learning-starts 10000
```

**Important:** Training needs 1000+ episodes (500k-1M steps) to see good performance!

### 4. Monitor Training
Training automatically saves:
- `checkpoints/checkpoint_ep*.pt` - Model checkpoints every 500 episodes
- `checkpoints/final_model.pt` - Final model after training completes
- `logs/training_progress.png` - Training curves (updated periodically)

### 5. Inspect Your Model
Check training progress and epsilon value:
```bash
python inspect_checkpoint.py checkpoints/final_model.pt
```

This shows:
- Total steps trained
- Current epsilon (exploration rate)
- Recommendations for next steps

### 6. Resume Training
Continue training from a checkpoint:
```bash
# Resume with current epsilon
python train.py --resume checkpoints/final_model.pt --episodes 1000

# Resume but reset epsilon to explore more
python train.py --resume checkpoints/final_model.pt --reset-epsilon --episodes 1000
```

### 7. Watch Your Agent Play
After training, watch your agent:
```bash
python watch_agent.py --checkpoint checkpoints/final_model.pt
```

Or watch a specific checkpoint:
```bash
python watch_agent.py --checkpoint checkpoints/checkpoint_ep1000.pt --episodes 3
```

## Understanding the Code

### dqn_agent.py
The core DQN implementation with educational comments:

```python
# Key components:
- ReplayBuffer: Stores and samples experiences
- DQN: Neural network (3 conv layers + 2 FC layers)
- DQNAgent: Manages training, action selection, target updates
```

**Important methods:**
- `select_action()`: Epsilon-greedy action selection
- `train_step()`: One gradient descent step
- `update_target_network()`: Copy policy net to target net

### preprocessing.py
Frame preprocessing for Atari:

```python
# Preprocessing steps:
1. RGB (210×160×3) → Grayscale (210×160)
2. Resize → 84×84 (reduces computation by ~84%)
3. Stack 4 frames → (4×84×84) for motion info
4. Clip rewards → {-1, 0, +1} for stability
```

### train.py
Main training loop:

```python
# Training flow:
1. Collect experiences by playing
2. Store in replay buffer
3. Sample random batch
4. Compute loss and update network
5. Periodically update target network
6. Evaluate and save checkpoints
```

## Training Details

### Understanding Training Progress

**The most important metric is STEPS, not episodes:**
- Episodes vary in length (100-300 steps each)
- Epsilon decays based on **total steps**, not episodes
- A model trained for 100k steps behaves very differently than 500k steps

### Expected Performance by Steps

| Steps | Episodes (approx) | Epsilon | Expected Behavior | Avg Score |
|-------|------------------|---------|-------------------|-----------|
| 50k-100k | 200-400 | 0.90-0.95 | Mostly random, occasionally hits ball | 1-3 |
| 200k-300k | 800-1200 | 0.78-0.85 | Starting to track ball, still very random | 2-5 |
| 500k | ~2000 | 0.61 | Learning to position paddle | 5-15 |
| 750k | ~3000 | 0.47 | Decent ball tracking | 10-25 |
| 1M | ~4000 | 0.37 | Good gameplay emerging | 20-40 |
| 2M+ | ~8000+ | 0.01 | Strong performance | 50-100+ |

**Key Insight:** If you train for only 150-350 episodes (~250k steps), epsilon is still 0.75-0.80! The agent is 75-80% random during training, so when you watch it play (epsilon=0), you're only seeing the learned policy that was used 20-25% of the time. That's why it might look "lazy" or weak!

### Training Time Estimates

**On Apple M3 (MPS acceleration):**
- 100 episodes: ~10-15 minutes
- 500 episodes: ~45-60 minutes
- 1000 episodes: ~2-3 hours
- 2000 episodes: ~5-6 hours
- 5000 episodes: ~12-15 hours

**On CPU (without GPU):**
- Approximately 3-5x slower than above

*Note: Performance varies based on hyperparameters, random seed, and hardware.*

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| learning_rate | 0.00025 | Step size for gradient descent |
| gamma | 0.99 | Discount factor (future reward importance) |
| epsilon_start | 1.0 | Initial exploration rate |
| epsilon_end | 0.01 | Final exploration rate |
| epsilon_decay | 1M steps | Steps to decay epsilon |
| buffer_size | 100k | Max experiences in replay buffer |
| batch_size | 32 | Batch size for training |
| learning_starts | 50k | Steps before training starts |
| target_update_freq | 10k | Steps between target net updates |

### Training Time
- **CPU**: 1-2 days for 5000 episodes
- **GPU**: 12-24 hours for 5000 episodes
- **Early stopping**: You can stop once performance plateaus

### Monitoring Progress
Check `logs/training_progress.png` for:
1. **Episode Rewards**: Should increase over time
2. **Episode Lengths**: Longer = agent survives more
3. **Loss**: Should decrease (but may be noisy)
4. **Epsilon**: Should decay to 0.01

## Common Misconceptions & FAQs

### "My agent looks lazy/random after training!"

**This is usually because epsilon is still high!**

Use the inspection tool to check:
```bash
python inspect_checkpoint.py checkpoints/final_model.pt
```

If epsilon is > 0.5 (50% random), your agent needs much more training:
- **Epsilon 0.9 (90% random)**: Only used learned policy 10% of time during training
- **Epsilon 0.5 (50% random)**: Needs 2-3x more training
- **Epsilon 0.2 (20% random)**: Getting there! Maybe 1.5-2x more training
- **Epsilon 0.05 (5% random)**: Good! Agent mostly uses learned policy

**Solution:** Resume training for many more episodes:
```bash
python train.py --resume checkpoints/final_model.pt --episodes 1000
```

### "The agent learned a 'safe but boring' strategy"

This is called finding a **local optimum**. Common in RL! Solutions:

1. **Train longer** - With more exploration, it might discover better strategies
2. **Resume with reset epsilon** - Force more exploration:
   ```bash
   python train.py --resume checkpoints/final_model.pt --reset-epsilon --episodes 500
   ```
3. **Faster epsilon decay** - Start fresh with `train_fast_decay.py` (makes agent exploit sooner)

### "How long should I actually train?"

**Minimum for decent results:** 1000-2000 episodes (500k-1M steps)
**For good performance:** 3000-5000 episodes (1.5M-2.5M steps)
**For best results:** 10000+ episodes (5M+ steps)

The DQN paper trained for **50 million frames** (~12M steps). RL requires patience!

### "Should I start fresh or resume training?"

**Resume if:**
- ✅ Epsilon is still high (> 0.3) - Still exploring, not stuck
- ✅ You want to save time - Keep existing knowledge
- ✅ Loss is decreasing - Agent is learning

**Start fresh if:**
- ❌ Epsilon is very low (< 0.1) AND performance is bad - Might be stuck
- ❌ You want to try different hyperparameters
- ❌ You suspect something broke during training

**Pro tip:** Use `inspect_checkpoint.py` - it gives specific recommendations!

## Troubleshooting

### Issue: Training is very slow
**Solutions:**
- Reduce `learning_starts` to 10000 (trains sooner)
- Use GPU: Check with `python -c "import torch; print(torch.cuda.is_available())"`
- Reduce `epsilon_decay` to 500000 (explores less)

### Issue: Agent not improving
**First, check epsilon:**
```bash
python inspect_checkpoint.py checkpoints/final_model.pt
```

**If epsilon > 0.5:** Agent needs more training! Resume for 1000+ more episodes

**If epsilon < 0.3 and still not improving:**
1. **Check the loss curve** in `logs/training_progress.png` - Is it decreasing?
2. **Watch the agent play** - Is it totally random or showing some learned behavior?
   ```bash
   python watch_agent.py --checkpoint checkpoints/final_model.pt --episodes 5
   ```
3. **Try resuming with reset epsilon** - Force more exploration
4. **Learning rate might be wrong** - Try modifying in `dqn_agent.py` (0.0001 or 0.0005)

**If loss is increasing or exploding:**
- Try restarting training (might be a bad initialization)
- Reduce learning rate to 0.0001

### Issue: "ROM not found" error
```bash
pip install gymnasium[accept-rom-license]
```

### Issue: Out of memory
**Solutions:**
- Reduce `buffer_size` to 50000
- Reduce `batch_size` to 16
- Close other programs

## Learning Resources

### Understanding DQN
1. **Original Paper**: "Playing Atari with Deep Reinforcement Learning" (DeepMind, 2013)
2. **Improved DQN**: "Human-level control through deep reinforcement learning" (Nature, 2015)

### RL Fundamentals
- **Sutton & Barto**: "Reinforcement Learning: An Introduction" (free online)
- **OpenAI Spinning Up**: https://spinningup.openai.com/
- **Deep RL Course**: https://huggingface.co/deep-rl-course/

### Next Steps
After mastering DQN, try:
1. **Double DQN**: Reduces overestimation
2. **Dueling DQN**: Separates value and advantage
3. **Prioritized Experience Replay**: Samples important experiences more
4. **Rainbow DQN**: Combines many improvements
5. **Policy Gradient Methods**: A2C, PPO
6. **Other environments**: Try different Atari games or continuous control

## Code Organization for Learning

Each file is heavily commented to help you learn:

1. **Start with**: `preprocessing.py` - Understand data representation
2. **Then read**: `dqn_agent.py` - Core RL algorithm
3. **Finally**: `train.py` - Training loop and evaluation

Each section has:
- Docstrings explaining "what" and "why"
- Inline comments for implementation details
- References to RL concepts

## Experiment Ideas

Try modifying hyperparameters to see effects:

```bash
# Fast exploration (agent explores less)
python train.py --learning-starts 10000

# Different environment
python train.py --env ALE/Pong-v5

# Longer training
python train.py --episodes 10000
```

## Contributing

Feel free to:
- Add more comments to clarify concepts
- Implement improvements (Double DQN, Dueling DQN, etc.)
- Try different games
- Add better visualization

## License

This is an educational project. Use and modify as needed for learning!

---

**Happy Learning!** If you have questions, check the code comments first - they're designed to be educational. Good luck with your RL journey! 🎮🤖
