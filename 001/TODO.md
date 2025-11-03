# TODO List - DQN Breakout Project

This file tracks immediate training tasks and potential code improvements for the DQN Breakout implementation.

---

## 🎯 Immediate Training Tasks

Current model status: **487,340 steps trained**, **epsilon: 0.618** (61.8% random, 38.2% learned)

- [ ] **Continue training for 1000+ episodes** to reduce epsilon (target: ~0.1-0.3)
  - Command: `python train.py --resume checkpoints/final_model.pt --episodes 1000`
  - Goal: Reach ~1.5M+ total steps for strong performance

- [ ] **Monitor training progress** in `logs/training_progress.png`
  - Check if loss is decreasing steadily
  - Verify rewards are improving over time

- [ ] **Inspect checkpoint after training** to verify progress
  - Command: `python inspect_checkpoint.py checkpoints/final_model.pt`
  - Verify epsilon has decreased to 0.1-0.3 range
  - Confirm steps are ~1.5M+

- [ ] **Evaluate improved model performance**
  - Command: `python watch_agent.py --checkpoint checkpoints/final_model.pt --episodes 5`
  - Look for consistent scoring, better paddle positioning

- [ ] **Adjust strategy if needed** (if performance is still poor)
  - Option 1: Continue training with `--reset-epsilon` to force more exploration
  - Option 2: Try fast decay training: `python train_fast_decay.py --resume checkpoints/final_model.pt`

---

## ⚡ Performance Improvements

### High Priority

- [ ] **Fix Double DQN Implementation** (CRITICAL - misleading comment!)
  - **Location**: `dqn_agent.py:254-259`
  - **Issue**: Comment claims "Double DQN" but code uses vanilla DQN
  - **Fix**: Use policy network to select action, target network to evaluate
  - **Impact**: Reduces Q-value overestimation, more stable learning
  - **Effort**: 1 line change

- [ ] **Optimize GPU Utilization**
  - **Location**: `dqn_agent.py:44-56` (ReplayBuffer.sample)
  - **Issue**: Creates tensors on CPU, then moves to device
  - **Fix**: Pre-allocate device tensors, use `pin_memory=True` for DataLoader
  - **Impact**: 10-15% training speedup

- [ ] **Enable cuDNN Benchmark**
  - **Location**: `dqn_agent.py:174` (DQN.__init__)
  - **Fix**: Add `torch.backends.cudnn.benchmark = True`
  - **Impact**: 5-10% faster forward passes

### Medium Priority

- [ ] **Implement Lazy Frames for Memory Efficiency**
  - **Location**: `dqn_agent.py:26-60` (ReplayBuffer)
  - **Issue**: Stores full numpy arrays, high memory usage
  - **Fix**: Use LazyFrames or shared memory for frame stacking
  - **Impact**: 3-4x memory reduction, can use larger buffer

- [ ] **Add Batched Action Selection**
  - **Location**: `dqn_agent.py:203-225` (select_action)
  - **Fix**: Support batch dimension for vectorized evaluation
  - **Impact**: 2-3x faster evaluation when testing multiple episodes

- [ ] **Optimize Preprocessing Pipeline**
  - **Location**: `preprocessing.py:69-85`
  - **Fix**: Pre-allocate buffers for cv2 operations
  - **Impact**: Reduce allocation overhead during data collection

### Low Priority

- [ ] **Add Mixed Precision Training**
  - **Location**: `dqn_agent.py` (train_step)
  - **Fix**: Use `torch.cuda.amp` for automatic mixed precision
  - **Impact**: 30-50% speedup on compatible GPUs (A100, RTX 30xx+)

---

## 📚 Documentation Improvements

### High Priority

- [ ] **Document Tensor Shapes in ReplayBuffer.sample()**
  - **Location**: `dqn_agent.py:44`
  - **Add**: Return type documentation with shapes
  - **Example**: `Returns: (states: [B,4,84,84], actions: [B], rewards: [B], ...)`

- [ ] **Explain Epsilon Decay Formula**
  - **Location**: `dqn_agent.py:190-201`
  - **Add**: Mathematical explanation and visualization reference
  - **Why**: Most common confusion point for users

- [ ] **Document Network Architecture Choices**
  - **Location**: `dqn_agent.py:76-83`
  - **Add**: Explain why 8x8→4x4→3x3 kernel progression
  - **Note**: Matches original DQN Nature paper

### Medium Priority

- [ ] **Explain Gradient Clipping Value**
  - **Location**: `dqn_agent.py:268`
  - **Add**: Why max_norm=10 specifically, reference to paper

- [ ] **Document Frame Stacking Initialization**
  - **Location**: `preprocessing.py:55-57`
  - **Add**: Why repeat same frame vs zeros (prevents artificial motion)

- [ ] **Add Common Pitfalls Section to README**
  - **Location**: `CLAUDE.md` or new `PITFALLS.md`
  - **Topics**: Forgetting epsilon decay is step-based, importance of learning_starts, etc.

### Low Priority

- [ ] **Add Algorithm Flowchart**
  - **Location**: `docs/` folder or README
  - **Tool**: mermaid.js diagram or image
  - **Content**: Visual DQN algorithm flow

- [ ] **Document Target Network Update Alternatives**
  - **Location**: `dqn_agent.py:273-280`
  - **Add**: Mention polyak averaging (soft updates) as alternative

---

## ✨ Feature Additions

### High Priority

- [ ] **Add TensorBoard Logging**
  - **Location**: `train.py` (Trainer class)
  - **Note**: `tensorboard` already in requirements.txt but not used!
  - **Add**: Loss, rewards, epsilon, Q-values per step
  - **Impact**: Real-time training visualization, better than static plots
  - **Command**: `tensorboard --logdir=logs/tensorboard`

- [ ] **Implement Checkpoint Management**
  - **Location**: `train.py:180-184`
  - **Features**:
    - Save best model (by evaluation score)
    - Keep only last N checkpoints to save disk space
    - Add metadata (training time, hyperparameters) to checkpoints
  - **Impact**: Easy to find best model, cleaner directory

- [ ] **Add Early Stopping**
  - **Location**: `train.py` (Trainer.train method)
  - **Logic**: Stop if no improvement in evaluation score for 500 episodes
  - **Impact**: Save compute time, automatic convergence detection

### Medium Priority

- [ ] **Implement Prioritized Experience Replay (PER)**
  - **Location**: `dqn_agent.py:26-60` (new PrioritizedReplayBuffer class)
  - **Algorithm**: Sample experiences by TD-error priority
  - **Impact**: 30-40% fewer samples needed to reach same performance
  - **Complexity**: Significant implementation effort (sum tree data structure)

- [ ] **Add Video Recording**
  - **Location**: `train.py` or `watch_agent.py`
  - **Use**: `gym.wrappers.RecordVideo` wrapper
  - **Feature**: Auto-save videos of best episodes during training
  - **Impact**: Visual confirmation of learning without manual watching

- [ ] **Implement Learning Rate Scheduling**
  - **Location**: `dqn_agent.py:180`
  - **Strategy**: Reduce LR by 0.5 every 1M steps
  - **Impact**: Better fine-tuning in later training stages

- [ ] **Add Comprehensive Evaluation Metrics**
  - **Location**: `train.py:191-217`
  - **Metrics**:
    - Max score per episode
    - Episode length distribution
    - Q-value statistics (mean, std, max)
    - Action distribution histogram
  - **Impact**: Better understanding of agent behavior

- [ ] **Create Q-Value Visualization Tool**
  - **Location**: New file `visualize_qvalues.py`
  - **Feature**: Show Q-values for each action during gameplay
  - **Impact**: Understand what agent "thinks" about each action

### Low Priority

- [ ] **Add Dueling Network Architecture**
  - **Location**: `dqn_agent.py:89-93` (modify DQN network)
  - **Change**: Split into value stream and advantage stream
  - **Impact**: Better learning for states where actions don't matter much

- [ ] **Implement Hyperparameter Config Files**
  - **Location**: New file `config.yaml`, modify `train.py`
  - **Format**: YAML or JSON for experiment tracking
  - **Impact**: Easy to reproduce different training runs

- [ ] **Add Multi-Environment Support**
  - **Location**: `train.py`, add `--env` argument
  - **Feature**: Easy scripts for other Atari games
  - **Games**: Pong, Space Invaders, Pac-Man, etc.

- [ ] **Create Interactive Q-Value Explorer**
  - **Location**: New file `explore_qvalues.py`
  - **Feature**: Step through game frame-by-frame, see Q-values
  - **Impact**: Great educational tool

- [ ] **Add Comparison with Random Policy**
  - **Location**: New file `compare_policies.py`
  - **Feature**: Quantify improvement over random baseline
  - **Impact**: Clear metrics on actual learning

- [ ] **Create Ablation Study Helper**
  - **Location**: New file `ablation_study.py`
  - **Tests**: Train without frame stacking, target network, etc.
  - **Impact**: Understand importance of each DQN component

---

## 🔧 Code Quality Improvements

### High Priority

- [ ] **Add Unit Tests**
  - **Location**: New directory `tests/`
  - **Framework**: pytest
  - **Tests Needed**:
    - ReplayBuffer sampling correctness
    - DQN forward pass output shapes
    - Epsilon decay calculation accuracy
    - Preprocessing transformations
    - Save/load checkpoint integrity
  - **Impact**: Catch bugs early, safer refactoring

- [ ] **Add Input Validation**
  - **Location**: `dqn_agent.py` (select_action, train_step)
  - **Add**: Assert statements or exceptions for:
    - State shape validation
    - Action bounds checking
    - Batch size consistency
  - **Impact**: Better error messages, easier debugging

- [ ] **Improve Checkpoint Loading Robustness**
  - **Location**: `dqn_agent.py:292-299`
  - **Add**: Try-except with informative errors
  - **Check**: Architecture compatibility, version matching
  - **Impact**: Better error messages when loading fails

- [ ] **Add Environment Cleanup**
  - **Location**: `train.py:85-88`
  - **Fix**: Use context managers or try-finally blocks
  - **Impact**: Prevent resource leaks on exceptions

### Medium Priority

- [ ] **Add Type Hints Throughout**
  - **Location**: All Python files
  - **Add**: Type annotations for function parameters and returns
  - **Impact**: Better IDE support, catch type errors early
  - **Effort**: ~2-3 hours for entire codebase

- [ ] **Replace Logging print() with logging Module**
  - **Location**: All Python files
  - **Replace**: All print() statements with proper logging
  - **Levels**: INFO, DEBUG, WARNING, ERROR
  - **Impact**: Configurable verbosity, can log to file

- [ ] **Add Reproducibility via Seeding**
  - **Location**: `train.py` (add `--seed` argument)
  - **Seed**: torch, numpy, random, environment
  - **Impact**: Reproducible results for debugging and experiments

- [ ] **Pin Requirements Versions**
  - **Location**: `requirements.txt`
  - **Change**: Replace `>=` with pinned versions or ranges
  - **Example**: `torch==2.1.0` or `torch>=2.0,<3.0`
  - **Impact**: Prevent compatibility issues

### Low Priority

- [ ] **Refactor Magic Numbers**
  - **Location**: `dqn_agent.py:86` (hardcoded `7*7`)
  - **Fix**: Calculate conv output size dynamically or use constants
  - **Impact**: More maintainable code

- [ ] **Extract Common Training Code**
  - **Location**: `train.py` and `train_fast_decay.py`
  - **Issue**: Duplicate environment setup code
  - **Fix**: Create shared utility functions
  - **Impact**: DRY principle, easier maintenance

- [ ] **Improve Device Handling**
  - **Location**: `dqn_agent.py:151-160`
  - **Fix**: Create `get_device()` utility with fallback logic
  - **Impact**: Better MPS edge case handling

- [ ] **Modernize with dataclass**
  - **Location**: `dqn_agent.py:23` (Experience namedtuple)
  - **Fix**: Replace namedtuple with dataclass
  - **Impact**: More modern, easier to extend

- [ ] **Add Model.eval() During Inference**
  - **Location**: `dqn_agent.py:222` (select_action)
  - **Fix**: Call `.eval()` before inference, `.train()` after
  - **Note**: Not critical here (no dropout/batch norm), but good practice

---

## 🧪 Testing Gaps

### High Priority

- [ ] **Create Unit Test Suite**
  - **Location**: `tests/test_dqn_agent.py`
  - **Tests**:
    - Test replay buffer sampling returns correct shapes
    - Test DQN network forward pass
    - Test epsilon decay calculation
    - Test train_step updates weights
  - **Framework**: pytest

- [ ] **Add Preprocessing Tests**
  - **Location**: `tests/test_preprocessing.py`
  - **Tests**:
    - Output shape always (4, 84, 84)
    - Pixel values in [0, 255]
    - Frame stacking order correct
    - Grayscale conversion matches expected method

- [ ] **Add Checkpoint Tests**
  - **Location**: `tests/test_checkpoints.py`
  - **Tests**:
    - Save/load preserves weights
    - Save/load preserves optimizer state
    - Loading invalid checkpoint fails gracefully

### Medium Priority

- [ ] **Create Training Smoke Tests**
  - **Location**: `tests/test_training.py`
  - **Test**: Run 10 episodes, check loss decreases
  - **Impact**: Catch training bugs before long runs
  - **Duration**: ~2 minutes

- [ ] **Add Determinism Tests**
  - **Location**: `tests/test_reproducibility.py`
  - **Test**: Same seed produces same results
  - **Impact**: Ensure reproducibility claims are valid

- [ ] **Convert test_setup.py to pytest**
  - **Location**: `test_setup.py`
  - **Change**: Add proper assertions instead of print statements
  - **Impact**: Can run in CI/CD pipeline

---

## 🎓 Educational Enhancements

### Medium Priority

- [ ] **Create Step-by-Step Tutorial**
  - **Location**: `docs/TUTORIAL.md`
  - **Content**: Walkthrough of training first agent, interpreting results
  - **Impact**: Lower barrier to entry for beginners

- [ ] **Add Glossary of RL Terms**
  - **Location**: `docs/GLOSSARY.md`
  - **Terms**: Epsilon-greedy, replay buffer, Bellman equation, etc.
  - **Impact**: Helpful reference for learners

- [ ] **Create Jupyter Notebook Demo**
  - **Location**: `notebooks/demo.ipynb`
  - **Content**: Interactive training walkthrough with visualizations
  - **Impact**: Great for teaching and presentations

### Low Priority

- [ ] **Add Training Timeline Visualization**
  - **Feature**: Visual chart showing expected progress at different steps
  - **Impact**: Set realistic expectations for new users

- [ ] **Create Architecture Diagram**
  - **Tool**: Draw.io or similar
  - **Content**: DQN network architecture with dimensions
  - **Impact**: Visual learners understand better

---

## 🚀 Advanced Features (Future Work)

These are stretch goals for potential future development:

### Challenging but Valuable

- [ ] **Multi-GPU Training Support**
  - **Impact**: Train multiple agents in parallel
  - **Complexity**: Requires DataParallel or DistributedDataParallel
  - **Effort**: 1-2 days

- [ ] **Asynchronous Training (A3C-style)**
  - **Impact**: Multiple environment workers collecting experiences in parallel
  - **Complexity**: Requires multiprocessing and careful synchronization
  - **Effort**: 3-5 days

### Research-Oriented

- [ ] **Curriculum Learning**
  - **Idea**: Start with easier game variants, gradually increase difficulty
  - **Impact**: Potentially faster learning
  - **Effort**: Depends on environment availability

- [ ] **Transfer Learning**
  - **Idea**: Pre-train on one game, fine-tune on another
  - **Impact**: Interesting research direction
  - **Effort**: 2-3 days experiment setup

- [ ] **Rainbow DQN**
  - **Combine**: Double DQN + Dueling + PER + Noisy Nets + Distributional + Multi-step
  - **Impact**: State-of-the-art performance
  - **Effort**: 1-2 weeks implementation

---

## 📊 Priority Summary

### Must Fix Now
1. ✅ Fix misleading Double DQN comment (1 line!)
2. ✅ Continue training current model to 1.5M+ steps
3. ✅ Add basic unit tests

### High Impact, Should Do Soon
4. ✅ Add TensorBoard logging
5. ✅ Implement checkpoint management
6. ✅ Add proper error handling
7. ✅ Type hints throughout

### Nice to Have
8. ⚠️ Prioritized Experience Replay
9. ⚠️ Video recording
10. ⚠️ Learning rate scheduling

### Future Exploration
11. 💡 Dueling architecture
12. 💡 Multi-environment support
13. 💡 Advanced features (A3C, Rainbow)

---

## 📝 Notes

**Current Codebase Quality**: ⭐⭐⭐⭐☆ (4/5)
- **Strengths**: Excellent documentation, clean code, great for learning
- **Gaps**: Missing tests, some misleading comments, no advanced features
- **Overall**: One of the best-documented educational RL implementations

**Estimated Time to Complete All High Priority Items**: ~2-3 days of focused work

**Most Important Insight from Analysis**:
The comment on line 112 claims "Double DQN" but the actual implementation (lines 254-259) uses vanilla DQN. This should be fixed to either:
1. Update comment to say "Vanilla DQN", or
2. Actually implement Double DQN (recommended - it's just 1 line!)

---

*Last Updated: 2025-11-01*
*Generated with assistance from Claude Code*
