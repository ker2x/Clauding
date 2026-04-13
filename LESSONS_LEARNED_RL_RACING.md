# Lessons Learned: Reinforcement Learning for Sim Racing

This document captures the key insights, mistakes, and hard-won knowledge from projects 001 through 006 -- a progression from DQN on Atari Breakout to a custom-physics SAC racing agent with domain randomization, parallel evolutionary training, and GT SOPHY-inspired observation spaces.

---

## Project Timeline

| Project | Algorithm | Environment | Key Innovation |
|---------|-----------|-------------|----------------|
| 001 | DQN | Atari Breakout | Baseline discrete RL |
| 002 | DDQN | CarRacing-v3 (Box2D) | Discretized continuous actions, snapshot state mode |
| 003 | SAC | CarRacing-v3 (Box2D) | True continuous actions, entropy tuning |
| 004 | SAC | CarRacing-v3 (Box2D) | Checkpoint-based rewards, polygon collision |
| 005 | SAC | CarRacing-v3 (custom physics) | Custom 2D engine, parallel selection training |
| 006 | SAC | CarRacing-v3 (custom physics) | Suspension, domain randomization, frame stacking |

---

## 1. Algorithm Selection

### DQN is the wrong tool for continuous control

Project 002 discretized the 3D continuous action space (steering, gas, brake) into 9 bins (3 steering x 3 gas/brake). This worked but imposed hard limitations:
- Coarse steering with only 3 levels (-0.8, 0, +0.8) produced jerky, unstable control.
- No fine-grained throttle modulation -- the agent could only coast, brake hard, or gas hard.
- The combinatorial explosion problem: finer discretization means exponentially more actions and sparser Q-value coverage.

**Lesson:** If the problem has a continuous action space, use an algorithm designed for it. Switching to SAC in project 003 was the single biggest leap in agent quality.

### SAC's entropy tuning is powerful but needs monitoring

SAC's automatic alpha (entropy coefficient) tuning is one of its best features, but it can silently fail:
- Alpha should decay from ~0.8 to 0.01-0.2 during training.
- If alpha stays stuck at 0.8, the agent is exploring too much and never exploiting.
- If alpha crashes to near-zero too quickly, the policy collapses to a deterministic mode and stops exploring useful alternatives.

**Lesson:** Always plot alpha over training. It's the single best health indicator for SAC.

---

## 2. State Representation Matters More Than Algorithm Tuning

### Visual mode is too slow for training

All projects experimented with visual (CNN-based) and vector (MLP-based) state modes. The numbers are clear:

| Mode | Steps/sec | Training time (1M steps) |
|------|-----------|--------------------------|
| Visual (96x96 RGB) | ~57 | ~4.9 hours |
| Vector (36-67D) | ~300-580 | ~0.5-1 hour |

Beyond raw speed, the replay buffer sampling bottleneck in visual mode dominates: converting 75 MB of image data per batch from list -> numpy -> torch takes ~20ms -- 25x slower than the actual neural network forward pass.

**Lesson:** Use vector state for training, visual mode only for watching/debugging. If you must train on pixels, invest in a proper tensor-backed replay buffer to eliminate the conversion pipeline.

### The observation space evolved significantly

| Project | Observation | Dimensions |
|---------|-------------|------------|
| 002 | Basic car state (no track info) | 11D |
| 002 | Snapshot: car + track + lookahead | 36D |
| 005 | + obstacles/competitors | 67D |
| 006 | + slip angles, forces, steering state, accelerations | 53D base x N frames |

The 11D vector mode in 002 was fast but useless -- without track geometry or lookahead waypoints, the agent couldn't anticipate turns and just learned to go straight.

**Lesson:** The agent can only learn from what it can observe. Skimp on observations and no amount of training will help. The jump from 11D to 36D (adding track geometry and 10 lookahead waypoints) was the difference between an agent that crashes into every wall and one that follows the track.

### Frame stacking provides temporal information cheaply

Project 006 added configurable frame stacking (concatenating N consecutive observations). With a single frame, the agent knows current slip angles but not whether they're increasing or decreasing. With 4 frames, it can compute finite differences to estimate:
- Slip angle rates (anticipate tire saturation before grip loss)
- Load transfer rates (predict weight shift during maneuvers)
- Lateral drift rate (correct trajectory deviations early)

**Lesson:** Frame stacking is nearly free in vector mode (53D x 4 = 212D is still tiny for an MLP) and gives the network temporal derivatives it would otherwise have to memorize internally. Always consider it for physics-based environments.

---

## 3. Reward Engineering is the Hardest Part

### The reward structure was rewritten at least 4 times

| Project | Reward Strategy | Problem |
|---------|----------------|---------|
| 002 | Raw gym rewards (-0.1/frame, tiles) | Agent learned to sit still to avoid -100 off-track penalty |
| 003 | + speed bonus, off-track penalty | Agent wiggled steering to game stationary detection |
| 004 | Checkpoint-based (10 x 100 pts) | Sparse, but cleaner signal |
| 005-006 | Checkpoint + velocity + step penalty + off-track | Balanced sparse + dense |

### Agents will exploit every loophole

Project 002's most memorable bug: the stationary car termination checked if the agent was "making progress" by looking at whether new tiles were visited. Agents discovered they could wiggle the steering wheel while stationary -- the action changed each frame, which some early implementations interpreted as "activity." The fix required checking actual velocity (`speed > 0.5 m/s`), not just whether inputs changed.

**Lesson:** Assume the agent will find and exploit any gap between your reward intent and your reward implementation. The agent optimizes the reward function you wrote, not the one you meant.

### Collision detection accuracy matters enormously

Project 004 discovered that the distance-to-center collision detection was fundamentally broken:
- Threshold too small (2.0): wheels centered on the track but between tile centers were falsely marked as off-track, generating massive false penalties.
- Threshold too large (8.0): wheels clearly off the road counted as on-track, eliminating the penalty signal entirely.

The fix was polygon-based ray-casting collision detection against actual tile boundaries. Performance impact was negligible (~0.05ms/step) because spatial partitioning already limited checks to ~61 nearby tiles.

**Lesson:** Your reward signal is only as good as the sensors feeding it. Inaccurate collision detection is indistinguishable from a broken reward function.

### Sparse vs. dense rewards: both are needed

Project 004's checkpoint-based reward (10 checkpoints x 100 points) was clean and unambiguous, but pure sparse rewards make early learning very slow -- the agent has to randomly stumble into a checkpoint before getting any positive signal.

The final reward structure in 005-006 combines:
- **Sparse:** +100 per checkpoint, +1000 for lap completion (clear milestones)
- **Dense:** +0.1 per m/s forward velocity (continuous progress signal), -2.0 per frame (time pressure)
- **Constraints:** -1.0 per wheel off-track (only beyond threshold), -100 + termination for all 4 wheels off

**Lesson:** Dense rewards bootstrap learning; sparse rewards define the objective. Use both, and make sure the dense rewards don't create local optima that conflict with the sparse goal.

---

## 4. Environment and Physics

### Custom physics was worth the investment

Project 005 replaced Box2D with a custom 2D physics engine. This was a major effort but paid off:
- **Interpretability:** Every force, slip angle, and load transfer is directly accessible for the observation space.
- **Tunability:** Magic Formula tire parameters, suspension, drivetrain -- all configurable as dataclass constants.
- **Domain randomization:** Trivial to randomize mass, grip, power, etc. when you own the physics.
- **Debugging:** When the agent behaves unexpectedly, you can instrument every physics quantity (telemetry viewer, dyno GUI).

The code review confirmed the physics is sound: Symplectic Euler integration, correct Coriolis terms in the rotating reference frame, proper load transfer, and realistic Pacejka tire model.

**Lesson:** If you're going to iterate heavily on reward shaping and observation design, owning the environment pays for itself quickly. Box2D was a black box that made debugging reward anomalies painful.

### Suspension and steering dynamics add realism but increase learning difficulty

Project 006 added per-wheel spring-damper suspension and rate-limited steering (lock-to-lock time ~0.53s). These make the physics more realistic but also:
- Introduced positive feedback loops that required careful damping (overdamped ratio 1.09).
- Required physics bugfixes (double-counting weight forces, brake force too high at 780 rad/s^2 -> fixed to 50).
- Made the observation space more complex (steering angle + rate added to state).

**Lesson:** More realistic physics helps sim-to-real transfer but requires careful numerical stability work. Every new physics feature needs its own validation tests (skidpad, braking distance, etc.).

---

## 5. Training Infrastructure

### Parallel selection training is a practical win

Project 005 introduced parallel selection training: N independent agents train on separate CPU cores, with periodic tournaments selecting the best performer. The winner is cloned to other slots.

| Method | Wall-clock | CPU Usage | Data/sec |
|--------|-----------|-----------|----------|
| Single agent | 10h baseline | 120% | 1x |
| Multi-car (same track) | 30h | 120% | 1.33x |
| VectorEnv (parallel) | 4-5h | 400%+ | 3.3x |
| Selection parallel (8 agents) | ~2h | 800% | ~8x |

**Lesson:** For CPU-bound vector mode training, parallel selection is the best approach on multi-core machines. The evolutionary pressure from tournaments also helps escape local optima. For GPU-bound visual mode, VectorEnv with synchronized seeds is more appropriate.

### Device selection depends on the workload

A counterintuitive finding: for small MLP networks with vector state mode, **CPU is faster than GPU (MPS)**. The GPU transfer overhead for tiny tensors outweighs the parallelism benefit.

| Workload | Best Device | Reason |
|----------|------------|--------|
| Vector mode MLP (36-67D input) | CPU | GPU transfer overhead > compute savings |
| Visual mode CNN (4x96x96 input) | MPS/CUDA | 10x speedup, convolutions are GPU-native |

MacBook Air thermal throttling was also an issue: visual mode training on CPU starts fast but slows 2-3x after 5-10 minutes as the passive cooling saturates.

**Lesson:** Always benchmark your specific workload on available devices. Don't assume GPU is always faster.

### The replay buffer sampling is the real bottleneck (for visual mode)

Detailed profiling in projects 004-005 revealed that the Conv2D forward pass takes ~0.1ms, but batch sampling takes ~20ms -- a 200x ratio. The culprit: converting Python lists of numpy arrays into torch tensors involves 3 copy stages (list -> np.array -> torch.FloatTensor -> .to(device)), copying ~75 MB per batch.

**Lesson:** If training is slow, profile before assuming the network is the bottleneck. Data pipelines often dominate. For visual mode, consider pre-allocating tensor storage or using shared-memory replay buffers.

---

## 6. Domain Randomization and Generalization

Project 006 implemented configurable domain randomization (mass, grip, power, friction, drag, etc.) with four presets:

| Preset | Variation | Convergence Slowdown |
|--------|-----------|---------------------|
| Conservative | +-5% | ~5-10% |
| Moderate | +-10% | ~15-25% |
| Aggressive | +-15-25% | ~30-50% |
| Wet surface | 60-80% grip | Similar to moderate |

**Lesson:** Start conservative, progress through curriculum learning. Aggressive randomization from the start can prevent learning entirely. The recommended workflow is: validate without randomization -> conservative -> moderate -> aggressive.

**Lesson:** Always evaluate on nominal (non-randomized) parameters. Training reward variance increases with randomization, making it hard to assess progress without a fixed evaluation baseline.

---

## 7. Debugging and Observability

### The step counter bug (002) wasted days

Project 002 had a circular dependency where `steps_done` was only incremented inside `train_step()`, but `train_step()` was only called when `steps_done >= learning_starts`. Training never started. The fix was trivial (increment the counter in the main loop), but the symptoms were subtle -- the agent appeared to train but epsilon never decayed and no loss values appeared.

**Lesson:** Always verify that your training loop actually enters the learning phase. Log the first loss value and the step count prominently.

### Comprehensive logging pays for itself

Project 002 added CSV metrics, human-readable logs, evaluation logs, and system info files. Every subsequent project inherited this infrastructure. The ability to `tail -f logs/training.log` during training and `pandas.read_csv('logs/training_metrics.csv')` afterward made debugging 10x faster.

**Lesson:** Invest in logging infrastructure early. At minimum: per-episode CSV metrics, a human-readable log with timestamps, and a system config snapshot for reproducibility.

### Network health analysis (WeightWatcher) is useful for diagnosis

Project 006 integrated WeightWatcher for analyzing trained networks using random matrix theory. The alpha (power law exponent) metric was particularly useful:
- Alpha ~ 2.0: optimal generalization
- Alpha 2-5: well-trained
- Alpha > 6: undertrained/random
- Alpha < 2: overfitting

**Lesson:** Having a model-quality metric that doesn't require test data is valuable for long training runs. It helps decide when to stop training and whether a checkpoint is worth keeping.

---

## 8. Meta-Lessons About Working With AI Assistants

These projects were primarily built through collaboration with Claude Code. Some observations:

### Documentation accumulates fast
Each CLAUDE.md file grew to 300-600 lines as implementation details, bug fixes, and design decisions were documented. This was invaluable for context continuity across conversations but needs periodic pruning.

### Incremental complexity beats big-bang rewrites
The progression 001->002->003 worked well: each project built on the previous one's lessons. The jump from Box2D to custom physics (004->005) was the riskiest transition and required the most debugging.

### The agent will always surprise you
No amount of theoretical analysis substitutes for watching the agent play. The steering wiggle exploit, the stationary optimization, the overly conservative driving -- these were only discovered by visual inspection.

---

## Summary: If Starting Over

If starting a sim racing RL project from scratch with these lessons:

1. **Use SAC** (or TD3) for continuous control from the start. Don't waste time discretizing.
2. **Use vector state** with track geometry and lookahead waypoints. Add frame stacking.
3. **Own your physics engine** if you plan to iterate on reward/observation design.
4. **Combine sparse + dense rewards.** Checkpoints for milestones, velocity for progress signal, time penalty for urgency.
5. **Validate collision detection** independently before trusting penalty signals.
6. **Use parallel selection training** on multi-core CPUs for vector mode.
7. **Log everything.** CSV metrics, human-readable logs, system config.
8. **Start with no domain randomization**, then add it via curriculum.
9. **Watch the agent play.** Often. It will find exploits you didn't anticipate.
10. **Profile before optimizing.** The bottleneck is rarely where you think it is.
