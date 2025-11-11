# Training Script Comparison

This document compares the three training scripts available for CarRacing-v3 with SAC.

## Quick Summary

| Script | Parallelism | Memory | CPU Usage | Best For |
|--------|-------------|--------|-----------|----------|
| `train.py` | None | ~50 MB | 120% | Baseline, debugging |
| `train_multicar.py` | Sequential | ~50 MB | 120% | Data efficiency (N× data) |
| `train_vectorenv.py` | True parallel | ~200 MB | 400%+ | Speed (4× CPU parallelism) |

## 1. train.py (Baseline Single-Car)

**Architecture:**
- Single environment, single car
- Traditional RL training loop
- Sequential execution

**Key Characteristics:**
```python
env = make_carracing_env(num_cars=1)  # Single car
obs, _ = env.reset()
action = agent.select_action(obs)
next_obs, reward, terminated, truncated, _ = env.step(action)
```

**Performance:**
- Memory: ~50 MB
- CPU usage: 120% (1 core + overhead)
- Data collection: 1× baseline
- Wall-clock time: baseline

**Use when:**
- Learning the codebase
- Debugging issues
- Comparing baseline performance
- Testing new reward functions

---

## 2. train_multicar.py (Multi-Car on Same Track)

**Architecture:**
- Single environment with N cars racing as ghost cars
- All cars on the **same track** (same physics simulation)
- Sequential execution (Python loop)
- Natural selection: best car identified each episode

**Key Characteristics:**
```python
env = make_carracing_env(num_cars=4)  # 4 ghost cars
obs, _ = env.reset()  # shape: (4, 67)

# All cars get actions from same policy
actions = np.array([agent.select_action(obs[i]) for i in range(4)])

# All cars step together on SAME track
next_obs, rewards, terminated, truncated, infos = env.step(actions)

# Select best performer
best_car_idx = np.argmax(rewards)
```

**Performance:**
- Memory: ~50 MB (single track, minimal overhead)
- CPU usage: 120% (still sequential in Python)
- Data collection: **4× faster** (4 cars = 4× experiences per episode)
- Wall-clock time: ~3× slower per episode (4 cars to simulate)
- **Net effect: ~1.33× data per second**

**Benefits:**
- **Fair competition**: All cars race on same track
- **Data efficiency**: 4× more experiences per episode
- **Natural selection**: Evolutionary approach (best car wins)
- **Minimal memory**: ~300 bytes per car overhead

**Limitations:**
- No true parallelism (Python GIL)
- Sequential execution in loop
- CPU usage same as single car (~120%)

**Use when:**
- You want 4× data collection efficiency
- Memory is limited
- You want fair competition (same track)
- You want natural selection mechanism
- CPU parallelism not critical

---

## 3. train_vectorenv.py (True Parallel with Synchronized Seeds)

**Architecture:**
- N separate environments (AsyncVectorEnv)
- Each environment runs in separate process
- **Synchronized seeds** → all envs generate **same track**
- True parallel execution

**Key Characteristics:**
```python
# Create N parallel environments
vec_env = AsyncVectorEnv([make_env_fn() for _ in range(4)])

# Generate episode seed
episode_seed = 1000 + episode

# Reset all envs with SAME seed (synchronized track)
obs, _ = vec_env.reset(seed=[episode_seed] * 4)

# VectorEnv API (vectorized)
actions = np.array([agent.select_action(obs[i]) for i in range(4)])
next_obs, rewards, terminated, truncated, infos = vec_env.step(actions)

# Select best performer
best_env_idx = np.argmax(rewards)
```

**Performance:**
- Memory: **~200 MB** (4 separate processes, each with own track)
- CPU usage: **400%+** (true 4× parallelism)
- Data collection: 4× faster (parallel execution)
- Wall-clock time: **Potentially 2-3× faster** (parallel speedup)
- **Net effect: ~2-3× faster wall-clock training**

**Benefits:**
- **TRUE PARALLEL EXECUTION**: 4× CPU cores utilized
- **Fair competition**: Synchronized seeds = same track
- **Fastest wall-clock training**: 2-3× speedup expected
- **Natural selection**: Evolutionary approach (best env wins)

**Tradeoffs:**
- **4× memory usage** (~200 MB total for 4 envs)
- More complex setup (VectorEnv, multiprocessing)
- Slightly more overhead (IPC between processes)

**Use when:**
- You have CPU cores available (4+ cores recommended)
- Memory is not a constraint (~200 MB is fine)
- You want **fastest wall-clock training time**
- You want fair competition (synchronized seeds)

---

## Detailed Comparison

### Data Collection Efficiency

| Script | Experiences/Episode | Wall-Clock/Episode | Experiences/Second |
|--------|--------------------:|-------------------:|-------------------:|
| train.py | 100 | 1.0× | 100 |
| train_multicar.py | 400 (4×) | 3.0× | 133 (1.33×) |
| train_vectorenv.py | 400 (4×) | 1.2× | 333 (3.33×) |

### Resource Usage

| Script | Memory | CPU Cores | GPU Usage | Process Count |
|--------|--------|-----------|-----------|---------------|
| train.py | 50 MB | 1.2 | Optional | 1 |
| train_multicar.py | 50 MB | 1.2 | Optional | 1 |
| train_vectorenv.py | 200 MB | 4+ | Optional | 5 (main + 4 workers) |

### Competition Fairness

| Script | Same Track? | How? |
|--------|-------------|------|
| train.py | N/A | Single car |
| train_multicar.py | ✓ YES | Same physics simulation |
| train_vectorenv.py | ✓ YES | Synchronized seeds |

### When to Use Each

**Use train.py if:**
- Learning the codebase
- Debugging
- Minimal resource usage
- Baseline comparison

**Use train_multicar.py if:**
- Memory is limited (<200 MB available)
- CPU cores are limited (1-2 cores)
- You want data efficiency without parallelism
- You want natural selection with minimal overhead

**Use train_vectorenv.py if:**
- You have 4+ CPU cores
- Memory is not a constraint (200 MB is fine)
- You want **fastest training** (2-3× wall-clock speedup)
- You want both parallelism AND fair competition

---

## Example Usage

### train.py (Baseline)
```bash
python train.py --episodes 2000 --device cpu
```

### train_multicar.py (Data Efficiency)
```bash
# 4 cars on same track (4× data collection)
python train_multicar.py --num-cars 4 --episodes 2000

# 8 cars for even more data
python train_multicar.py --num-cars 8 --episodes 1000
```

### train_vectorenv.py (True Parallelism)
```bash
# 4 parallel environments with synchronized seeds
python train_vectorenv.py --num-envs 4 --episodes 2000

# 8 parallel environments (requires 8+ CPU cores)
python train_vectorenv.py --num-envs 8 --episodes 1000
```

---

## Expected Training Time (2000 Episodes)

Assuming 4 cars/environments:

| Script | Wall-Clock Time | Speedup |
|--------|-----------------|---------|
| train.py | 10 hours | 1.0× |
| train_multicar.py | 30 hours | 0.33× (slower!) |
| train_vectorenv.py | **4-5 hours** | **2-2.5× faster** |

**Note:** train_multicar.py is slower in wall-clock time but collects 4× more data, which may lead to better learning.

---

## Memory Usage Details

### train_multicar.py (50 MB)
```
Environment:        50 MB
├─ Track:           50 MB (shared)
├─ Car 0:           0.3 KB
├─ Car 1:           0.3 KB
├─ Car 2:           0.3 KB
└─ Car 3:           0.3 KB
Total:              ~50 MB
```

### train_vectorenv.py (200 MB)
```
Main Process:       50 MB
Worker 1:           50 MB (separate track)
Worker 2:           50 MB (separate track)
Worker 3:           50 MB (separate track)
Worker 4:           50 MB (separate track)
Total:              ~200 MB
```

---

## Recommendation

For most use cases with modern hardware (4+ cores, 500+ MB RAM):

**→ Use `train_vectorenv.py` for fastest training**

It provides:
- True parallel execution (4× CPU usage)
- Fair competition (synchronized seeds)
- Fastest wall-clock time (~2-3× speedup)
- Natural selection (best performer identified)

The memory tradeoff (200 MB vs 50 MB) is negligible on modern systems.

---

## Testing

To verify synchronized seeds work:
```bash
# Run train_vectorenv.py with verbose mode
python train_vectorenv.py --num-envs 4 --episodes 1 --verbose

# Should see: "Seed: 1000" in output
# All 4 environments will generate identical tracks
```

---

## Future Improvements

Possible enhancements:
1. **Hybrid approach**: VectorEnv where each env has num_cars > 1
   - Would give N_envs × N_cars parallelism
   - Example: 4 envs × 4 cars = 16× data collection
   - But memory would be 4× and complexity higher

2. **Adaptive seed strategy**:
   - Use same seed for first K episodes (exploration)
   - Then randomize seeds (generalization)

3. **SyncVectorEnv option**:
   - For debugging (easier to trace)
   - Slightly less overhead than AsyncVectorEnv
