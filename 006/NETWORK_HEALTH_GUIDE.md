# Network Health Analysis Guide

This guide explains how to use the `analyze_network_health.py` script to diagnose and understand the quality of your trained SAC networks.

## Overview

The network health analyzer uses the **WeightWatcher** package, which applies Random Matrix Theory (RMT) and other theoretical frameworks to assess neural network quality without requiring test data.

### Key Metrics

#### 1. Alpha (α) - Power Law Exponent
The most important metric for assessing generalization (from [WeightWatcher.ai](https://weightwatcher.ai)):

- **α ≈ 2.0**: **OPTIMAL** - Best trained models, ideal target
- **α ∈ [2, 5]**: **RECOMMENDED** - Well-trained with good generalization
- **α ∈ (5, 6]**: Undertrained - needs more training/optimization
- **α > 6**: Severely undertrained or random weights
- **α < 2**: Overfit - Network has memorized training data, early stopping needed

**Key principle**: "Smaller is better" within the valid 2-6 range, with α≈2.0 being the optimal value.

**What it means**: Alpha measures how the eigenvalues of weight matrices are distributed according to power-law behavior. Higher alpha values (>6) indicate random-like untrained weights, while very low values (<2) indicate overfitting. The best performing models have all layers approaching α≈2.0.

#### 2. Stable Rank
Measures the **effective dimensionality** of weight matrices - how much of your network's capacity is actually being used.

**What it measures**: The number of dimensions that meaningfully contribute to the layer's computation.

**Formula** (conceptual): `stable_rank = (Frobenius norm)² / (spectral norm)²`

**Interpretation**:
- **Full rank**: Would be `min(rows, columns)` of weight matrix
- **Stable rank**: Between 1 and full rank, shows "effective" dimensions
- **Higher stable rank** (closer to full rank): Network using most of its capacity
  - Different neurons learning diverse features
  - Parameters are being utilized efficiently
  - Good sign - no wasted capacity
- **Lower stable rank** (much smaller than full rank): Redundancy or correlation
  - Many weights are redundant or correlated
  - Network might be over-parameterized
  - Could potentially use a smaller network

**Practical Examples**:

For a layer with 256 neurons (256×256 weight matrix, full rank = 256):
- **Stable rank = 200**: Good! Using ~78% of capacity
- **Stable rank = 128**: Moderate, using ~50% of capacity
- **Stable rank = 50**: Low! Only ~20% contributing meaningfully
- **Stable rank = 10**: Very low! Highly redundant, consider smaller layer

For a layer with 512 neurons (512×512 weight matrix, full rank = 512):
- **Stable rank = 400**: Excellent! Using ~78% of capacity
- **Stable rank = 100**: Low! Only ~20% of neurons contributing

**When to care about stable rank**:
1. **Very low values** (< 20% of full rank) → Network may be too big, wasting parameters
2. **Decreasing over training** → May indicate collapse or saturation
3. **Big differences between layers** → Architectural imbalance
4. **Comparing architectures** → Helps decide if larger network is justified

**For SAC networks**: Stable rank is **secondary to alpha**. Focus on getting alpha ≈ 2.0 first. Use stable rank as a diagnostic to understand if your network architecture is appropriately sized.

**In your reports**: When you see `Stable rank mean: 45.3`, check this against your layer sizes:
- For 256-neuron layers: 45 is ~18% (low, possibly over-parameterized)
- For 128-neuron layers: 45 is ~35% (moderate)
- For 64-neuron layers: 45 is ~70% (good utilization)

**Typical healthy ranges**: 30-80% of full rank indicates good capacity utilization without severe redundancy.

#### 3. Log Spectral Norm
Measures the largest singular value of each layer:

- **Lower is better**: Indicates better conditioning
- **Very high values**: May indicate instability or exploding gradients
- **Use case**: Helps identify layers that might benefit from regularization

#### 4. Dead Neurons
Counts neurons that contribute negligibly to network output.

**What it measures** (activation-based, default): Neurons where the activation variance across sample inputs is below a threshold (default: 1e-6). This works for **any activation function** (ReLU, Leaky ReLU, GELU, etc.).

**Alternative method** (weight-based): Neurons where the L2 norm of incoming weights is below threshold. Simpler but less accurate for networks with Leaky ReLU or other non-zero activations.

**Why they occur**:
- **Gradient flow issues**: Some neurons never receive meaningful gradients during training
- **Over-parameterization**: Network too large for the task complexity
- **Poor initialization**: Some neurons start unfavorably and never recover
- **Saturation**: Neurons stuck in flat regions of activation functions

**Interpretation**:
- **< 5% dead**: ✅ **GOOD** - Healthy network utilization
- **5-15% dead**: ⚠️ **MODERATE** - Some wasted capacity, consider monitoring
- **15-30% dead**: ⚠️ **HIGH** - Significant waste, consider smaller network or pruning
- **> 30% dead**: ❌ **CRITICAL** - Major capacity waste, network likely over-parameterized

**What to do about dead neurons**:

1. **< 5% dead**: Nothing - this is normal and healthy
2. **5-15% dead**:
   - Monitor during training to see if it worsens
   - No immediate action needed if performance is good
3. **15-30% dead**:
   - Consider using a smaller network for future training
   - Could apply network pruning to remove dead neurons
   - Check if learning rate or initialization could be improved
4. **> 30% dead**:
   - **Strong signal** that network is over-parameterized
   - Try training with 50% fewer neurons per layer
   - Review network architecture - may not need so many layers/neurons

**Relationship to other metrics**:
- **High dead neurons + low stable rank**: Strong evidence of over-parameterization
- **High dead neurons + good alpha (≈2.0)**: Network can still generalize well despite waste
- **High dead neurons + poor alpha (>6)**: Network both too large and undertrained

**Example**: If you have a 512-neuron layer with 200 dead neurons (39% dead), you're effectively using only ~312 neurons. A 256-neuron network might perform similarly with less computation.

## Installation

```bash
# Activate virtual environment
source ../.venv/bin/activate

# Install weightwatcher
pip install weightwatcher

# Or install from requirements
pip install -r requirements.txt
```

## Usage Examples

### Basic Analysis

Analyze a single checkpoint (uses activation-based dead neuron detection by default):

```bash
python analyze_network_health.py --checkpoint checkpoints_selection_parallel/best_model.pt
```

This will:
- Load the checkpoint
- Analyze all networks (actor, critic_1, critic_2, targets)
- Print summary statistics including dead neurons
- Show health diagnostics

### Dead Neuron Detection Methods

**Activation-based (default, recommended for Leaky ReLU networks)**:
```bash
# Default: 1000 samples
python analyze_network_health.py --checkpoint best_model.pt

# More samples for higher accuracy
python analyze_network_health.py --checkpoint best_model.pt --dead-neuron-samples 5000

# Fewer samples for faster analysis
python analyze_network_health.py --checkpoint best_model.pt --dead-neuron-samples 500
```

**Weight-based (simpler, faster, but less accurate for non-ReLU)**:
```bash
python analyze_network_health.py --checkpoint best_model.pt --dead-neuron-method weight
```

**When to use each**:
- **Activation-based**: Networks with Leaky ReLU, GELU, or any non-standard activation (most accurate)
- **Weight-based**: Quick analysis, or networks with standard ReLU (faster, less accurate)

### Save Detailed Reports

Generate reports and visualizations:

```bash
python analyze_network_health.py \
    --checkpoint checkpoints_selection_parallel/best_model.pt \
    --output-dir health_reports/
```

This creates:
- `network_health_report.txt`: Detailed text report
- `actor_details.csv`: Layer-by-layer metrics for actor
- `critic_1_details.csv`: Layer-by-layer metrics for critic_1
- `actor_health_metrics.png`: Visualization plots
- `critic_1_health_metrics.png`: Visualization plots

### Compare Multiple Checkpoints

Track network health over training:

```bash
python analyze_network_health.py \
    --checkpoint \
        checkpoints_selection_parallel/generation_100.pt \
        checkpoints_selection_parallel/generation_200.pt \
        checkpoints_selection_parallel/generation_300.pt \
        checkpoints_selection_parallel/best_model.pt
```

This compares alpha metrics across checkpoints to see if training is improving or degrading.

### Analyze Specific Networks

Only analyze actor and primary critic:

```bash
python analyze_network_health.py \
    --checkpoint checkpoints_selection_parallel/best_model.pt \
    --networks actor critic_1
```

## Interpreting Results

### Example Output

```
📊 Overall Metrics:
   Alpha (generalization): 2.847
   Health: ✅ GOOD (well-trained, should generalize)

💀 Dead Neurons (activation_based):
   Count: 12/1536
   Ratio: 0.78%
   Status: ✅ GOOD (<5% dead - healthy utilization)

📈 Layer Statistics:
   Alpha range: [2.134, 3.521]
   Alpha mean: 2.847 ± 0.412
   Log spectral norm mean: 1.234
   Stable rank mean: 45.3
   Total layers analyzed: 8

⚠️  Potential Issues:
   ✓ No obvious issues detected
```

### What to Look For

#### Healthy Network (Good Training)
- Alpha mean in [2.0, 5.0], ideally close to 2.0
- Most layers have alpha in [2.0, 5.0]
- Low standard deviation in alpha values
- Reasonable spectral norms (< 2.0)

#### Undertrained Network
- Alpha mean > 5.0
- Many layers with alpha > 6.0 (approaching random)
- High variance in metrics
- **Action**: Train longer, increase learning rate, or improve optimization

#### Overfitting Network
- Alpha mean < 2.0
- Multiple layers with very low alpha
- May have very high spectral norms
- **Action**: Implement early stopping, add regularization, or reduce network size

#### Unstable Training
- Large variance in alpha across layers
- Very high spectral norms (> 5.0)
- Inconsistent metrics
- **Action**: Reduce learning rate, increase batch size, check gradients

## Practical Workflow

### During Training

1. **Early stages (< 100 episodes)**:
   - Networks are still random
   - Alpha will be very high (> 6.0, approaching 8 for random matrices)

2. **Mid training (100-500 episodes)**:
   - Check if alpha is decreasing into [2.0, 5.0] range
   - Look for stability in metrics
   - Target: alpha approaching 2.0-3.0 for best performance
   - If alpha not decreasing, may need to adjust hyperparameters

3. **Late training (500+ episodes)**:
   - Monitor for overtraining (alpha < 2.0)
   - Optimal performance typically at alpha ≈ 2.0
   - Compare current checkpoint with best checkpoint
   - If alpha decreasing below 2.0, implement early stopping immediately

### Debugging Poor Performance

If your agent performs poorly, check:

1. **Alpha too high** (> 5.0, especially > 6.0):
   - Network hasn't learned meaningful features (still random-like)
   - Increase training time, adjust learning rate, or improve hyperparameters
   - Check if loss is decreasing
   - For alpha > 6: May need to reduce layer size or add more data

2. **Alpha too low** (< 2.0):
   - Network is overfit to training data
   - **This is an early stopping signal** - use current or previous checkpoint
   - Consider adding regularization for future training
   - Use earlier checkpoint with alpha closer to 2.0

3. **High variance across layers**:
   - Some layers learning, others not
   - May indicate gradient flow issues
   - Check layer normalization is working

4. **Critics vs Actor mismatch**:
   - If actor alpha >> critic alpha: Policy not learning as fast as value function
   - If actor alpha << critic alpha: Value estimation lagging behind policy
   - Adjust learning rates to balance network training speeds

## Common Patterns

### Pattern 1: Healthy Progression
```
Generation 100: Alpha = 6.5 (starting to learn, decreasing from ~8)
Generation 200: Alpha = 3.8 (good, learning useful features)
Generation 300: Alpha = 2.4 (excellent, near-optimal)
Generation 400: Alpha = 2.1 (optimal - best performance expected)
```
**Interpretation**: Healthy training, generation 400 is optimal (α≈2.0)

### Pattern 2: Overtraining
```
Generation 100: Alpha = 5.2 (learning well)
Generation 200: Alpha = 2.8 (good, approaching optimal)
Generation 300: Alpha = 2.0 (optimal - stop here!)
Generation 400: Alpha = 1.7 (overfit - early stopping missed)
Generation 500: Alpha = 1.3 (severely overfit)
```
**Interpretation**: Use checkpoint from generation 300 (α≈2.0), implement early stopping at α<2.0

### Pattern 3: Not Learning
```
Generation 100: Alpha = 7.8 (very random)
Generation 200: Alpha = 7.5 (barely learning)
Generation 300: Alpha = 7.2 (still random)
Generation 400: Alpha = 6.9 (minimal improvement)
```
**Interpretation**: Network not learning effectively, check:
- Learning rate too low
- Poor hyperparameter choices
- Not enough training steps per episode
- Replay buffer issues
- Network architecture may need adjustment

## Visualization Plots

The script generates four plots for each network:

1. **Alpha Distribution**: Histogram showing how many layers fall into healthy range
2. **Spectral Norm by Layer**: Shows conditioning of each layer
3. **Stable Rank by Layer**: Shows capacity utilization
4. **Alpha by Layer**: Shows which specific layers might be problematic

Green zones indicate healthy ranges.

## Integration with Training

### Periodic Health Checks

Add to your training script:

```python
# After every 100 episodes
if episode % 100 == 0:
    os.system(f"python analyze_network_health.py --checkpoint {checkpoint_path}")
```

### Automatic Early Stopping

Monitor alpha in training:

```python
# After saving checkpoint
analyzer = NetworkHealthAnalyzer(checkpoint_path)
analyzer.load_checkpoint()
results = analyzer.analyze_all_networks(['actor'])

alpha = results['actor']['summary']['alpha']
if alpha < 2.0:
    print("⚠️  Alpha < 2.0, EARLY STOPPING SIGNAL!")
    print("    Network is overfit - use this or previous checkpoint")
    # Stop training or revert to checkpoint with alpha ≈ 2.0
elif alpha > 6.0:
    print("⚠️  Alpha > 6.0, network still random/severely undertrained")
    print("    Continue training or adjust hyperparameters")
elif 2.0 <= alpha <= 2.5:
    print("✅ Alpha ≈ 2.0 - OPTIMAL TRAINING!")
    print("    This is the target range for best generalization")
```

## References

- **Martin & Mahoney (2019)**: "Traditional and Heavy-Tailed Self Regularization in Neural Network Learning"
- **Martin & Mahoney (2021)**: "Implicit Self-Regularization in Deep Neural Networks"
- **WeightWatcher**: https://github.com/CalculatedContent/WeightWatcher
- **Blog post**: https://calculatedcontent.com/2019/12/27/weightwatcher/

## FAQ

**Q: How often should I check network health?**
A: Every 50-100 episodes during active training, or when performance plateaus.

**Q: Which networks should I focus on?**
A: Primarily the `actor` (policy) and `critic_1` (main value network). Target networks will be similar to their source networks.

**Q: My alpha is 7.5 after 500 episodes, is this bad?**
A: Yes, this is severely undertrained (alpha > 6). The network hasn't learned meaningful patterns. Check your hyperparameters, learning rate, and training setup.

**Q: Alpha is 1.5, should I stop training?**
A: Yes! Alpha < 2.0 is an early stopping signal indicating overfitting. Use this checkpoint or an earlier one with alpha ≈ 2.0.

**Q: What's the ideal alpha value to target?**
A: Alpha ≈ 2.0 is optimal according to WeightWatcher theory. Values in [2.0, 2.5] indicate excellent training.

**Q: Can I use this for other algorithms besides SAC?**
A: Yes! WeightWatcher works with any PyTorch neural network. Just load the networks and analyze them.

**Q: What does stable rank tell me, and when should I care?**
A: Stable rank measures effective dimensionality - what percentage of your network's capacity is being used. Check stable rank / layer_size:
- 30-80%: Healthy utilization
- < 20%: Network may be over-parameterized, wasting parameters
- Focus on alpha first; stable rank is secondary diagnostic

**Q: My stable rank is 45 for a 256-neuron layer, is this bad?**
A: That's ~18% utilization (low). Your network might be over-parameterized. However, if alpha ≈ 2.0 and performance is good, don't worry about it. Alpha is more important than stable rank for generalization.

**Q: What are dead neurons and how many is too many?**
A: Dead neurons have near-zero weights and don't contribute to the network. They indicate wasted capacity:
- < 5%: Normal and healthy
- 5-15%: Moderate waste, monitor but okay
- 15-30%: High waste, consider smaller network
- > 30%: Critical waste, network over-parameterized

**Q: I have 25% dead neurons but alpha ≈ 2.0 and good performance. Should I worry?**
A: Your network is working well despite the waste. For current training, no action needed. For future training, try a smaller network - you'll likely get similar performance with less computation and memory.

**Q: Can dead neurons recover during training?**
A: Rarely. Once a neuron's weights approach zero, gradients become tiny and it's unlikely to recover. This is why initialization and learning rate are important early in training.

## Troubleshooting

### Error: "weightwatcher not installed"
```bash
pip install weightwatcher
```

### Error: "matplotlib not available"
Plots will be skipped but analysis will continue. Install with:
```bash
pip install matplotlib
```

### Very slow analysis
WeightWatcher can be slow for large networks. Use `--networks actor critic_1` to analyze fewer networks.

### NaN or inf values
Usually indicates:
- Corrupted checkpoint
- Numerical instability during training
- Very poorly initialized weights

---

*Last updated: 2025-01-22*
