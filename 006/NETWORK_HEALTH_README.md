# Network Health Analysis - Quick Start

## What I Created

I've created a complete network health analysis system using the **WeightWatcher** package to diagnose your SAC networks:

### New Files

1. **`analyze_network_health.py`** - Main analysis script
   - Loads SAC checkpoints
   - Analyzes actor and critic networks
   - Generates health metrics and reports
   - Creates visualizations

2. **`NETWORK_HEALTH_GUIDE.md`** - Comprehensive documentation
   - Detailed explanation of all metrics
   - Interpretation guidelines
   - Usage examples and workflows
   - Troubleshooting tips

3. **`test_network_health.sh`** - Quick test script
   - Checks installation
   - Runs basic analysis
   - Verifies everything works

4. **Updated `requirements.txt`** - Added weightwatcher dependency

5. **Updated `CLAUDE.md`** - Documentation integrated

## Installation

```bash
# Activate virtual environment
source ../.venv/bin/activate

# Install weightwatcher (you may need to fix permissions first)
pip install weightwatcher
```

**If you get permission errors**, try:
```bash
# Option 1: Install to user directory
pip install --user weightwatcher

# Option 2: Fix venv permissions and reinstall
rm -rf ../.venv/lib/python3.13/site-packages/pytz
pip install --no-cache-dir weightwatcher
```

## Quick Usage

### Basic Analysis
```bash
python analyze_network_health.py --checkpoint checkpoints_selection_parallel/generation_5.pt
```

### With Visualizations
```bash
python analyze_network_health.py \
    --checkpoint checkpoints_selection_parallel/generation_5.pt \
    --output-dir health_reports/
```

### Compare Training Progress
```bash
python analyze_network_health.py --checkpoint \
    checkpoints_selection_parallel/generation_1.pt \
    checkpoints_selection_parallel/generation_5.pt \
    checkpoints_selection_parallel/generation_10.pt
```

## What It Does

The analyzer uses **Random Matrix Theory** to assess network quality:

### Key Metric: Alpha (α)

**Most Important Metric** - Indicates how well your network will generalize ([WeightWatcher.ai](https://weightwatcher.ai)):

- **α ≈ 2.0**: ✅ **OPTIMAL** - Best trained models, target this value!
- **α ∈ [2, 5]**: ✅ **RECOMMENDED** - Well-trained with good generalization
- **α ∈ (5, 6]**: ⚠️ Undertrained - needs more training/optimization
- **α > 6**: ❌ Severely undertrained/random - network hasn't learned useful patterns
- **α < 2**: ❌ Overfit - Early stopping signal, memorizing training data

**Key principle**: "Smaller is better" within the 2-6 range, with α≈2.0 being the optimal value.

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

### Output Files

When using `--output-dir`, generates:

1. **Text Reports**:
   - `network_health_report.txt` - Full analysis including dead neurons
   - `actor_details.csv` - Per-layer WeightWatcher metrics
   - `critic_1_details.csv` - Per-layer WeightWatcher metrics
   - `actor_dead_neurons.csv` - Per-layer dead neuron statistics
   - `critic_1_dead_neurons.csv` - Per-layer dead neuron statistics

2. **Visualizations**:
   - `actor_health_metrics.png` - 4-panel diagnostic plot
   - `critic_1_health_metrics.png` - 4-panel diagnostic plot

## Use Cases

### 1. Diagnose Training Issues

**Problem**: "My agent isn't learning after 500 episodes"

**Solution**:
```bash
python analyze_network_health.py --checkpoint checkpoints_selection_parallel/latest_generation.pt
```

**Interpretation**:
- If α > 6.0: Network is severely undertrained, continue training or adjust hyperparameters
- If α ≈ 2.0: Optimal! This checkpoint has best generalization
- If α < 2.0: Network is overfit, use this or earlier checkpoint (early stopping signal)

### 2. Find Best Checkpoint

**Problem**: "Which checkpoint has best generalization?"

**Solution**:
```bash
python analyze_network_health.py --checkpoint checkpoints_selection_parallel/*.pt
```

**Result**: Compare alpha values, choose checkpoint closest to α = 2.0 (optimal)

### 3. Monitor Training

**Problem**: "Is my training getting worse?"

**Solution**: Check alpha over time
- α decreasing from 6.0 → 2.5 = ✅ Good, learning useful features
- α decreasing from 2.5 → 2.0 = ✅ Excellent, approaching optimal
- α decreasing from 2.0 → 1.5 = ⚠️ Overtraining, implement early stopping

### 4. Compare Network Architectures

**Problem**: "Should I use a bigger network?"

**Solution**: Train both, compare alpha values
- Similar alpha with smaller network = smaller network is sufficient
- Much higher alpha with smaller network = smaller network is undertrained, need more capacity

## Integration with Training

You can add periodic health checks to your training:

```python
# In train_selection_parallel.py, after saving checkpoint:
if generation % 10 == 0:  # Every 10 generations
    os.system(f"python analyze_network_health.py --checkpoint {checkpoint_path}")
```

## Scientific Background

WeightWatcher is based on research by Martin & Mahoney (UC Berkeley):

- Uses Random Matrix Theory (RMT) to analyze weight matrices
- Computes power law exponents (alpha) of eigenvalue distributions
- Validated on ImageNet, BERT, GPT models
- No test data required - analyzes network structure directly

**Key Papers**:
- "Traditional and Heavy-Tailed Self Regularization" (NeurIPS 2019)
- "Implicit Self-Regularization in Deep Neural Networks" (2021)

## Troubleshooting

### "weightwatcher not installed"
```bash
pip install weightwatcher
```

### "matplotlib not available"
Install matplotlib (optional, for plots):
```bash
pip install matplotlib
```

### Very slow analysis
Analyze fewer networks:
```bash
python analyze_network_health.py --checkpoint best.pt --networks actor
```

### Permission errors during pip install
See installation section above for alternatives.

## Next Steps

1. **Install weightwatcher**: `pip install weightwatcher`
2. **Run test**: `bash test_network_health.sh`
3. **Analyze your checkpoints**: Try the examples above
4. **Read detailed guide**: See `NETWORK_HEALTH_GUIDE.md`

## Summary

This tool helps you:
- ✅ Know when to stop training (α < 2.0 = overfit, early stopping signal)
- ✅ Know when to train longer (α > 6.0 = severely undertrained)
- ✅ Choose best checkpoint (α ≈ 2.0 is optimal, [2, 5] is good)
- ✅ Debug why agent isn't learning
- ✅ Compare different training runs
- ✅ Optimize hyperparameters

**No test data or evaluation episodes required** - analyzes network weights directly using mathematical theory!

---

For detailed documentation, see **NETWORK_HEALTH_GUIDE.md**
