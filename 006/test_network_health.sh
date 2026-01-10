#!/bin/bash
# Quick test script for network health analysis

echo "Testing Network Health Analyzer..."
echo "=================================="
echo ""

# Check if weightwatcher is installed
echo "1. Checking weightwatcher installation..."
python -c "import weightwatcher; print(f'   ✓ WeightWatcher version: {weightwatcher.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "   ❌ WeightWatcher not installed"
    echo "   Installing now..."
    pip install weightwatcher
fi

echo ""
echo "2. Finding test checkpoint..."
CHECKPOINT=$(find checkpoints_selection_parallel -name "*.pt" -type f | head -1)

if [ -z "$CHECKPOINT" ]; then
    echo "   ❌ No checkpoint files found in checkpoints_selection_parallel/"
    echo "   Please train a model first with:"
    echo "   python train_selection_parallel.py --num-agents 4 --episodes 100"
    exit 1
fi

echo "   ✓ Found checkpoint: $CHECKPOINT"
echo ""

echo "3. Running health analysis..."
python analyze_network_health.py --checkpoint "$CHECKPOINT" --networks actor critic_1

echo ""
echo "=================================="
echo "Test complete!"
echo ""
echo "Try these commands:"
echo "  # Analyze specific checkpoint:"
echo "  python analyze_network_health.py --checkpoint $CHECKPOINT"
echo ""
echo "  # Generate full report with plots:"
echo "  python analyze_network_health.py --checkpoint $CHECKPOINT --output-dir health_reports/"
echo ""
echo "  # Compare multiple checkpoints:"
echo "  python analyze_network_health.py --checkpoint checkpoints_selection_parallel/*.pt"
