---
description: Run the 8x8 checkers training simulation with visualization
---

# Training Workflow

To run the training process with the integrated visualizer:

1. Ensure no other training sessions are running to avoid `mps` device conflicts.
2. Run the following command:

// turbo
```bash
../.venv/bin/python scripts8x8/train.py --visualize
```

3. If you need to resume from the latest checkpoint:

// turbo
```bash
../.venv/bin/python scripts8x8/train.py --visualize --resume
```
