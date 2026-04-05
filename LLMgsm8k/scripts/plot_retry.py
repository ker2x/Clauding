#!/usr/bin/env python3
"""Plot cumulative accuracy vs retry attempts from bench_retry results."""

import json
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import matplotlib.pyplot as plt

INPUT = sys.argv[1] if len(sys.argv) > 1 else "data/bench_retry.jsonl"

results = []
with open(INPUT) as f:
    for line in f:
        if line.strip():
            results.append(json.loads(line))

total = len(results)
max_attempts = max(r["total_attempts"] for r in results)

attempts = list(range(1, max_attempts + 1))
attempt_labels = ["Initial"] + [f"Retry {i}" for i in range(1, max_attempts)]
cumulative = []
new_correct = []
prev = 0
for i in attempts:
    solved = sum(1 for r in results if r["correct_at"] is not None and r["correct_at"] <= i)
    cumulative.append(100 * solved / total)
    new_correct.append(solved - prev)
    prev = solved

fig, ax1 = plt.subplots(figsize=(10, 6))

# Cumulative accuracy
color1 = "#2563eb"
ax1.plot(attempts, cumulative, "o-", color=color1, linewidth=2.5, markersize=10, label="Cumulative accuracy")
ax1.set_xlabel("Attempt", fontsize=13)
ax1.set_ylabel("Cumulative Accuracy (%)", fontsize=13, color=color1)
ax1.tick_params(axis="y", labelcolor=color1)
ax1.set_ylim(min(cumulative) - 3, max(cumulative) + 3)
ax1.set_xticks(attempts)
ax1.set_xticklabels(attempt_labels)

for x, y in zip(attempts, cumulative):
    ax1.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                 xytext=(0, 14), ha="center", fontsize=11, color=color1)

# Newly solved per retry (skip initial — it's the baseline, not a retry)
ax2 = ax1.twinx()
color2 = "#22c55e"
retry_x = attempts[1:]
retry_new = new_correct[1:]
ax2.bar(retry_x, retry_new, alpha=0.3, color=color2, width=0.4, label="Newly solved by retry")
ax2.set_ylabel("Newly Solved Questions", fontsize=13, color=color2)
ax2.tick_params(axis="y", labelcolor=color2)

for x, y in zip(retry_x, retry_new):
    if y > 0:
        ax2.annotate(f"+{y}", (x, y), textcoords="offset points",
                     xytext=(0, 5), ha="center", fontsize=10, color=color2)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=11)

ax1.set_title("LFM2.5-Thinking GSM8K Self-Correction Performance", fontsize=14, fontweight="bold")
ax1.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
out = INPUT.rsplit(".", 1)[0] + ".png"
plt.savefig(out, dpi=150)
print(f"Saved to {out}")
plt.show()
