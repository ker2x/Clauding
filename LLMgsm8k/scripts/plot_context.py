#!/usr/bin/env python3
"""Plot accuracy vs context size from bench_context results."""

import json
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import matplotlib.pyplot as plt

INPUT = sys.argv[1] if len(sys.argv) > 1 else "data/bench_context.jsonl"

results = []
with open(INPUT) as f:
    for line in f:
        if line.strip():
            results.append(json.loads(line))

sizes = [r["max_tokens"] for r in results]
accuracy = [100 * r["accuracy"] for r in results]
boxed_pct = [100 * r["extraction_methods"].get("boxed", 0) / r["answered"] for r in results]
fallback_pct = [100 * r["extraction_methods"].get("fallback", 0) / r["answered"] for r in results]
avg_tokens = [r.get("avg_completion_tokens", 0) for r in results]
has_tokens = any(t > 0 for t in avg_tokens)

fig, ax1 = plt.subplots(figsize=(10, 6))

# Accuracy
color1 = "#2563eb"
ax1.plot(sizes, accuracy, "o-", color=color1, linewidth=2.5, markersize=8, label="Accuracy")
ax1.set_xlabel("Max Tokens (Context Size)", fontsize=13)
ax1.set_ylabel("Accuracy (%)", fontsize=13, color=color1)
ax1.tick_params(axis="y", labelcolor=color1)
ax1.set_ylim(min(accuracy) - 5, max(accuracy) + 5)
ax1.set_xticks(sizes)

# Annotate accuracy values
for x, y in zip(sizes, accuracy):
    ax1.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                 xytext=(0, 12), ha="center", fontsize=10, color=color1)

# Secondary axis: extraction rates + avg tokens
ax2 = ax1.twinx()
color2 = "#dc2626"
ax2.plot(sizes, boxed_pct, "s--", color=color2, linewidth=1.5, markersize=6, alpha=0.7, label="\\boxed{} rate")
ax2.plot(sizes, fallback_pct, "^--", color="#f59e0b", linewidth=1.5, markersize=6, alpha=0.7, label="Fallback rate")
if has_tokens:
    ax2.plot(sizes, avg_tokens, "D--", color="#22c55e", linewidth=1.5, markersize=6, alpha=0.7, label="Avg tokens")
    for x, y in zip(sizes, avg_tokens):
        if y > 0:
            ax2.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                         xytext=(0, -14), ha="center", fontsize=9, color="#22c55e")
ax2.set_ylabel("Rate (%) / Tokens", fontsize=13, color=color2)
ax2.tick_params(axis="y", labelcolor=color2)

ax2.tick_params(axis="y", labelcolor=color2)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=11)

ax1.set_title("LFM2.5-Thinking GSM8K Performance vs Context Size", fontsize=14, fontweight="bold")
ax1.grid(True, alpha=0.3)

plt.tight_layout()
out = INPUT.rsplit(".", 1)[0] + ".png"
plt.savefig(out, dpi=150)
print(f"Saved to {out}")
plt.show()
