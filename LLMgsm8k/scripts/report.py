#!/usr/bin/env python3
"""Print accuracy report from evaluated JSONL."""

import json
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from config import Config


def report(config: Config) -> None:
    records = []
    with open(config.EVALUATED_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("No evaluated records found.")
        return

    # Numeric match (extraction-based)
    numeric_correct = sum(1 for r in records if r.get("correct"))
    total = len(records)

    print(f"=== GSM8K Evaluation Report ===")
    print(f"Total records: {total}")
    print(f"\nNumeric match (extracted == ground truth): {numeric_correct}/{total} ({100*numeric_correct/total:.1f}%)")

    # Per-evaluator breakdown
    for eval_key, label in [("self_eval", "Self-eval"), ("smart_eval", "Smart-eval")]:
        rated = [r for r in records if eval_key in r]
        if not rated:
            print(f"\n{label}: no data")
            continue

        counts = {"CORRECT": 0, "INCORRECT": 0, "UNSURE": 0}
        for r in rated:
            rating = r[eval_key]["rating"]
            counts[rating] = counts.get(rating, 0) + 1

        print(f"\n{label} ({len(rated)} rated):")
        for k in ("CORRECT", "INCORRECT", "UNSURE"):
            pct = 100 * counts[k] / len(rated) if rated else 0
            print(f"  {k}: {counts[k]} ({pct:.1f}%)")

    # Agreement
    both = [r for r in records if "self_eval" in r and "smart_eval" in r]
    if both:
        agree = sum(1 for r in both if r["self_eval"]["rating"] == r["smart_eval"]["rating"])
        print(f"\nAgreement (self vs smart): {agree}/{len(both)} ({100*agree/len(both):.1f}%)")

    # Per-split breakdown
    splits = sorted(set(r.get("split", "unknown") for r in records))
    if len(splits) > 1:
        print(f"\n--- Per-split breakdown ---")
        for split in splits:
            subset = [r for r in records if r.get("split") == split]
            nc = sum(1 for r in subset if r.get("correct"))
            print(f"  {split}: {nc}/{len(subset)} numeric match ({100*nc/len(subset):.1f}%)")

    # Disagreement breakdown: regex vs judges
    for eval_key, label in [("self_eval", "Self-eval"), ("smart_eval", "Smart-eval")]:
        rated = [r for r in records if eval_key in r]
        if not rated:
            continue

        # regex correct + judge incorrect, etc.
        combos = {"C/C": 0, "C/I": 0, "C/U": 0, "I/C": 0, "I/I": 0, "I/U": 0}
        for r in rated:
            regex = "C" if r.get("correct") else "I"
            judge = r[eval_key]["rating"][0]  # C, I, or U
            key = f"{regex}/{judge}"
            combos[key] = combos.get(key, 0) + 1

        print(f"\nRegex vs {label} (regex/judge):")
        for key in ("C/C", "C/I", "C/U", "I/C", "I/I", "I/U"):
            if combos[key]:
                print(f"  {key}: {combos[key]}")
        disagree = combos["C/I"] + combos["I/C"]
        print(f"  Disagree: {disagree} ({100*disagree/len(rated):.1f}%)")

    # Extraction method breakdown
    methods = {}
    for r in records:
        m = r.get("extraction_method", "unknown")
        methods[m] = methods.get(m, 0) + 1
    print(f"\nExtraction methods:")
    for m, c in sorted(methods.items(), key=lambda x: -x[1]):
        print(f"  {m}: {c} ({100*c/total:.1f}%)")


if __name__ == "__main__":
    report(Config())
