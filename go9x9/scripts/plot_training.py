#!/usr/bin/env python3
"""
Visualize training progress from logs/training_log.csv.

Usage:
    ../.venv/bin/python scripts/plot_training.py
    ../.venv/bin/python scripts/plot_training.py --log logs/training_log.csv
    ../.venv/bin/python scripts/plot_training.py --watch   # refresh every 60s
"""

import argparse
import time
import sys
import os
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import csv

project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

from config import Config

COLORS = {
    'total':     '#e74c3c',
    'policy':    '#3498db',
    'value':     '#2ecc71',
    'ownership': '#f39c12',
    'game_len':  '#9b59b6',
    'time_sp':   '#1abc9c',
    'time_tr':   '#e67e22',
}


def load_log(path: str) -> dict:
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({k: float(v) for k, v in row.items()})
            except (ValueError, TypeError):
                continue  # skip malformed rows

    if not rows:
        return {}

    keys = list(rows[0].keys())
    data = {k: np.array([r[k] for r in rows]) for k in keys}
    return data


def smooth(y, window=5):
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    padded = np.pad(y, (window // 2, window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(y)]


def plot(data: dict, save_path: str = None):
    iters = data['iteration']
    n = len(iters)

    fig = plt.figure(figsize=(14, 7))
    fig.patch.set_facecolor('#1a1a2e')
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.4)

    ax_policy    = fig.add_subplot(gs[0, 0])
    ax_value     = fig.add_subplot(gs[0, 1])
    ax_ownership = fig.add_subplot(gs[0, 2])
    ax_total     = fig.add_subplot(gs[0, 3])
    ax_gamelen   = fig.add_subplot(gs[1, 0])
    ax_time      = fig.add_subplot(gs[1, 1])
    ax_lr        = fig.add_subplot(gs[1, 2])
    ax_stats     = fig.add_subplot(gs[1, 3])

    axes = [ax_policy, ax_value, ax_ownership, ax_total, ax_gamelen, ax_time, ax_lr, ax_stats]
    for ax in axes:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='#aaaaaa', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444466')
        ax.xaxis.label.set_color('#aaaaaa')
        ax.yaxis.label.set_color('#aaaaaa')
        ax.title.set_color('#dddddd')

    def plot_line(ax, x, y, color, label, alpha_raw=0.25, lw=2):
        ax.plot(x, y, color=color, alpha=alpha_raw, linewidth=1)
        if len(y) >= 3:
            ax.plot(x, smooth(y), color=color, linewidth=lw, label=label)
        else:
            ax.lines[-1].set_alpha(1.0)
            ax.lines[-1].set_label(label)

    # ── Policy loss ────────────────────────────────────────────
    plot_line(ax_policy, iters, data['policy_loss'], COLORS['policy'], 'policy loss')
    ax_policy.set_title('Policy Loss (cross-entropy)', fontsize=10, fontweight='bold')
    ax_policy.set_xlabel('Iteration')
    ax_policy.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='#dddddd', framealpha=0.6)
    ax_policy.set_xlim(left=1)

    # ── Value loss ───────────────────────────────────────────────
    plot_line(ax_value, iters, data['value_loss'], COLORS['value'], 'value loss')
    ax_value.set_title('Value Loss (MSE)', fontsize=10, fontweight='bold')
    ax_value.set_xlabel('Iteration')
    ax_value.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='#dddddd', framealpha=0.6)
    ax_value.set_xlim(left=1)

    # ── Ownership loss ───────────────────────────────────────────
    if 'ownership_loss' in data and np.any(data['ownership_loss'] > 0):
        plot_line(ax_ownership, iters, data['ownership_loss'], COLORS['ownership'], 'ownership loss')
        ax_ownership.axhline(np.log(2), color='#666688', linestyle='--', linewidth=1, label='ln(2) random')
    ax_ownership.set_title('Ownership Loss (BCE)', fontsize=10, fontweight='bold')
    ax_ownership.set_xlabel('Iteration')
    ax_ownership.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='#dddddd', framealpha=0.6)
    ax_ownership.set_xlim(left=1)

    # ── Total loss ──────────────────────────────────────────────
    plot_line(ax_total, iters, data['total_loss'], COLORS['total'], 'total loss')
    ax_total.set_title('Total Loss', fontsize=10, fontweight='bold')
    ax_total.set_xlabel('Iteration')
    ax_total.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='#dddddd', framealpha=0.6)
    ax_total.set_xlim(left=1)

    # ── Average game length ─────────────────────────────────────────
    if 'avg_game_length' in data and np.any(data['avg_game_length'] > 0):
        game_lens = data['avg_game_length']
        plot_line(ax_gamelen, iters, game_lens, COLORS['game_len'], 'avg length')
        ax_gamelen.set_title('Avg Game Length', fontsize=10, fontweight='bold')
        ax_gamelen.set_xlabel('Iteration')
        ax_gamelen.set_ylabel('moves')
        ax_gamelen.set_xlim(left=1)
        ax_gamelen.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='#dddddd', framealpha=0.6)
    else:
        ax_gamelen.set_title('Avg Game Length', fontsize=10, fontweight='bold')
        ax_gamelen.text(0.5, 0.5, 'no data yet', transform=ax_gamelen.transAxes,
                        ha='center', va='center', color='#666688', fontsize=10)

    # ── Time per iteration ───────────────────────────────────────
    sp_min = data['time_selfplay'] / 60
    tr_min = data['time_training'] / 60
    ax_time.bar(iters, sp_min, color=COLORS['time_sp'], alpha=0.8, label='self-play', width=0.8)
    ax_time.bar(iters, tr_min, bottom=sp_min, color=COLORS['time_tr'], alpha=0.8, label='training', width=0.8)
    ax_time.set_title('Time per Iteration', fontsize=10, fontweight='bold')
    ax_time.set_xlabel('Iteration')
    ax_time.set_ylabel('minutes')
    ax_time.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='#dddddd', framealpha=0.6)
    ax_time.set_xlim(left=0.5)

    # ── Learning rate ─────────────────────────────────────────────
    if 'learning_rate' in data:
        plot_line(ax_lr, iters, data['learning_rate'], '#e74c3c', 'LR')
        ax_lr.set_title('Learning Rate', fontsize=10, fontweight='bold')
        ax_lr.set_xlabel('Iteration')
        ax_lr.ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))
        ax_lr.set_xlim(left=1)
    else:
        ax_lr.axis('off')

    # ── Stats summary ────────────────────────────────────────────
    ax_stats.axis('off')
    last = {k: v[-1] for k, v in data.items()}
    recent_policy = np.mean(data['policy_loss'][-5:]) if n >= 5 else data['policy_loss'][-1]
    recent_value  = np.mean(data['value_loss'][-5:])  if n >= 5 else data['value_loss'][-1]
    recent_own    = np.mean(data['ownership_loss'][-5:]) if ('ownership_loss' in data and n >= 5) else (data.get('ownership_loss', [0])[-1] if 'ownership_loss' in data else 0)
    total_games   = int(np.sum(data['games_played']))
    total_time_h  = np.sum(data['time_selfplay'] + data['time_training']) / 3600
    avg_gl = np.mean(data['avg_game_length'][-5:]) if ('avg_game_length' in data and n >= 5 and np.any(data['avg_game_length'] > 0)) else 0

    summary = [
        ('Iterations',        f"{int(last['iteration'])}"),
        ('Total games',       f"{total_games:,}"),
        ('Avg game length',   f"{avg_gl:.1f}" if avg_gl > 0 else 'n/a'),
        ('Total train time',  f"{total_time_h:.1f} h"),
        ('',                  ''),
        ('Policy loss (5-avg)', f"{recent_policy:.4f}"),
        ('Value loss (5-avg)',  f"{recent_value:.4f}"),
        ('Ownership loss',      f"{recent_own:.4f}"),
    ]

    y_pos = 0.95
    for label, value in summary:
        if not label:
            y_pos -= 0.06
            continue
        ax_stats.text(0.05, y_pos, label + ':', transform=ax_stats.transAxes,
                      color='#888899', fontsize=9, va='top')
        ax_stats.text(0.95, y_pos, value, transform=ax_stats.transAxes,
                      color='#eeeeff', fontsize=9, va='top', ha='right', fontweight='bold')
        y_pos -= 0.10

    fig.suptitle(f'9×9 Go Training Progress  —  iteration {int(last["iteration"])}',
                 color='#eeeeff', fontsize=13, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Plot Go training log')
    parser.add_argument('--log', default='logs/training_log.csv',
                        help='Path to CSV log file')
    parser.add_argument('--save', default=None,
                        help='Save plot to file instead of showing it')
    parser.add_argument('--watch', action='store_true',
                        help='Refresh plot every --interval seconds')
    parser.add_argument('--interval', type=int, default=60,
                        help='Refresh interval in seconds (with --watch)')
    args = parser.parse_args()

    if not Path(args.log).exists():
        print(f"Log file not found: {args.log}")
        sys.exit(1)

    if args.watch:
        matplotlib.use('Agg')  # non-interactive for watch mode
        save = args.save or 'logs/training_progress.png'
        print(f"Watch mode: refreshing every {args.interval}s → {save}")
        while True:
            data = load_log(args.log)
            if len(data) > 0 and len(data.get('iteration', [])) > 0:
                plot(data, save_path=save)
            else:
                print("  No data yet, waiting...")
            time.sleep(args.interval)
    else:
        data = load_log(args.log)
        if not data or len(data.get('iteration', [])) == 0:
            print("No data in log file yet.")
            sys.exit(0)
        plot(data, save_path=args.save)


if __name__ == '__main__':
    main()
