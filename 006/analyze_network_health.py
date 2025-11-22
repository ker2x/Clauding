"""
Network Health Analysis using WeightWatcher

This script uses the WeightWatcher package to analyze the health and quality of trained
SAC networks using Random Matrix Theory (RMT) and other theoretical frameworks.

WeightWatcher provides insights into:
- Generalization capacity (alpha metrics)
- Over-parameterization (log spectral norm)
- Training quality (stable rank, condition number)
- Layer-wise health diagnostics

Usage:
    # Analyze a single checkpoint
    python analyze_network_health.py --checkpoint checkpoints_selection_parallel/best_model.pt

    # Compare multiple checkpoints
    python analyze_network_health.py --checkpoint gen_100.pt gen_200.pt gen_300.pt

    # Save detailed report
    python analyze_network_health.py --checkpoint best_model.pt --output-dir health_reports/

    # Analyze specific networks only
    python analyze_network_health.py --checkpoint best_model.pt --networks actor critic_1

References:
    - Martin & Mahoney, 2019: "Traditional and Heavy-Tailed Self Regularization in Neural Networks"
    - Martin & Mahoney, 2021: "Implicit Self-Regularization in Deep Neural Networks"
    - WeightWatcher: https://github.com/CalculatedContent/WeightWatcher
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

try:
    import weightwatcher as ww
except ImportError:
    print("❌ ERROR: weightwatcher package not installed")
    print("   Install with: pip install weightwatcher")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  WARNING: matplotlib not available, skipping visualizations")

from sac.agent import SACAgent
from sac.actor import VectorActor
from sac.critic import VectorCritic


class NetworkHealthAnalyzer:
    """Analyzes neural network health using WeightWatcher."""

    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        """
        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load networks on
        """
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.checkpoint = None
        self.networks = {}
        self.ww_results = {}

    def load_checkpoint(self) -> dict[str, Any]:
        """Load checkpoint and extract network state dicts."""
        print(f"\n📦 Loading checkpoint: {self.checkpoint_path}")

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Extract dimensions
        state_dim = self.checkpoint.get('state_dim', 73)
        action_dim = self.checkpoint.get('action_dim', 2)

        print(f"   State dim: {state_dim}")
        print(f"   Action dim: {action_dim}")

        # Create networks and load state dicts
        self.networks = {
            'actor': VectorActor(state_dim, action_dim).to(self.device),
            'critic_1': VectorCritic(state_dim, action_dim).to(self.device),
            'critic_2': VectorCritic(state_dim, action_dim).to(self.device),
            'critic_target_1': VectorCritic(state_dim, action_dim).to(self.device),
            'critic_target_2': VectorCritic(state_dim, action_dim).to(self.device),
        }

        # Load weights
        for name, network in self.networks.items():
            if name in self.checkpoint:
                network.load_state_dict(self.checkpoint[name])
                network.eval()
                print(f"   ✓ Loaded {name}")

        return self.checkpoint

    def analyze_network(self, network: nn.Module, name: str) -> dict[str, Any]:
        """
        Analyze a single network using WeightWatcher.

        Args:
            network: PyTorch network to analyze
            name: Network name for reporting

        Returns:
            Dictionary with WeightWatcher metrics
        """
        print(f"\n🔬 Analyzing {name}...")

        # Create WeightWatcher analyzer
        watcher = ww.WeightWatcher(model=network)

        # Run analysis with various metrics
        details = watcher.analyze(
            randomize=False,  # Don't randomize weights
            plot=False,  # Don't auto-plot
        )

        # Extract summary statistics
        summary = watcher.get_summary()

        print(f"   ✓ Analyzed {len(details)} layers")

        return {
            'details': details,
            'summary': summary,
            'name': name
        }

    def analyze_all_networks(self, network_names: list[str] | None = None) -> dict[str, dict]:
        """
        Analyze all networks in checkpoint.

        Args:
            network_names: Optional list of specific networks to analyze

        Returns:
            Dictionary mapping network names to analysis results
        """
        if network_names:
            networks_to_analyze = {k: v for k, v in self.networks.items() if k in network_names}
        else:
            networks_to_analyze = self.networks

        results = {}
        for name, network in networks_to_analyze.items():
            results[name] = self.analyze_network(network, name)

        self.ww_results = results
        return results

    def print_summary(self) -> None:
        """Print summary of network health metrics."""
        print("\n" + "="*80)
        print("NETWORK HEALTH SUMMARY")
        print("="*80)

        for name, result in self.ww_results.items():
            summary = result['summary']
            details = result['details']

            print(f"\n{'━'*80}")
            print(f"Network: {name.upper()}")
            print(f"{'━'*80}")

            # Overall metrics
            if 'alpha' in summary:
                alpha = summary['alpha']
                print(f"\n📊 Overall Metrics:")
                print(f"   Alpha (generalization): {alpha:.4f}")

                # Interpret alpha
                if alpha < 2.0:
                    health = "⚠️  CONCERNING (very undertrained or random)"
                elif 2.0 <= alpha < 3.0:
                    health = "✅ GOOD (well-trained, should generalize)"
                elif 3.0 <= alpha < 4.0:
                    health = "⚠️  BORDERLINE (may be overfit)"
                else:
                    health = "❌ POOR (likely overfit)"

                print(f"   Health: {health}")

            # Layer-wise statistics
            print(f"\n📈 Layer Statistics:")

            # Get key metrics from details dataframe
            if not details.empty:
                # Check which columns are available
                available_cols = details.columns.tolist()

                if 'alpha' in available_cols:
                    alphas = details['alpha'].dropna()
                    if len(alphas) > 0:
                        print(f"   Alpha range: [{alphas.min():.3f}, {alphas.max():.3f}]")
                        print(f"   Alpha mean: {alphas.mean():.3f} ± {alphas.std():.3f}")

                if 'log_spectral_norm' in available_cols:
                    log_norms = details['log_spectral_norm'].dropna()
                    if len(log_norms) > 0:
                        print(f"   Log spectral norm mean: {log_norms.mean():.3f}")

                if 'stable_rank' in available_cols:
                    ranks = details['stable_rank'].dropna()
                    if len(ranks) > 0:
                        print(f"   Stable rank mean: {ranks.mean():.3f}")

                # Layer count
                print(f"   Total layers analyzed: {len(details)}")

                # Potential issues
                print(f"\n⚠️  Potential Issues:")
                issues = []

                if 'alpha' in available_cols:
                    alphas = details['alpha'].dropna()
                    overfit_layers = (alphas > 4.0).sum()
                    underfit_layers = (alphas < 2.0).sum()

                    if overfit_layers > 0:
                        issues.append(f"   • {overfit_layers} layers with alpha > 4.0 (possible overfit)")
                    if underfit_layers > 0:
                        issues.append(f"   • {underfit_layers} layers with alpha < 2.0 (undertrained)")

                if 'log_spectral_norm' in available_cols:
                    high_norm_layers = (details['log_spectral_norm'] > 2.0).sum()
                    if high_norm_layers > 0:
                        issues.append(f"   • {high_norm_layers} layers with high spectral norm (may need regularization)")

                if issues:
                    for issue in issues:
                        print(issue)
                else:
                    print("   ✓ No obvious issues detected")

    def generate_report(self, output_dir: str | None = None) -> None:
        """
        Generate detailed HTML/text report.

        Args:
            output_dir: Directory to save report files
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            report_path = os.path.join(output_dir, 'network_health_report.txt')
        else:
            report_path = 'network_health_report.txt'

        print(f"\n💾 Saving detailed report to: {report_path}")

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("NETWORK HEALTH ANALYSIS REPORT\n")
            f.write(f"Checkpoint: {self.checkpoint_path}\n")
            f.write("="*80 + "\n\n")

            for name, result in self.ww_results.items():
                details = result['details']
                summary = result['summary']

                f.write(f"\n{'━'*80}\n")
                f.write(f"Network: {name.upper()}\n")
                f.write(f"{'━'*80}\n\n")

                # Write summary
                f.write("Summary Metrics:\n")
                for key, value in summary.items():
                    f.write(f"  {key}: {value}\n")

                # Write layer details
                f.write("\nLayer-by-Layer Analysis:\n")
                f.write(details.to_string())
                f.write("\n\n")

        print(f"   ✓ Report saved")

        # Save CSV for each network
        if output_dir:
            for name, result in self.ww_results.items():
                csv_path = os.path.join(output_dir, f'{name}_details.csv')
                result['details'].to_csv(csv_path, index=False)
                print(f"   ✓ Saved {name} details to {csv_path}")

    def plot_metrics(self, output_dir: str | None = None) -> None:
        """
        Generate visualization plots of network health metrics.

        Args:
            output_dir: Directory to save plots
        """
        if not HAS_MATPLOTLIB:
            print("\n⚠️  Skipping plots (matplotlib not available)")
            return

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = '.'

        print(f"\n📊 Generating visualizations...")

        for name, result in self.ww_results.items():
            details = result['details']

            if details.empty:
                continue

            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'Network Health Metrics: {name.upper()}', fontsize=16, fontweight='bold')

            # Plot 1: Alpha distribution
            if 'alpha' in details.columns:
                ax = axes[0, 0]
                alphas = details['alpha'].dropna()
                if len(alphas) > 0:
                    ax.hist(alphas, bins=20, edgecolor='black', alpha=0.7)
                    ax.axvline(2.0, color='green', linestyle='--', label='Good (α=2)')
                    ax.axvline(4.0, color='red', linestyle='--', label='Overfit (α=4)')
                    ax.set_xlabel('Alpha (Power Law Exponent)')
                    ax.set_ylabel('Layer Count')
                    ax.set_title('Alpha Distribution\n(2-4 is healthy range)')
                    ax.legend()
                    ax.grid(True, alpha=0.3)

            # Plot 2: Log Spectral Norm
            if 'log_spectral_norm' in details.columns:
                ax = axes[0, 1]
                log_norms = details['log_spectral_norm'].dropna()
                if len(log_norms) > 0:
                    layer_ids = range(len(log_norms))
                    ax.plot(layer_ids, log_norms, marker='o', linestyle='-', linewidth=2)
                    ax.set_xlabel('Layer Index')
                    ax.set_ylabel('Log Spectral Norm')
                    ax.set_title('Spectral Norm by Layer\n(lower is better)')
                    ax.grid(True, alpha=0.3)

            # Plot 3: Stable Rank
            if 'stable_rank' in details.columns:
                ax = axes[1, 0]
                ranks = details['stable_rank'].dropna()
                if len(ranks) > 0:
                    layer_ids = range(len(ranks))
                    ax.plot(layer_ids, ranks, marker='s', linestyle='-', linewidth=2, color='purple')
                    ax.set_xlabel('Layer Index')
                    ax.set_ylabel('Stable Rank')
                    ax.set_title('Stable Rank by Layer\n(effective dimensionality)')
                    ax.grid(True, alpha=0.3)

            # Plot 4: Alpha by layer (if available)
            if 'alpha' in details.columns:
                ax = axes[1, 1]
                alphas = details['alpha'].dropna()
                if len(alphas) > 0:
                    layer_ids = range(len(alphas))
                    ax.plot(layer_ids, alphas, marker='^', linestyle='-', linewidth=2, color='darkgreen')
                    ax.axhline(2.0, color='green', linestyle='--', alpha=0.5)
                    ax.axhline(4.0, color='red', linestyle='--', alpha=0.5)
                    ax.fill_between(layer_ids, 2.0, 4.0, alpha=0.1, color='green')
                    ax.set_xlabel('Layer Index')
                    ax.set_ylabel('Alpha')
                    ax.set_title('Alpha by Layer\n(green zone = healthy)')
                    ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save plot
            plot_path = os.path.join(output_dir, f'{name}_health_metrics.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"   ✓ Saved {name} plots to {plot_path}")

    def compare_networks(self) -> None:
        """Compare health metrics across different networks."""
        print("\n" + "="*80)
        print("NETWORK COMPARISON")
        print("="*80)

        metrics = {}
        for name, result in self.ww_results.items():
            summary = result['summary']
            details = result['details']

            metrics[name] = {}
            if 'alpha' in summary:
                metrics[name]['alpha_overall'] = summary['alpha']

            if 'alpha' in details.columns:
                alphas = details['alpha'].dropna()
                if len(alphas) > 0:
                    metrics[name]['alpha_mean'] = alphas.mean()
                    metrics[name]['alpha_std'] = alphas.std()
                    metrics[name]['alpha_min'] = alphas.min()
                    metrics[name]['alpha_max'] = alphas.max()

        # Print comparison table
        print("\n📊 Alpha Metrics Comparison:")
        print(f"{'Network':<20} {'Mean Alpha':<15} {'Std':<10} {'Range':<20}")
        print("-"*80)

        for name, m in metrics.items():
            if 'alpha_mean' in m:
                range_str = f"[{m['alpha_min']:.2f}, {m['alpha_max']:.2f}]"
                print(f"{name:<20} {m['alpha_mean']:>8.3f}        {m['alpha_std']:>8.3f}  {range_str:<20}")

        # Identify best/worst networks
        if metrics:
            alpha_means = {k: v.get('alpha_mean', 0) for k, v in metrics.items() if 'alpha_mean' in v}
            if alpha_means:
                best_network = min(alpha_means.items(), key=lambda x: abs(x[1] - 3.0))  # Closest to 3.0
                worst_network = max(alpha_means.items(), key=lambda x: abs(x[1] - 3.0) if x[1] > 4.0 else 0)

                print(f"\n✅ Healthiest network: {best_network[0]} (alpha = {best_network[1]:.3f})")
                if worst_network[1] > 4.0:
                    print(f"⚠️  Most concerning: {worst_network[0]} (alpha = {worst_network[1]:.3f})")


def compare_checkpoints(checkpoint_paths: list[str], output_dir: str | None = None) -> None:
    """
    Compare health metrics across multiple checkpoints.

    Args:
        checkpoint_paths: List of checkpoint paths
        output_dir: Directory to save comparison report
    """
    print("\n" + "="*80)
    print("MULTI-CHECKPOINT COMPARISON")
    print("="*80)

    all_results = {}

    for ckpt_path in checkpoint_paths:
        analyzer = NetworkHealthAnalyzer(ckpt_path)
        analyzer.load_checkpoint()
        analyzer.analyze_all_networks(['actor', 'critic_1'])  # Analyze main networks
        all_results[ckpt_path] = analyzer.ww_results

    # Compare actor networks across checkpoints
    print("\n📊 Actor Alpha Comparison:")
    print(f"{'Checkpoint':<40} {'Mean Alpha':<15} {'Health':<30}")
    print("-"*80)

    for ckpt_path, results in all_results.items():
        if 'actor' in results:
            details = results['actor']['details']
            if 'alpha' in details.columns:
                alphas = details['alpha'].dropna()
                if len(alphas) > 0:
                    mean_alpha = alphas.mean()

                    if 2.0 <= mean_alpha < 3.0:
                        health = "✅ GOOD"
                    elif 3.0 <= mean_alpha < 4.0:
                        health = "⚠️  BORDERLINE"
                    elif mean_alpha >= 4.0:
                        health = "❌ Mostly Random"
                    else:
                        health = "⚠️  UNDERTRAINED"

                    ckpt_name = Path(ckpt_path).name
                    print(f"{ckpt_name:<40} {mean_alpha:>8.3f}        {health:<30}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze neural network health using WeightWatcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single checkpoint
  python analyze_network_health.py --checkpoint checkpoints_selection_parallel/best_model.pt

  # Compare multiple checkpoints
  python analyze_network_health.py --checkpoint gen_100.pt gen_200.pt gen_300.pt

  # Save detailed reports and plots
  python analyze_network_health.py --checkpoint best_model.pt --output-dir health_reports/

  # Analyze specific networks
  python analyze_network_health.py --checkpoint best_model.pt --networks actor critic_1
        """
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        nargs='+',
        required=True,
        help='Path(s) to checkpoint file(s)'
    )
    parser.add_argument(
        '--networks',
        type=str,
        nargs='+',
        choices=['actor', 'critic_1', 'critic_2', 'critic_target_1', 'critic_target_2'],
        help='Specific networks to analyze (default: all)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Directory to save reports and plots'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'mps'],
        help='Device to load networks on (default: cpu)'
    )

    args = parser.parse_args()

    # Multiple checkpoint comparison
    if len(args.checkpoint) > 1:
        compare_checkpoints(args.checkpoint, args.output_dir)
        return

    # Single checkpoint analysis
    checkpoint_path = args.checkpoint[0]

    analyzer = NetworkHealthAnalyzer(checkpoint_path, device=args.device)
    analyzer.load_checkpoint()
    analyzer.analyze_all_networks(network_names=args.networks)
    analyzer.print_summary()
    analyzer.compare_networks()

    if args.output_dir:
        analyzer.generate_report(args.output_dir)
        analyzer.plot_metrics(args.output_dir)

    print("\n✅ Analysis complete!")
    print("\n" + "="*80)
    print("INTERPRETATION GUIDE")
    print("="*80)
    print("""
Alpha (Power Law Exponent):
  • α > 6.0  : random weights
  • α ∈ [4, 6): possibly underfit
  • α ∈ [2, 4): Well-trained, good generalization (IDEAL)
  • α < 2.0  : Likely overfit, poor generalization

Log Spectral Norm:
  • Lower values indicate better conditioning
  • Very high values may indicate instability or need for regularization

Stable Rank:
  • Measures effective dimensionality of weight matrices
  • Higher rank = more capacity being used
  • Very low rank may indicate redundancy

For more details, see:
  - Martin & Mahoney (2019): "Traditional and Heavy-Tailed Self Regularization"
  - WeightWatcher documentation: https://github.com/CalculatedContent/WeightWatcher
    """)


if __name__ == '__main__':
    main()
