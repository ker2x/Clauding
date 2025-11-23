"""
Network Health Analysis using WeightWatcher

This script uses the WeightWatcher package to analyze the health and quality of trained
SAC networks using Random Matrix Theory (RMT) and other theoretical frameworks.

WeightWatcher provides insights into:
- Generalization capacity (alpha metrics)
- Over-parameterization (log spectral norm)
- Training quality (stable rank, condition number)
- Layer-wise health diagnostics
- Dead neuron detection:
  * Activation-based (default): Measures activation variance - works for any activation function
  * Weight-based: Measures weight norms - simpler but less accurate for Leaky ReLU

Usage:
    # Analyze a single checkpoint (activation-based by default)
    python analyze_network_health.py --checkpoint checkpoints_selection_parallel/best_model.pt

    # Use weight-based dead neuron detection (faster, less accurate)
    python analyze_network_health.py --checkpoint best_model.pt --dead-neuron-method weight

    # Adjust number of samples for activation-based analysis
    python analyze_network_health.py --checkpoint best_model.pt --dead-neuron-samples 5000

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

    def count_dead_neurons_activation_based(
        self,
        network: nn.Module,
        sample_states: torch.Tensor,
        sample_actions: torch.Tensor | None = None,
        variance_threshold: float = 1e-6
    ) -> dict[str, Any]:
        """
        Count dead neurons using activation-based analysis.

        Runs sample data through the network and measures activation variance.
        A neuron is "dead" if its activation variance across samples is below threshold.
        This works for any activation function (ReLU, Leaky ReLU, etc.).

        Args:
            network: PyTorch network to analyze
            sample_states: Tensor of sample inputs [batch_size, input_dim]
            sample_actions: Optional tensor of actions for critic networks [batch_size, action_dim]
            variance_threshold: Variance threshold below which neuron is dead

        Returns:
            Dictionary with dead neuron statistics
        """
        network.eval()
        activations = {}
        handles = []

        # Hook to capture activations from Linear layers
        def get_activation_hook(name):
            def hook(module, input, output):
                # Store activations for this layer
                activations[name] = output.detach()
            return hook

        # Register hooks on all Linear layers
        for name, module in network.named_modules():
            if isinstance(module, nn.Linear):
                handle = module.register_forward_hook(get_activation_hook(name))
                handles.append(handle)

        # Run forward pass with sample data
        with torch.no_grad():
            if sample_actions is not None:
                # Critic network: needs state and action
                _ = network(sample_states, sample_actions)
            else:
                # Actor network: needs only state
                _ = network(sample_states)

        # Remove hooks
        for handle in handles:
            handle.remove()

        # Analyze activations
        total_neurons = 0
        dead_neurons = 0
        layer_stats = []

        for name, acts in activations.items():
            # acts shape: [batch_size, num_neurons]
            num_neurons = acts.shape[1] if len(acts.shape) > 1 else 1

            if len(acts.shape) == 1:
                # Single neuron layer (like output layer)
                acts = acts.unsqueeze(1)

            # Compute variance across samples for each neuron
            neuron_variance = acts.var(dim=0)
            neuron_mean = acts.mean(dim=0)

            # Count dead neurons (low variance)
            layer_dead = (neuron_variance < variance_threshold).sum().item()
            layer_total = num_neurons

            total_neurons += layer_total
            dead_neurons += layer_dead

            layer_stats.append({
                'layer_name': name,
                'total_neurons': layer_total,
                'dead_neurons': layer_dead,
                'dead_ratio': layer_dead / layer_total if layer_total > 0 else 0.0,
                'min_variance': neuron_variance.min().item(),
                'max_variance': neuron_variance.max().item(),
                'mean_variance': neuron_variance.mean().item(),
                'min_activation': neuron_mean.min().item(),
                'max_activation': neuron_mean.max().item(),
                'mean_activation': neuron_mean.mean().item(),
            })

        dead_ratio = dead_neurons / total_neurons if total_neurons > 0 else 0.0

        return {
            'total_neurons': total_neurons,
            'dead_neurons': dead_neurons,
            'dead_ratio': dead_ratio,
            'layer_stats': layer_stats,
            'method': 'activation_based',
        }

    def count_dead_neurons_weight_based(self, network: nn.Module, threshold: float = 1e-6) -> dict[str, Any]:
        """
        Count dead neurons using weight-based analysis (fallback method).

        A neuron is considered "dead" if the L2 norm of its incoming weights
        is below the threshold, meaning it contributes negligibly to the output.

        Args:
            network: PyTorch network to analyze
            threshold: L2 norm threshold below which a neuron is considered dead

        Returns:
            Dictionary with dead neuron statistics
        """
        total_neurons = 0
        dead_neurons = 0
        layer_stats = []

        for name, module in network.named_modules():
            if isinstance(module, nn.Linear):
                # Weight shape: [out_features, in_features]
                # Each row represents the incoming weights to one output neuron
                weights = module.weight.data

                # Compute L2 norm of incoming weights for each neuron (row)
                neuron_norms = torch.norm(weights, p=2, dim=1)

                # Count dead neurons in this layer
                layer_dead = (neuron_norms < threshold).sum().item()
                layer_total = weights.shape[0]

                total_neurons += layer_total
                dead_neurons += layer_dead

                layer_stats.append({
                    'layer_name': name,
                    'total_neurons': layer_total,
                    'dead_neurons': layer_dead,
                    'dead_ratio': layer_dead / layer_total if layer_total > 0 else 0.0,
                    'min_norm': neuron_norms.min().item(),
                    'max_norm': neuron_norms.max().item(),
                    'mean_norm': neuron_norms.mean().item(),
                })

        dead_ratio = dead_neurons / total_neurons if total_neurons > 0 else 0.0

        return {
            'total_neurons': total_neurons,
            'dead_neurons': dead_neurons,
            'dead_ratio': dead_ratio,
            'layer_stats': layer_stats,
            'method': 'weight_based',
        }

    def generate_sample_states(self, state_dim: int, num_samples: int = 1000) -> torch.Tensor:
        """
        Generate random sample states for activation analysis.

        Args:
            state_dim: Dimension of state space
            num_samples: Number of samples to generate

        Returns:
            Tensor of random states
        """
        # Generate states from a reasonable distribution
        # Using standard normal as a proxy for normalized state features
        return torch.randn(num_samples, state_dim, device=self.device)

    def analyze_network(
        self,
        network: nn.Module,
        name: str,
        use_activation_based: bool = True,
        num_samples: int = 1000
    ) -> dict[str, Any]:
        """
        Analyze a single network using WeightWatcher.

        Args:
            network: PyTorch network to analyze
            name: Network name for reporting
            use_activation_based: Use activation-based dead neuron detection
            num_samples: Number of samples for activation analysis

        Returns:
            Dictionary with WeightWatcher metrics and dead neuron statistics
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

        # Count dead neurons
        if use_activation_based:
            state_dim = self.checkpoint.get('state_dim', 73)
            action_dim = self.checkpoint.get('action_dim', 2)
            sample_states = self.generate_sample_states(state_dim, num_samples)

            # Check if this is a critic network (needs actions)
            is_critic = 'critic' in name.lower()
            if is_critic:
                # Generate random actions for critics
                sample_actions = torch.randn(num_samples, action_dim, device=self.device)
                dead_neuron_stats = self.count_dead_neurons_activation_based(
                    network, sample_states, sample_actions
                )
            else:
                dead_neuron_stats = self.count_dead_neurons_activation_based(
                    network, sample_states
                )
            method_str = "activation-based"
        else:
            dead_neuron_stats = self.count_dead_neurons_weight_based(network)
            method_str = "weight-based"

        print(f"   ✓ Analyzed {len(details)} layers")
        print(f"   ✓ Dead neurons ({method_str}): {dead_neuron_stats['dead_neurons']}/{dead_neuron_stats['total_neurons']} "
              f"({dead_neuron_stats['dead_ratio']*100:.2f}%)")

        return {
            'details': details,
            'summary': summary,
            'dead_neurons': dead_neuron_stats,
            'name': name
        }

    def analyze_all_networks(
        self,
        network_names: list[str] | None = None,
        use_activation_based: bool = True,
        num_samples: int = 1000
    ) -> dict[str, dict]:
        """
        Analyze all networks in checkpoint.

        Args:
            network_names: Optional list of specific networks to analyze
            use_activation_based: Use activation-based dead neuron detection
            num_samples: Number of samples for activation analysis

        Returns:
            Dictionary mapping network names to analysis results
        """
        if network_names:
            networks_to_analyze = {k: v for k, v in self.networks.items() if k in network_names}
        else:
            networks_to_analyze = self.networks

        results = {}
        for name, network in networks_to_analyze.items():
            results[name] = self.analyze_network(
                network, name,
                use_activation_based=use_activation_based,
                num_samples=num_samples
            )

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

                # Interpret alpha (based on WeightWatcher theory: optimal range 2-6, best ~2)
                if alpha > 6.0:
                    health = "⚠️  CONCERNING (undertrained/random weights)"
                elif 5.0 < alpha <= 6.0:
                    health = "⚠️  BORDERLINE (needs more training)"
                elif 2.0 <= alpha <= 5.0:
                    health = "✅ GOOD (well-trained, should generalize)"
                    if alpha < 2.5:
                        health = "✅ EXCELLENT (near-optimal training)"
                else:
                    health = "❌ POOR (overfit - early stopping needed)"

                print(f"   Health: {health}")

            # Dead neurons
            if 'dead_neurons' in result:
                dead_stats = result['dead_neurons']
                dead_count = dead_stats['dead_neurons']
                total_count = dead_stats['total_neurons']
                dead_ratio = dead_stats['dead_ratio']
                method = dead_stats.get('method', 'unknown')

                print(f"\n💀 Dead Neurons ({method}):")
                print(f"   Count: {dead_count}/{total_count}")
                print(f"   Ratio: {dead_ratio*100:.2f}%")

                # Interpret dead neuron ratio
                if dead_ratio > 0.3:
                    status = "❌ CRITICAL (>30% dead - significant capacity waste)"
                elif dead_ratio > 0.15:
                    status = "⚠️  HIGH (>15% dead - consider pruning/smaller network)"
                elif dead_ratio > 0.05:
                    status = "⚠️  MODERATE (>5% dead - some waste)"
                else:
                    status = "✅ GOOD (<5% dead - healthy utilization)"

                print(f"   Status: {status}")

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
                    random_layers = (alphas > 6.0).sum()
                    overfit_layers = (alphas < 2.0).sum()
                    undertrained_layers = ((alphas > 5.0) & (alphas <= 6.0)).sum()

                    if random_layers > 0:
                        issues.append(f"   • {random_layers} layers with alpha > 6.0 (random/untrained)")
                    if undertrained_layers > 0:
                        issues.append(f"   • {undertrained_layers} layers with alpha ∈ (5, 6] (undertrained)")
                    if overfit_layers > 0:
                        issues.append(f"   • {overfit_layers} layers with alpha < 2.0 (overfit)")

                if 'log_spectral_norm' in available_cols:
                    high_norm_layers = (details['log_spectral_norm'] > 2.0).sum()
                    if high_norm_layers > 0:
                        issues.append(f"   • {high_norm_layers} layers with high spectral norm (may need regularization)")

                # Check for dead neurons
                if 'dead_neurons' in result:
                    dead_stats = result['dead_neurons']
                    dead_ratio = dead_stats['dead_ratio']
                    if dead_ratio > 0.15:
                        issues.append(f"   • {dead_ratio*100:.1f}% dead neurons (consider smaller network or pruning)")
                    elif dead_ratio > 0.05:
                        issues.append(f"   • {dead_ratio*100:.1f}% dead neurons (moderate waste)")

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

                # Write dead neuron statistics
                if 'dead_neurons' in result:
                    dead_stats = result['dead_neurons']
                    method = dead_stats.get('method', 'unknown')
                    f.write(f"\nDead Neuron Analysis ({method}):\n")
                    f.write(f"  Total neurons: {dead_stats['total_neurons']}\n")
                    f.write(f"  Dead neurons: {dead_stats['dead_neurons']}\n")
                    f.write(f"  Dead ratio: {dead_stats['dead_ratio']*100:.2f}%\n")
                    f.write("\n  Per-Layer Dead Neurons:\n")
                    for layer_stat in dead_stats['layer_stats']:
                        f.write(f"    {layer_stat['layer_name']}:\n")
                        f.write(f"      Total: {layer_stat['total_neurons']}, ")
                        f.write(f"Dead: {layer_stat['dead_neurons']} ({layer_stat['dead_ratio']*100:.1f}%)\n")

                        # Different stats based on method
                        if method == 'activation_based':
                            f.write(f"      Activation variance - min: {layer_stat['min_variance']:.6e}, ")
                            f.write(f"max: {layer_stat['max_variance']:.6e}, ")
                            f.write(f"mean: {layer_stat['mean_variance']:.6e}\n")
                            f.write(f"      Mean activation - min: {layer_stat['min_activation']:.6f}, ")
                            f.write(f"max: {layer_stat['max_activation']:.6f}, ")
                            f.write(f"mean: {layer_stat['mean_activation']:.6f}\n")
                        else:
                            f.write(f"      Weight norms - min: {layer_stat['min_norm']:.6f}, ")
                            f.write(f"max: {layer_stat['max_norm']:.3f}, ")
                            f.write(f"mean: {layer_stat['mean_norm']:.3f}\n")
                    f.write("\n")

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

                # Save dead neuron statistics
                if 'dead_neurons' in result:
                    import pandas as pd
                    dead_csv_path = os.path.join(output_dir, f'{name}_dead_neurons.csv')
                    dead_df = pd.DataFrame(result['dead_neurons']['layer_stats'])
                    dead_df.to_csv(dead_csv_path, index=False)
                    print(f"   ✓ Saved {name} dead neurons to {dead_csv_path}")

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
                    ax.axvline(2.0, color='darkgreen', linestyle='--', linewidth=2, label='Optimal (α≈2)')
                    ax.axvline(5.0, color='orange', linestyle='--', label='Upper good (α=5)')
                    ax.axvline(6.0, color='red', linestyle='--', label='Undertrained (α=6)')
                    ax.fill_between([2.0, 5.0], 0, ax.get_ylim()[1], alpha=0.1, color='green')
                    ax.set_xlabel('Alpha (Power Law Exponent)')
                    ax.set_ylabel('Layer Count')
                    ax.set_title('Alpha Distribution\n(2-6 is valid range, 2-5 preferred)')
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
                    ax.axhline(2.0, color='darkgreen', linestyle='--', linewidth=2, alpha=0.7, label='Optimal')
                    ax.axhline(5.0, color='orange', linestyle='--', alpha=0.5, label='Upper good')
                    ax.axhline(6.0, color='red', linestyle='--', alpha=0.5, label='Undertrained')
                    ax.fill_between(layer_ids, 2.0, 5.0, alpha=0.1, color='green')
                    ax.set_xlabel('Layer Index')
                    ax.set_ylabel('Alpha')
                    ax.set_title('Alpha by Layer\n(green zone = good, closer to 2 is better)')
                    ax.legend(fontsize=8)
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
                # Best is closest to 2.0 (optimal), within [2, 6] range
                best_network = min(alpha_means.items(), key=lambda x: abs(x[1] - 2.0) if 2.0 <= x[1] <= 6.0 else float('inf'))
                worst_network = max(alpha_means.items(), key=lambda x: abs(x[1] - 2.0))

                print(f"\n✅ Healthiest network: {best_network[0]} (alpha = {best_network[1]:.3f})")
                if abs(worst_network[1] - 2.0) > 3.0:  # Significantly far from optimal
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

                    if 2.0 <= mean_alpha <= 5.0:
                        health = "✅ GOOD"
                        if mean_alpha < 2.5:
                            health = "✅ EXCELLENT"
                    elif 5.0 < mean_alpha <= 6.0:
                        health = "⚠️  BORDERLINE (undertrained)"
                    elif mean_alpha > 6.0:
                        health = "❌ Random/Untrained"
                    else:
                        health = "⚠️  OVERFIT"

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
    parser.add_argument(
        '--dead-neuron-method',
        type=str,
        default='activation',
        choices=['activation', 'weight'],
        help='Method for dead neuron detection: activation-based (default, works for any activation) or weight-based (simpler, faster)'
    )
    parser.add_argument(
        '--dead-neuron-samples',
        type=int,
        default=1000,
        help='Number of samples for activation-based dead neuron detection (default: 1000)'
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

    # Determine dead neuron detection method
    use_activation_based = (args.dead_neuron_method == 'activation')

    analyzer.analyze_all_networks(
        network_names=args.networks,
        use_activation_based=use_activation_based,
        num_samples=args.dead_neuron_samples
    )
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
Alpha (Power Law Exponent) - Based on WeightWatcher Theory:
  • α ≈ 2.0  : OPTIMAL - best trained models (ideal target)
  • α ∈ [2, 5]: Well-trained, good generalization (RECOMMENDED)
  • α ∈ (5, 6]: Undertrained, needs more training
  • α > 6.0  : Severely undertrained/random weights
  • α < 2.0  : Overfit - early stopping recommended

  Rule: "Smaller is better" within the 2-6 range, with α≈2 being optimal.

Log Spectral Norm:
  • Lower values indicate better conditioning
  • Very high values may indicate instability or need for regularization

Stable Rank:
  • Measures effective dimensionality of weight matrices
  • Higher rank = more capacity being used
  • Very low rank may indicate redundancy

For more details, see:
  - WeightWatcher official docs: https://weightwatcher.ai
  - Martin & Mahoney (2019): "Traditional and Heavy-Tailed Self Regularization"
  - WeightWatcher GitHub: https://github.com/CalculatedContent/WeightWatcher
    """)


if __name__ == '__main__':
    main()
