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
- Input feature relevance:
  * Analyzes which input dimensions are actually used by the network
  * Identifies potentially useless observation features
  * Helps optimize observation space design
- Layer-by-layer information flow:
  * Tracks how information propagates through each layer
  * Identifies bottlenecks and underutilized layers
  * Shows input/output usage ratios for every layer
  * Helps optimize network architecture

Usage:
    # Analyze a single checkpoint (activation-based by default)
    python analyze_network_health.py --checkpoint checkpoints_selection_parallel/best_model.pt

    # Use weight-based dead neuron detection (faster, less accurate)
    python analyze_network_health.py --checkpoint best_model.pt --dead-neuron-method weight

    # Adjust number of samples for activation-based analysis
    python analyze_network_health.py --checkpoint best_model.pt --dead-neuron-samples 5000

    # Compare multiple checkpoints
    python analyze_network_health.py --checkpoint gen_100.pt gen_200.pt gen_300.pt

    # Save detailed report (plots saved to health_reports/plots/)
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

    def analyze_input_relevance(
        self,
        network: nn.Module,
        network_name: str,
        state_dim: int,
        action_dim: int = 0
    ) -> dict[str, Any]:
        """
        Analyzes which input dimensions the network actually uses by examining
        first layer weight magnitudes.

        Args:
            network: PyTorch network to analyze
            network_name: Name of network (for identifying critics vs actors)
            state_dim: Dimension of state space
            action_dim: Dimension of action space (for critics)

        Returns:
            Dictionary with input relevance statistics
        """
        # Get first layer weights
        first_layer = None
        for module in network.modules():
            if isinstance(module, nn.Linear):
                first_layer = module
                break

        if first_layer is None:
            return {
                'error': 'No linear layers found',
                'useless_count': 0,
                'total_inputs': 0,
            }

        # Get weights: [out_features, in_features]
        w = first_layer.weight.data.abs()

        # Sum across output neurons to get relevance per input dimension
        relevance = w.sum(dim=0).cpu().numpy()

        # Normalize to 0-1
        if relevance.max() > 0:
            relevance_normalized = relevance / relevance.max()
        else:
            relevance_normalized = relevance

        # For critics, separate state and action relevance
        is_critic = 'critic' in network_name.lower()
        if is_critic and action_dim > 0:
            state_relevance = relevance_normalized[:state_dim]
            action_relevance = relevance_normalized[state_dim:state_dim + action_dim]
        else:
            state_relevance = relevance_normalized
            action_relevance = None

        # Identify useless inputs (< 5% of max)
        threshold = 0.05
        useless_state_indices = [i for i, r in enumerate(state_relevance) if r < threshold]

        if action_relevance is not None:
            useless_action_indices = [i for i, r in enumerate(action_relevance) if r < threshold]
        else:
            useless_action_indices = []

        return {
            'state_relevance': state_relevance,
            'action_relevance': action_relevance,
            'relevance_raw': relevance,
            'relevance_normalized': relevance_normalized,
            'useless_state_indices': useless_state_indices,
            'useless_action_indices': useless_action_indices,
            'useless_state_count': len(useless_state_indices),
            'useless_action_count': len(useless_action_indices),
            'total_state_inputs': len(state_relevance),
            'total_action_inputs': len(action_relevance) if action_relevance is not None else 0,
            'threshold': threshold,
        }

    def analyze_layer_by_layer_relevance(self, network: nn.Module, network_name: str) -> dict[str, Any]:
        """
        Analyzes relevance for every layer in the network, showing how information
        flows through the entire architecture.

        For each layer:
        - Input relevance: which inputs from previous layer matter most
        - Output strength: which output neurons have strong connections
        - Bottlenecks: layers where information is heavily filtered

        Args:
            network: PyTorch network to analyze
            network_name: Name of network for reporting

        Returns:
            Dictionary with per-layer relevance statistics
        """
        layer_stats = []
        threshold = 0.05

        # Collect all linear layers with names
        linear_layers = []
        for name, module in network.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers.append((name, module))

        if len(linear_layers) == 0:
            return {'error': 'No linear layers found'}

        for layer_idx, (layer_name, layer) in enumerate(linear_layers):
            # Get weights: [out_features, in_features]
            w = layer.weight.data.abs()

            # Input relevance: sum across output neurons for each input
            input_relevance = w.sum(dim=0).cpu().numpy()
            input_relevance_normalized = input_relevance / input_relevance.max() if input_relevance.max() > 0 else input_relevance

            # Output strength: L2 norm of incoming weights for each output neuron
            output_strength = torch.norm(w, p=2, dim=1).cpu().numpy()
            output_strength_normalized = output_strength / output_strength.max() if output_strength.max() > 0 else output_strength

            # Identify weak inputs and outputs
            weak_inputs = (input_relevance_normalized < threshold).sum()
            weak_outputs = (output_strength_normalized < threshold).sum()

            # Information flow metrics
            input_usage_ratio = 1.0 - (weak_inputs / len(input_relevance_normalized))
            output_usage_ratio = 1.0 - (weak_outputs / len(output_strength_normalized))

            layer_stats.append({
                'layer_name': layer_name,
                'layer_index': layer_idx,
                'input_dim': w.shape[1],
                'output_dim': w.shape[0],
                'input_relevance': input_relevance_normalized,
                'output_strength': output_strength_normalized,
                'weak_inputs': int(weak_inputs),
                'weak_outputs': int(weak_outputs),
                'input_usage_ratio': input_usage_ratio,
                'output_usage_ratio': output_usage_ratio,
                'avg_input_relevance': float(input_relevance_normalized.mean()),
                'avg_output_strength': float(output_strength_normalized.mean()),
                'min_input_relevance': float(input_relevance_normalized.min()),
                'min_output_strength': float(output_strength_normalized.min()),
                'max_input_relevance': float(input_relevance_normalized.max()),
                'max_output_strength': float(output_strength_normalized.max()),
            })

        return {
            'layer_stats': layer_stats,
            'num_layers': len(layer_stats),
            'threshold': threshold,
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

        # Analyze input relevance
        state_dim = self.checkpoint.get('state_dim', 73)
        action_dim = self.checkpoint.get('action_dim', 2)
        is_critic = 'critic' in name.lower()

        input_relevance = self.analyze_input_relevance(
            network,
            name,
            state_dim,
            action_dim if is_critic else 0
        )

        if 'error' not in input_relevance:
            print(f"   ✓ Input relevance: {input_relevance['useless_state_count']}/{input_relevance['total_state_inputs']} "
                  f"state inputs likely unused (<{input_relevance['threshold']*100:.0f}% relevance)")
            if is_critic and input_relevance['total_action_inputs'] > 0:
                print(f"   ✓ Action relevance: {input_relevance['useless_action_count']}/{input_relevance['total_action_inputs']} "
                      f"action inputs likely unused")

        # Analyze layer-by-layer relevance
        layer_relevance = self.analyze_layer_by_layer_relevance(network, name)
        if 'error' not in layer_relevance:
            print(f"   ✓ Layer-by-layer analysis: {layer_relevance['num_layers']} layers analyzed")

        return {
            'details': details,
            'summary': summary,
            'dead_neurons': dead_neuron_stats,
            'input_relevance': input_relevance,
            'layer_relevance': layer_relevance,
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

            # Input relevance statistics
            if 'input_relevance' in result and 'error' not in result['input_relevance']:
                rel_stats = result['input_relevance']
                print(f"\n🎯 Input Relevance:")
                print(f"   Total state inputs: {rel_stats['total_state_inputs']}")
                print(f"   Likely unused state inputs: {rel_stats['useless_state_count']} "
                      f"({rel_stats['useless_state_count']/rel_stats['total_state_inputs']*100:.1f}%)")

                # Interpret state input usage
                usage_ratio = 1.0 - (rel_stats['useless_state_count'] / rel_stats['total_state_inputs'])
                if usage_ratio < 0.5:
                    status = "❌ POOR (<50% inputs used - consider reducing observation space)"
                elif usage_ratio < 0.7:
                    status = "⚠️  MODERATE (50-70% inputs used - some waste)"
                elif usage_ratio < 0.85:
                    status = "✅ GOOD (70-85% inputs used)"
                else:
                    status = "✅ EXCELLENT (>85% inputs used - efficient)"

                print(f"   Status: {status}")

                # For critics, also show action relevance
                if rel_stats['total_action_inputs'] > 0:
                    print(f"\n   Total action inputs: {rel_stats['total_action_inputs']}")
                    print(f"   Likely unused action inputs: {rel_stats['useless_action_count']} "
                          f"({rel_stats['useless_action_count']/rel_stats['total_action_inputs']*100:.1f}%)")

                # Show top unused input indices
                if len(rel_stats['useless_state_indices']) > 0:
                    top_unused = rel_stats['useless_state_indices'][:10]  # Show up to 10
                    print(f"   Unused state indices (showing up to 10): {top_unused}")

            # Layer-by-layer relevance statistics
            if 'layer_relevance' in result and 'error' not in result['layer_relevance']:
                layer_rel = result['layer_relevance']
                print(f"\n🔗 Layer-by-Layer Information Flow:")
                print(f"   Total layers: {layer_rel['num_layers']}")

                # Summary stats
                all_stats = layer_rel['layer_stats']
                avg_input_usage = np.mean([s['input_usage_ratio'] for s in all_stats])
                avg_output_usage = np.mean([s['output_usage_ratio'] for s in all_stats])

                print(f"   Average input usage: {avg_input_usage*100:.1f}%")
                print(f"   Average output usage: {avg_output_usage*100:.1f}%")

                # Find bottlenecks (layers with low usage)
                bottlenecks = [s for s in all_stats if s['input_usage_ratio'] < 0.7 or s['output_usage_ratio'] < 0.7]
                if bottlenecks:
                    print(f"\n   ⚠️  Potential bottlenecks ({len(bottlenecks)} layers with <70% usage):")
                    for b in bottlenecks[:3]:  # Show top 3
                        print(f"      • {b['layer_name']}: input {b['input_usage_ratio']*100:.1f}%, output {b['output_usage_ratio']*100:.1f}%")

                # Per-layer summary (compact)
                print(f"\n   Per-layer summary:")
                print(f"   {'Layer':<20} {'Shape':<15} {'Input Use':<12} {'Output Use':<12}")
                print(f"   {'-'*59}")
                for s in all_stats:
                    shape_str = f"{s['input_dim']}→{s['output_dim']}"
                    print(f"   {s['layer_name']:<20} {shape_str:<15} {s['input_usage_ratio']*100:>6.1f}%      {s['output_usage_ratio']*100:>6.1f}%")

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

                # Check for unused inputs
                if 'input_relevance' in result and 'error' not in result['input_relevance']:
                    rel_stats = result['input_relevance']
                    usage_ratio = 1.0 - (rel_stats['useless_state_count'] / rel_stats['total_state_inputs'])
                    if usage_ratio < 0.5:
                        issues.append(f"   • {rel_stats['useless_state_count']}/{rel_stats['total_state_inputs']} "
                                     f"state inputs unused (<50% used - consider reducing observation space)")
                    elif usage_ratio < 0.7:
                        issues.append(f"   • {rel_stats['useless_state_count']}/{rel_stats['total_state_inputs']} "
                                     f"state inputs unused (50-70% used)")

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

                # Write input relevance statistics
                if 'input_relevance' in result and 'error' not in result['input_relevance']:
                    rel_stats = result['input_relevance']
                    f.write(f"Input Relevance Analysis:\n")
                    f.write(f"  Total state inputs: {rel_stats['total_state_inputs']}\n")
                    f.write(f"  Likely unused state inputs: {rel_stats['useless_state_count']} "
                           f"({rel_stats['useless_state_count']/rel_stats['total_state_inputs']*100:.1f}%)\n")
                    f.write(f"  Threshold: {rel_stats['threshold']*100:.0f}% of max relevance\n")

                    if rel_stats['total_action_inputs'] > 0:
                        f.write(f"  Total action inputs: {rel_stats['total_action_inputs']}\n")
                        f.write(f"  Likely unused action inputs: {rel_stats['useless_action_count']} "
                               f"({rel_stats['useless_action_count']/rel_stats['total_action_inputs']*100:.1f}%)\n")

                    # List unused indices
                    if len(rel_stats['useless_state_indices']) > 0:
                        f.write(f"\n  Unused state input indices:\n")
                        f.write(f"    {rel_stats['useless_state_indices']}\n")

                    if len(rel_stats['useless_action_indices']) > 0:
                        f.write(f"\n  Unused action input indices:\n")
                        f.write(f"    {rel_stats['useless_action_indices']}\n")

                    # Write top and bottom relevance scores
                    state_rel = rel_stats['state_relevance']
                    sorted_indices = np.argsort(state_rel)
                    f.write(f"\n  Top 10 most relevant state inputs:\n")
                    for i in range(min(10, len(state_rel))):
                        idx = sorted_indices[-(i+1)]
                        f.write(f"    Index {idx}: {state_rel[idx]:.3f}\n")

                    f.write(f"\n  Top 10 least relevant state inputs:\n")
                    for i in range(min(10, len(state_rel))):
                        idx = sorted_indices[i]
                        f.write(f"    Index {idx}: {state_rel[idx]:.3f}\n")
                    f.write("\n")

                # Write layer-by-layer relevance
                if 'layer_relevance' in result and 'error' not in result['layer_relevance']:
                    layer_rel = result['layer_relevance']
                    f.write(f"Layer-by-Layer Information Flow:\n")
                    f.write(f"  Number of layers: {layer_rel['num_layers']}\n")
                    f.write(f"  Threshold: {layer_rel['threshold']*100:.0f}% of max\n\n")

                    all_stats = layer_rel['layer_stats']
                    avg_input_usage = np.mean([s['input_usage_ratio'] for s in all_stats])
                    avg_output_usage = np.mean([s['output_usage_ratio'] for s in all_stats])

                    f.write(f"  Average input usage: {avg_input_usage*100:.1f}%\n")
                    f.write(f"  Average output usage: {avg_output_usage*100:.1f}%\n\n")

                    f.write(f"  Per-layer details:\n")
                    f.write(f"  {'Layer':<20} {'Shape':<15} {'InUse':<8} {'OutUse':<8} {'AvgInRel':<10} {'AvgOutStr':<10}\n")
                    f.write(f"  {'-'*80}\n")
                    for s in all_stats:
                        shape_str = f"{s['input_dim']}→{s['output_dim']}"
                        f.write(f"  {s['layer_name']:<20} {shape_str:<15} "
                               f"{s['input_usage_ratio']*100:>6.1f}% "
                               f"{s['output_usage_ratio']*100:>6.1f}% "
                               f"{s['avg_input_relevance']:>8.3f}  "
                               f"{s['avg_output_strength']:>8.3f}\n")

                    # Bottlenecks
                    bottlenecks = [s for s in all_stats if s['input_usage_ratio'] < 0.7 or s['output_usage_ratio'] < 0.7]
                    if bottlenecks:
                        f.write(f"\n  Potential bottlenecks ({len(bottlenecks)} layers with <70% usage):\n")
                        for b in bottlenecks:
                            f.write(f"    • {b['layer_name']}: input {b['input_usage_ratio']*100:.1f}%, output {b['output_usage_ratio']*100:.1f}%\n")
                    f.write("\n")

                # Write layer details
                f.write("\nWeightWatcher Layer-by-Layer Analysis:\n")
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

                # Save input relevance statistics
                if 'input_relevance' in result and 'error' not in result['input_relevance']:
                    import pandas as pd
                    rel_stats = result['input_relevance']

                    # Create DataFrame with input relevance
                    rel_data = {
                        'input_index': list(range(len(rel_stats['state_relevance']))),
                        'relevance': rel_stats['state_relevance'],
                        'is_unused': [i in rel_stats['useless_state_indices'] for i in range(len(rel_stats['state_relevance']))],
                    }
                    rel_df = pd.DataFrame(rel_data)

                    rel_csv_path = os.path.join(output_dir, f'{name}_input_relevance.csv')
                    rel_df.to_csv(rel_csv_path, index=False)
                    print(f"   ✓ Saved {name} input relevance to {rel_csv_path}")

                # Save layer-by-layer relevance statistics
                if 'layer_relevance' in result and 'error' not in result['layer_relevance']:
                    import pandas as pd
                    layer_rel = result['layer_relevance']

                    # Create summary DataFrame (without the arrays)
                    layer_summary = []
                    for s in layer_rel['layer_stats']:
                        layer_summary.append({
                            'layer_name': s['layer_name'],
                            'layer_index': s['layer_index'],
                            'input_dim': s['input_dim'],
                            'output_dim': s['output_dim'],
                            'weak_inputs': s['weak_inputs'],
                            'weak_outputs': s['weak_outputs'],
                            'input_usage_ratio': s['input_usage_ratio'],
                            'output_usage_ratio': s['output_usage_ratio'],
                            'avg_input_relevance': s['avg_input_relevance'],
                            'avg_output_strength': s['avg_output_strength'],
                            'min_input_relevance': s['min_input_relevance'],
                            'min_output_strength': s['min_output_strength'],
                        })
                    layer_df = pd.DataFrame(layer_summary)

                    layer_csv_path = os.path.join(output_dir, f'{name}_layer_relevance.csv')
                    layer_df.to_csv(layer_csv_path, index=False)
                    print(f"   ✓ Saved {name} layer relevance to {layer_csv_path}")

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
            # Create subdirectory for plots
            plots_dir = os.path.join(output_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
        else:
            output_dir = '.'
            plots_dir = 'plots'
            os.makedirs(plots_dir, exist_ok=True)

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
            plot_path = os.path.join(plots_dir, f'{name}_health_metrics.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"   ✓ Saved {name} health metrics plot to {plot_path}")

            # Plot input relevance (separate figure)
            if 'input_relevance' in result and 'error' not in result['input_relevance']:
                rel_stats = result['input_relevance']
                state_relevance = rel_stats['state_relevance']
                useless_indices = set(rel_stats['useless_state_indices'])

                fig, ax = plt.subplots(1, 1, figsize=(14, 5))
                fig.suptitle(f'Input Feature Relevance: {name.upper()}', fontsize=16, fontweight='bold')

                # Color bars based on whether they're unused
                colors = ['red' if i in useless_indices else 'teal' for i in range(len(state_relevance))]

                ax.bar(range(len(state_relevance)), state_relevance, color=colors, edgecolor='black', linewidth=0.5)
                ax.axhline(rel_stats['threshold'], color='orange', linestyle='--', linewidth=2,
                          label=f"Threshold ({rel_stats['threshold']*100:.0f}% of max)")
                ax.set_xlabel('Input Dimension Index')
                ax.set_ylabel('Normalized Weight Magnitude')
                ax.set_title(f'Input Feature Relevance\n'
                            f'({rel_stats["useless_state_count"]}/{rel_stats["total_state_inputs"]} '
                            f'inputs below threshold - marked in red)')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')

                # Add text annotation with stats
                usage_ratio = 1.0 - (rel_stats['useless_state_count'] / rel_stats['total_state_inputs'])
                ax.text(0.98, 0.98, f'Usage: {usage_ratio*100:.1f}%',
                       transform=ax.transAxes, ha='right', va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                       fontsize=12)

                plt.tight_layout()

                # Save plot
                relevance_plot_path = os.path.join(plots_dir, f'{name}_input_relevance.png')
                plt.savefig(relevance_plot_path, dpi=150, bbox_inches='tight')
                plt.close()

                print(f"   ✓ Saved {name} input relevance plot to {relevance_plot_path}")

            # Plot layer-by-layer relevance (separate figure)
            if 'layer_relevance' in result and 'error' not in result['layer_relevance']:
                layer_rel = result['layer_relevance']
                layer_stats = layer_rel['layer_stats']

                # Create figure with multiple subplots
                fig, axes = plt.subplots(2, 1, figsize=(14, 10))
                fig.suptitle(f'Layer-by-Layer Information Flow: {name.upper()}', fontsize=16, fontweight='bold')

                # Plot 1: Input and output usage ratios per layer
                ax = axes[0]
                layer_names = [s['layer_name'] for s in layer_stats]
                layer_indices = range(len(layer_names))
                input_usage = [s['input_usage_ratio'] * 100 for s in layer_stats]
                output_usage = [s['output_usage_ratio'] * 100 for s in layer_stats]

                x = np.arange(len(layer_indices))
                width = 0.35

                bars1 = ax.bar(x - width/2, input_usage, width, label='Input Usage', color='steelblue', edgecolor='black')
                bars2 = ax.bar(x + width/2, output_usage, width, label='Output Usage', color='coral', edgecolor='black')

                ax.axhline(70, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='70% threshold')
                ax.axhline(85, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='85% threshold')

                ax.set_xlabel('Layer Index')
                ax.set_ylabel('Usage Percentage (%)')
                ax.set_title('Input and Output Usage per Layer\n(higher is better - shows how much of each layer is utilized)')
                ax.set_xticks(x)
                ax.set_xticklabels(layer_indices)
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_ylim(0, 105)

                # Add value labels on bars
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        if height < 70:  # Highlight low usage
                            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                                   f'{height:.0f}%', ha='center', va='bottom', fontsize=8, color='red', fontweight='bold')

                # Plot 2: Average relevance and strength per layer
                ax = axes[1]
                avg_input_rel = [s['avg_input_relevance'] for s in layer_stats]
                avg_output_str = [s['avg_output_strength'] for s in layer_stats]

                ax.plot(layer_indices, avg_input_rel, marker='o', linestyle='-', linewidth=2,
                       label='Avg Input Relevance', color='steelblue', markersize=8)
                ax.plot(layer_indices, avg_output_str, marker='s', linestyle='-', linewidth=2,
                       label='Avg Output Strength', color='coral', markersize=8)

                ax.set_xlabel('Layer Index')
                ax.set_ylabel('Normalized Magnitude')
                ax.set_title('Average Connection Strengths per Layer\n(shows how strongly each layer processes information)')
                ax.set_xticks(layer_indices)
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Add layer names as secondary x-axis labels
                ax2 = ax.twiny()
                ax2.set_xlim(ax.get_xlim())
                ax2.set_xticks(layer_indices)
                ax2.set_xticklabels(layer_names, rotation=45, ha='left', fontsize=8)
                ax2.set_xlabel('Layer Name', fontsize=9)

                plt.tight_layout()

                # Save plot
                layer_flow_plot_path = os.path.join(plots_dir, f'{name}_layer_flow.png')
                plt.savefig(layer_flow_plot_path, dpi=150, bbox_inches='tight')
                plt.close()

                print(f"   ✓ Saved {name} layer flow plot to {layer_flow_plot_path}")

        print(f"\n✅ All plots saved to: {plots_dir}/")

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

  # Save detailed reports and plots (plots saved to health_reports/plots/)
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

Input Feature Relevance:
  • Analyzes first-layer weight magnitudes to identify which inputs matter
  • Relevance = sum of absolute weights connecting to that input dimension
  • Threshold: inputs with <5% of max relevance are flagged as "likely unused"
  • Usage ratios:
    - >85%: EXCELLENT - efficient observation space
    - 70-85%: GOOD - reasonable utilization
    - 50-70%: MODERATE - some waste, consider investigation
    - <50%: POOR - significant waste, reduce observation space
  • Use this to optimize your observation/state design
  • Red bars in plots indicate potentially removable features

Layer-by-Layer Information Flow:
  • Analyzes every layer to understand how information propagates through the network
  • For each layer, tracks:
    - Input usage: which inputs from previous layer are utilized
    - Output strength: which output neurons have strong connections
  • Usage ratios (similar to input relevance):
    - >85%: EXCELLENT - layer is fully utilized
    - 70-85%: GOOD - reasonable layer utilization
    - <70%: BOTTLENECK - layer may be filtering too much information
  • Bottlenecks indicate:
    - Layer may be too small (increase hidden size)
    - Information is being lost through the network
    - Previous layers may be providing redundant information
  • Helps optimize network architecture (layer sizes, depth)
  • Visualizations show usage trends across entire network

For more details, see:
  - WeightWatcher official docs: https://weightwatcher.ai
  - Martin & Mahoney (2019): "Traditional and Heavy-Tailed Self Regularization"
  - WeightWatcher GitHub: https://github.com/CalculatedContent/WeightWatcher
    """)


if __name__ == '__main__':
    main()
