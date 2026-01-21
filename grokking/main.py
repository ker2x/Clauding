import torch
import torch.nn as nn
import torch.optim as optim
import torch.fft
import random
import matplotlib.pyplot as plt
import numpy as np
import sys
import signal
import os
import argparse
from heatmap import visualize_embeddings, analyze_neuron_waves

# Flag for SIGINFO handler
generate_images_requested = False

def siginfo_handler(signum, frame):
    global generate_images_requested
    generate_images_requested = True
    print("\n📸 SIGINFO received - will generate images after current step...")


def calc_snr(W):
    """Calculate Signal-to-Noise Ratio using Fourier analysis."""
    # FFT along token dimension (dim 0)
    # Convert to float32 for FFT (bfloat16 not supported)
    fft_vals = torch.fft.rfft(W.cpu().float(), dim=0)
    power = fft_vals.abs() ** 2

    # For each neuron, find dominant frequency (skip DC at index 0)
    dominant_idx = power[1:, :].argmax(dim=0) + 1  # +1 to account for skipping DC

    # Calculate signal power (power at dominant frequency for each neuron)
    signal_power = power[dominant_idx, torch.arange(power.size(1))].sum()

    # Calculate total power (excluding DC)
    total_power = power[1:, :].sum()

    # Noise = total - signal
    noise_power = total_power - signal_power

    # SNR in dB (avoid division by zero)
    if noise_power > 1e-10:
        snr_db = 10 * torch.log10(signal_power / noise_power)
    else:
        snr_db = torch.tensor(float('inf'))

    return snr_db.item()


def print_weight_stats(model):
    """Diagnostic: print weight statistics to detect collapse."""
    print("\n📊 Weight Statistics:")
    W_emb = model.token_embed.weight.detach()
    W_head = model.head.weight.detach()
    print(f"   Embedding: min={W_emb.min():.4f}, max={W_emb.max():.4f}, std={W_emb.std():.4f}")
    print(f"   Head:      min={W_head.min():.4f}, max={W_head.max():.4f}, std={W_head.std():.4f}")

    # Calculate SNR using the standalone function
    emb_snr = calc_snr(W_emb)
    head_snr = calc_snr(W_head)
    print(f"   Embedding SNR: {emb_snr:.2f} dB")
    print(f"   Head SNR:      {head_snr:.2f} dB")


def visualize_sorted_embeddings(model, p):
    print("\n🧹 Sorting Neurons by Frequency...")

    # Get weights: (p, embed_dim)
    # Note: We slice [0:p] to ignore the '=' token if you have one,
    # or just take the whole thing if vocab is exactly p.
    W = model.token_embed.weight.detach().cpu()[:p, :]

    # 1. Calculate Fourier Transform of each neuron (column)
    # This tells us which frequency each neuron is "singing" at.
    # W is (p, dim) -> fft over dim 0
    fft_vals = torch.fft.rfft(W.float(), dim=0)

    # 2. Find the dominant frequency for each neuron
    # We take the magnitude (abs) and find the index of the max value
    # We skip index 0 (DC component/average value) to find the oscillation
    amplitudes = fft_vals.abs()
    dominant_freqs = amplitudes[1:, :].argmax(dim=0)

    # 3. Sort the neurons (columns) based on these frequencies
    sorted_indices = torch.argsort(dominant_freqs)
    W_sorted = W.float()[:, sorted_indices].numpy()

    # 4. Plot the "Unshuffled" Brain
    plt.figure(figsize=(10, 8))
    plt.imshow(W_sorted, aspect='auto', cmap='RdBu', interpolation='nearest')
    plt.colorbar(label='Weight Value')
    plt.title(f'Sorted Embedding Matrix (The "Grokking Rainbow")\nNeurons re-ordered by frequency')
    plt.xlabel('Neuron Dimension (Sorted by Frequency)')
    plt.ylabel('Input Number (0-96)')

    plt.savefig('sorted_heatmap.png')
    print("   -> Saved 'sorted_heatmap.png' (The Money Shot 📸)")
    plt.close()


def visualize_perfect_rainbow(model, p):
    print("\n🌈 Generating the Perfect Rainbow...")

    # 1. Get the weights
    # We detach and move to CPU
    W = model.token_embed.weight.detach().cpu()[:p, :]  # Shape: (p, dim)

    # 2. Fourier Transform to find frequencies
    # We want to sort neurons (columns) by their frequency
    fft_vals = torch.fft.rfft(W.float(), dim=0)
    amplitudes = fft_vals.abs()

    # Get the dominant frequency index for each neuron
    # Skip index 0 (DC component)
    dominant_freqs = amplitudes[1:, :].argmax(dim=0)

    # 3. Sort the columns (neurons)
    sorted_indices = torch.argsort(dominant_freqs)
    W_sorted = W.float()[:, sorted_indices].numpy()

    # 4. Plot
    plt.figure(figsize=(10, 6))
    plt.imshow(W_sorted, aspect='auto', cmap='RdBu', interpolation='bilinear')  # 'bilinear' makes it look smoother like the video
    plt.colorbar(label='Weight Value')
    plt.title(f'The "Grokking Rainbow" (Sorted by Frequency)\nThis is the algorithm your model learned.')
    plt.xlabel('Neuron Dimension (Low Freq -> High Freq)')
    plt.ylabel(f'Input Token (0-{p-1})')

    plt.tight_layout()
    plt.savefig('perfect_rainbow.png', dpi=300)  # High DPI for "excessive" quality
    print("   -> Saved 'perfect_rainbow.png'")
    plt.close()


def visualize_power_spectrum(model, p):
    print("\n⚡ Visualizing Fourier Power Spectrum...")

    # 1. Get the raw weights from the embedding layer
    # Shape: (p, embed_dim) -> e.g., (100, 256)
    W = model.token_embed.weight.detach().cpu()

    # 2. Compute 1D Fourier Transform down the columns
    # We want to know which frequencies exist in the neurons
    fft = torch.fft.rfft(W.float(), dim=0)

    # 3. Compute Power (Magnitude squared)
    # Power = How "strong" is this frequency?
    power = fft.abs() ** 2

    # 4. Sort neurons by their "Peak Frequency" to make the plot readable
    # We skip index 0 (DC component/average)
    peak_freqs = power[1:, :].argmax(dim=0)
    sorted_indices = torch.argsort(peak_freqs)

    # Re-order the power matrix
    power_sorted = power[:, sorted_indices].numpy()

    # 5. Log Scale: Essential because the "signal" is often 1000x stronger than noise
    power_log = np.log(power_sorted + 1e-9)

    # 6. Plotting
    plt.figure(figsize=(10, 6))

    # 'nearest' interpolation keeps the pixels sharp so you can see the "grid"
    plt.imshow(power_log, aspect='auto', cmap='inferno', interpolation='nearest', origin='lower')

    plt.colorbar(label='Log Power (Brightness = Strength)')
    plt.title(f'The "Epicycles" of Modulo {p}\n(Bright Spots = The "Gears" the model uses)')
    plt.xlabel('Neuron (Sorted by Main Frequency)')
    plt.ylabel(f'Frequency (0 to {p//2})')

    # Add helpful ticks to identify the "Gears"
    # For p=100, we expect action at 2, 4, 5, 10, 20, 25, 50
    plt.yticks(ticks=np.arange(0, p//2 + 1, 5))

    plt.tight_layout()
    plt.savefig('grok_power_spectrum.png', dpi=300)
    print("   -> Saved 'grok_power_spectrum.png'")
    plt.close()


def visualize_unembedding(model, p):
    print("\n📣 Visualizing Un-embedding (The Output Layer)...")

    # 1. Get the weights from the final Linear layer
    # Shape in PyTorch Linear is (out_features, in_features) -> (p, embed_dim)
    # This matches the shape of our input embedding matrix!
    W_out = model.head.weight.detach().cpu()

    # --- Plot 1: The "Output Rainbow" (Sorted) ---
    # We use the same logic as the input: sort neurons by their dominant frequency

    fft_vals = torch.fft.rfft(W_out.float(), dim=0)
    amplitudes = fft_vals.abs()

    # Find dominant freq for each neuron (skip DC at index 0)
    dominant_freqs = amplitudes[1:, :].argmax(dim=0)
    sorted_indices = torch.argsort(dominant_freqs)
    W_sorted = W_out.float()[:, sorted_indices].numpy()

    plt.figure(figsize=(10, 6))
    plt.imshow(W_sorted, aspect='auto', cmap='RdBu', interpolation='nearest')
    plt.colorbar(label='Weight Value')
    plt.title(f'The Un-embedding "Rainbow"\n(How the model translates waves back to numbers)')
    plt.xlabel('Neuron Dimension (Sorted by Frequency)')
    plt.ylabel(f'Output Logit (0-{p-1})')
    plt.savefig('unembedding_rainbow.png')
    print("   -> Saved 'unembedding_rainbow.png'")
    plt.close()

    # --- Plot 2: The "Output Tuner" (Power Spectrum) ---
    # We want to see WHICH frequencies the output layer is listening for.

    # Power = Magnitude squared
    power = fft_vals.abs() ** 2

    # Sort by peak frequency to organize the plot (same visualization as input)
    peak_freqs = power[1:, :].argmax(dim=0)
    sorted_indices_spec = torch.argsort(peak_freqs)
    power_sorted = power[:, sorted_indices_spec].numpy()

    # Log scale
    power_log = np.log(power_sorted + 1e-9)

    plt.figure(figsize=(10, 6))
    plt.imshow(power_log, aspect='auto', cmap='inferno', interpolation='nearest', origin='lower')
    plt.colorbar(label='Log Power')
    plt.title(f'Un-embedding Power Spectrum\n(Bright Spots = The Frequencies the Output "Hears")')
    plt.xlabel('Neuron (Sorted)')
    plt.ylabel(f'Frequency (0-{p//2})')
    plt.yticks(ticks=np.arange(0, p//2 + 1, 5))

    plt.savefig('unembedding_spectrum.png')
    print("   -> Saved 'unembedding_spectrum.png'")
    plt.close()

    # --- Plot 3: The "Handshake" (Input vs Output Correlation) ---
    # Are the input and output weights doing the same thing?
    # We compute the Cosine Similarity between Input Matrix and Output Matrix

    W_in = model.token_embed.weight.detach().cpu().float()  # (p, dim)
    W_out_f = W_out.float()

    # Normalize vectors
    W_in_norm = W_in / W_in.norm(dim=1, keepdim=True)
    W_out_norm = W_out_f / W_out_f.norm(dim=1, keepdim=True)

    # Similarity matrix: (p, p)
    # If perfect: Diagonal should be bright (Input 5 correlates with Output 5)
    similarity = torch.mm(W_in_norm, W_out_norm.T).numpy()

    # Create figure with 2D heatmap and 3D surface side by side
    fig = plt.figure(figsize=(16, 7))

    # Left: 2D Heatmap
    ax1 = fig.add_subplot(1, 2, 1)
    im = ax1.imshow(similarity, cmap='RdBu', origin='lower')
    fig.colorbar(im, ax=ax1, label='Cosine Similarity')
    ax1.set_title(f'Input vs. Output "Handshake"\n(Diagonal = The Model "Remembers" the Number)')
    ax1.set_xlabel('Output Token')
    ax1.set_ylabel('Input Token')

    # Right: 3D Surface
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    X, Y = np.meshgrid(np.arange(p), np.arange(p))
    surf = ax2.plot_surface(X, Y, similarity, cmap='RdBu', edgecolor='none', alpha=0.9)
    ax2.set_title('3D Surface View')
    ax2.set_xlabel('Output Token')
    ax2.set_ylabel('Input Token')
    ax2.set_zlabel('Similarity')
    ax2.view_init(elev=30, azim=45)

    plt.tight_layout()
    plt.savefig('io_handshake.png', dpi=150)
    print("   -> Saved 'io_handshake.png'")
    plt.close()


def visualize_embedding_cosine_distance(model, p):
    print("\n📐 Visualizing Embedding Cosine Distance...")

    # Get embedding weights
    W = model.token_embed.weight.detach().cpu()[:p, :].float()  # (p, embed_dim)

    # Normalize to unit vectors
    W_norm = W / W.norm(dim=1, keepdim=True)

    # Cosine similarity matrix: (p, p)
    cos_sim = torch.mm(W_norm, W_norm.T)

    # Cosine distance = 1 - cosine similarity
    cos_dist = (1 - cos_sim).numpy()

    plt.figure(figsize=(10, 8))
    plt.imshow(cos_dist, cmap='viridis', origin='lower')
    plt.colorbar(label='Cosine Distance')
    plt.title(f'Embedding Cosine Distance (p={p})\n(Dark = Similar, Bright = Different)')
    plt.xlabel('Token')
    plt.ylabel('Token')

    plt.tight_layout()
    plt.savefig('embedding_cosine_distance.png', dpi=150)
    print("   -> Saved 'embedding_cosine_distance.png'")
    plt.close()


def visualize_confidence_distribution(model, p, device):
    print("\n🎯 Visualizing Confidence Distribution...")

    model.eval()
    with torch.inference_mode():
        # Create all possible inputs
        a_vals = torch.arange(p, device=device)
        b_vals = torch.arange(p, device=device)

        # We'll create visualizations for a few specific 'a' values
        # Plus an average view
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Pick representative 'a' values
        sample_a = [0, p//4, p//2, 3*p//4, p-1]

        # Collect all distributions for averaging
        all_probs_aligned = []

        for idx, a in enumerate(sample_a):
            # Create inputs: (a, 0), (a, 1), ..., (a, p-1)
            inputs = torch.stack([
                torch.full((p,), a, dtype=torch.long, device=device),
                b_vals
            ], dim=1)  # (p, 2)

            logits = model(inputs)  # (p, p)
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()  # (p, p)

            # Plot: x=b, y=output, color=probability
            ax = axes[idx // 3, idx % 3]
            im = ax.imshow(probs.T, aspect='auto', cmap='hot', origin='lower',
                          vmin=0, vmax=1)
            ax.set_title(f'a = {a}')
            ax.set_xlabel('b (second operand)')
            ax.set_ylabel('Output logit')

            # Draw the "correct answer" line
            correct = [(a + b) % p for b in range(p)]
            ax.plot(range(p), correct, 'c-', linewidth=1, alpha=0.7, label='Correct')

            # Collect aligned probabilities (shift so correct answer is at center)
            for b in range(p):
                correct_ans = (a + b) % p
                shifted = np.roll(probs[b], p//2 - correct_ans)
                all_probs_aligned.append(shifted)

        # Last subplot: averaged aligned distribution
        ax = axes[1, 2]
        avg_aligned = np.mean(all_probs_aligned, axis=0)

        # Show as bar chart centered on correct answer
        x_shifted = np.arange(p) - p//2
        ax.bar(x_shifted, avg_aligned, color='steelblue', width=1.0)
        ax.axvline(0, color='red', linestyle='--', label='Correct answer')
        ax.set_title('Average Doubt Pattern\n(Aligned to correct answer)')
        ax.set_xlabel('Distance from correct answer')
        ax.set_ylabel('Average probability')
        ax.set_xlim(-p//2, p//2)
        ax.legend()

        plt.tight_layout()
        plt.savefig('confidence_distribution.png', dpi=150)
        print("   -> Saved 'confidence_distribution.png'")
        plt.close()

        # Second figure: Full p×p grid showing ALL inputs
        # Each row is one input (a, b), showing distribution over outputs
        print("   Generating full confidence matrix...")

        # Create all p^2 inputs
        all_inputs = torch.stack([
            a_vals.repeat_interleave(p),
            b_vals.repeat(p)
        ], dim=1)  # (p^2, 2)

        all_logits = model(all_inputs)  # (p^2, p)
        all_probs = torch.softmax(all_logits.float(), dim=-1).cpu().numpy()  # (p^2, p)

        # Reshape to (p, p, p): [a, b, output]
        prob_cube = all_probs.reshape(p, p, p)

        # Create a visualization: for each 'a', show b vs output heatmap
        # Stack them horizontally
        fig, axes = plt.subplots(1, 1, figsize=(14, 10))

        # Average over 'a' to get a single b vs output view
        avg_over_a = prob_cube.mean(axis=0)  # (p, p): [b, output]

        im = axes.imshow(avg_over_a.T, aspect='auto', cmap='hot', origin='lower')
        axes.set_title('Average Confidence: b vs Output\n(Averaged over all a values)')
        axes.set_xlabel('b (second operand)')
        axes.set_ylabel('Output token')
        plt.colorbar(im, ax=axes, label='Probability')

        # The "correct" answers form diagonal lines (one for each 'a')
        # For average, we expect a smeared diagonal

        plt.tight_layout()
        plt.savefig('confidence_matrix_avg.png', dpi=150)
        print("   -> Saved 'confidence_matrix_avg.png'")
        plt.close()


def save_video_frame(model, p, step, device, output_dir, train_acc=None, val_acc=None, loss=None):
    """Save a combined frame for video generation showing grokking progression."""
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(18, 18))  # Larger for 3x3 grid

    # --- Panel 1: Raw Embedding Matrix ---
    ax1 = fig.add_subplot(3, 3, 1)
    W = model.token_embed.weight.detach().cpu()[:p, :].float()
    im1 = ax1.imshow(W.numpy(), aspect='auto', cmap='RdBu', interpolation='bilinear')
    ax1.set_title('Embedding Matrix')
    ax1.set_xlabel('Neuron')
    ax1.set_ylabel('Input Token')
    fig.colorbar(im1, ax=ax1)

    # --- Panel 2: Power Spectrum ---
    ax2 = fig.add_subplot(3, 3, 2)
    fft_vals = torch.fft.rfft(W, dim=0)
    power = fft_vals.abs() ** 2
    power_log = np.log(power.numpy() + 1e-9)
    im2 = ax2.imshow(power_log, aspect='auto', cmap='inferno', interpolation='nearest', origin='lower')
    ax2.set_title('Power Spectrum')
    ax2.set_xlabel('Neuron')
    ax2.set_ylabel('Frequency')
    fig.colorbar(im2, ax=ax2)

    # --- Panel 3: IO Handshake ---
    ax3 = fig.add_subplot(3, 3, 3)
    W_in = W  # Already have it
    W_out = model.head.weight.detach().cpu()[:p, :].float()
    W_in_norm = W_in / W_in.norm(dim=1, keepdim=True)
    W_out_norm = W_out / W_out.norm(dim=1, keepdim=True)
    similarity = torch.mm(W_in_norm, W_out_norm.T).numpy()
    im3 = ax3.imshow(similarity, cmap='RdBu', origin='lower')
    ax3.set_title('IO Handshake')
    ax3.set_xlabel('Output Token')
    ax3.set_ylabel('Input Token')
    fig.colorbar(im3, ax=ax3)

    # --- Panels 4 & 5: Attention Patterns (both heads) ---
    embed_dim = CONFIG['embed_dim']
    num_heads = CONFIG['num_heads']
    head_dim = embed_dim // num_heads

    layer = model.transformer.layers[0]
    W_qkv = layer.self_attn.in_proj_weight.detach().cpu().float()
    W_q = W_qkv[:embed_dim, :]
    W_k = W_qkv[embed_dim:2*embed_dim, :]
    E = W  # Embedding matrix

    Q_all = E @ W_q.T
    K_all = E @ W_k.T
    Q_heads = Q_all.view(p, num_heads, head_dim)
    K_heads = K_all.view(p, num_heads, head_dim)
    attn_scores = torch.einsum('ihd, jhd -> hij', Q_heads, K_heads)
    attn_scores = attn_scores / (head_dim ** 0.5)

    for h in range(min(num_heads, 2)):  # Show up to 2 heads
        ax = fig.add_subplot(3, 3, 4 + h)
        im = ax.imshow(attn_scores[h].numpy(), cmap='RdBu', origin='lower')
        ax.set_title(f'Attention Head {h}')
        ax.set_xlabel('Key Token')
        ax.set_ylabel('Query Token')
        fig.colorbar(im, ax=ax)

    # --- Panel 6: Doubt Pattern ---
    ax6 = fig.add_subplot(3, 3, 6)
    model.eval()
    with torch.inference_mode():
        # Create all p^2 inputs for full evaluation
        a_vals = torch.arange(p, device=device)
        b_vals = torch.arange(p, device=device)
        all_inputs = torch.stack([
            a_vals.repeat_interleave(p),
            b_vals.repeat(p)
        ], dim=1)  # (p^2, 2)

        all_logits = model(all_inputs)  # (p^2, p)
        all_probs = torch.softmax(all_logits.float(), dim=-1)  # (p^2, p)

        # Compute correct answers for all inputs
        all_a = all_inputs[:, 0]
        all_b = all_inputs[:, 1]
        correct_answers = (all_a + all_b) % p  # (p^2,)

        # --- Doubt pattern (sampled) ---
        all_probs_aligned = []
        for i in range(0, p * p, max(1, p)):  # Sample every p-th input
            correct_ans = correct_answers[i].item()
            prob_row = all_probs[i].cpu().numpy()
            shifted = np.roll(prob_row, p // 2 - correct_ans)
            all_probs_aligned.append(shifted)

        avg_aligned = np.mean(all_probs_aligned, axis=0)
        x_shifted = np.arange(p) - p // 2
        ax6.bar(x_shifted, avg_aligned, color='steelblue', width=1.0)
        ax6.axvline(0, color='red', linestyle='--', linewidth=2)
        ax6.set_title('Doubt Pattern')
        ax6.set_xlabel('Distance from correct')
        ax6.set_ylabel('Probability')
        ax6.set_xlim(-p // 2, p // 2)

        # --- Panel 7: P(correct | a, b) heatmap ---
        ax7 = fig.add_subplot(3, 3, 7)
        # Extract probability of correct answer for each input
        p_correct = all_probs[torch.arange(p * p, device=device), correct_answers]  # (p^2,)
        p_correct_grid = p_correct.cpu().numpy().reshape(p, p)  # (p, p) where [a, b] = P(correct | a, b)
        im7 = ax7.imshow(p_correct_grid, aspect='auto', cmap='viridis', origin='lower', vmin=0, vmax=1)
        ax7.set_title('P(correct | a, b)')
        ax7.set_xlabel('b')
        ax7.set_ylabel('a')
        fig.colorbar(im7, ax=ax7)

        # --- Panel 8: Entropy heatmap ---
        ax8 = fig.add_subplot(3, 3, 8)
        # Entropy: -sum(p * log(p))
        entropy = -(all_probs * torch.log(all_probs + 1e-9)).sum(dim=-1)  # (p^2,)
        entropy_grid = entropy.cpu().numpy().reshape(p, p)
        max_entropy = np.log(p)  # Maximum entropy for uniform distribution
        im8 = ax8.imshow(entropy_grid, aspect='auto', cmap='plasma', origin='lower', vmin=0, vmax=max_entropy)
        ax8.set_title(f'Entropy (max={max_entropy:.2f})')
        ax8.set_xlabel('b')
        ax8.set_ylabel('a')
        fig.colorbar(im8, ax=ax8)

        # --- Panel 9: Logit magnitude histogram ---
        ax9 = fig.add_subplot(3, 3, 9)
        logits_flat = all_logits.cpu().numpy().flatten()
        logit_max = np.abs(logits_flat).max()
        logit_mean = np.abs(logits_flat).mean()
        logit_std = logits_flat.std()

        ax9.hist(logits_flat, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax9.axvline(logits_flat.mean(), color='red', linestyle='--', label=f'mean={logits_flat.mean():.1f}')
        ax9.set_title(f'Logit Distribution (max={logit_max:.1f}, std={logit_std:.1f})')
        ax9.set_xlabel('Logit value')
        ax9.set_ylabel('Count')
        ax9.legend(fontsize=8)

    # --- Main title with stats ---
    # Calculate SNR for embedding
    W_emb = model.token_embed.weight.detach()
    emb_snr = calc_snr(W_emb[:p, :])

    # Calculate mean entropy and logit magnitude for title
    mean_entropy = entropy.mean().item()
    mean_p_correct = p_correct.mean().item()

    stats_parts = [f'Step {step:,}']
    if train_acc is not None:
        stats_parts.append(f'Train: {train_acc*100:.1f}%')
    if val_acc is not None:
        stats_parts.append(f'Val: {val_acc*100:.1f}%')
    if loss is not None:
        stats_parts.append(f'Loss: {loss:.4f}')
    stats_parts.append(f'SNR: {emb_snr:.1f}dB')
    stats_parts.append(f'H: {mean_entropy:.2f}')
    stats_parts.append(f'|L|: {logit_max:.1f}')
    fig.suptitle(' | '.join(stats_parts), fontsize=14, fontweight='bold')

    plt.tight_layout()
    frame_path = os.path.join(output_dir, f'frame_{step:07d}.png')
    plt.savefig(frame_path, dpi=100)
    plt.close()


# --- Configuration ---
CONFIG = {
    'p': 59,
    'frac_train': 0.3,  
    'embed_dim': 32,     
    'num_layers': 2,
    'num_heads': 2,
    'lr': 1e-2,             
    'weight_decay': 0.1,  
    'steps': 300000,       
    'batch_size': 1044,     
    'seed': random.randint(0, 2**32 - 1)
}


def visualize_attention_patterns(model, p):
    print("\n👁️ Visualizing Global Attention Patterns...")

    # 1. Extract the Q and K weights from the first layer
    # Note: This assumes standard PyTorch TransformerEncoderLayer structure
    # The weights are packed in in_proj_weight: [WxQ, WxK, WxV]
    layer = model.transformer.layers[0]
    W_qkv = layer.self_attn.in_proj_weight.detach().cpu().float()  # (3*dim, dim)
    embed_dim = CONFIG['embed_dim']
    num_heads = CONFIG['num_heads']
    head_dim = embed_dim // num_heads

    # Slice out W_Q and W_K
    W_q = W_qkv[:embed_dim, :]  # (dim, dim)
    W_k = W_qkv[embed_dim:2*embed_dim, :]  # (dim, dim)

    # 2. Get the Embedding Matrix (The Vocabulary)
    # shape: (p, dim)
    E = model.token_embed.weight.detach().cpu()[:p, :].float()

    # 3. Compute Query and Key vectors for ALL numbers
    # Q = E @ W_Q^T
    Q_all = E @ W_q.T  # (p, dim)
    K_all = E @ W_k.T  # (p, dim)

    # 4. Reshape to split by Heads
    # (p, num_heads, head_dim)
    Q_heads = Q_all.view(p, num_heads, head_dim)
    K_heads = K_all.view(p, num_heads, head_dim)

    # 5. Compute Attention Scores (The "Compatibility" Matrix)
    # Score[h, i, j] = Q[i] dot K[j]
    # This shows how much number 'i' wants to attend to number 'j'

    # We use einsum for clarity:
    # i = query token index
    # j = key token index
    # h = head index
    # d = head dimension
    attn_scores = torch.einsum('ihd, jhd -> hij', Q_heads, K_heads)

    # Scale by sqrt(head_dim) for standard attention scaling
    attn_scores = attn_scores / (head_dim ** 0.5)

    # 6. Plotting
    fig, axes = plt.subplots(1, num_heads, figsize=(4 * num_heads, 4))
    if num_heads == 1:
        axes = [axes]

    for h in range(num_heads):
        ax = axes[h]
        # We look at the raw scores (pre-softmax) to see the full structure
        # Use 'RdBu' to see positive (blue) vs negative (red) alignment
        heatmap = ax.imshow(attn_scores[h].numpy(), cmap='RdBu', origin='lower')
        ax.set_title(f'Head {h} QK Circuit')
        ax.set_xlabel('Key Token (b)')
        ax.set_ylabel('Query Token (a)')

    plt.tight_layout()
    plt.savefig('attention_patterns.png')
    print("   -> Saved 'attention_patterns.png'")
    plt.close()


# --- 1. The Dataset ---
def make_dataset(p, frac_train):
    pairs = [(i, j, (i + j) % p) for i in range(p) for j in range(p)]
    random.shuffle(pairs)
    num_train = int(len(pairs) * frac_train)
    
    # Simple split
    train_data = pairs[:num_train]
    val_data = pairs[num_train:]
    
    def to_tensor(data):
        x = torch.tensor([(a, b) for a, b, c in data], dtype=torch.long)
        y = torch.tensor([c for a, b, c in data], dtype=torch.long)
        return x, y

    return to_tensor(train_data), to_tensor(val_data)

# --- 2. The Model ---
class CausalTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, context_len=2):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, context_len, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=4 * embed_dim, 
	    dropout=0.1,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.size()
        x = self.token_embed(idx) + self.pos_embed[:, :T, :]
        x = self.transformer(x)
        x = self.ln_f(x)
        logits = self.head(x[:, -1, :]) 
        return logits

# --- 3. Plotting Function ---
def plot_results(history):
    steps = [h['step'] for h in history]
    loss = [h['loss'] for h in history]
    lr = [h['lr'] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot (log scale)
    ax1.plot(steps, loss, color='blue', linewidth=1.5)
    ax1.set_yscale('log')
    ax1.set_xlabel('Optimization Steps')
    ax1.set_ylabel('Loss (log scale)')
    ax1.set_title(f"Training Loss (p={CONFIG['p']})")
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    # LR plot
    ax2.plot(steps, lr, color='green', linewidth=1.5)
    ax2.set_xlabel('Optimization Steps')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    plt.tight_layout()
    filename = 'grokking_curve.png'
    plt.savefig(filename)
    print(f"\n📊 Plot saved to {filename}")
    plt.close()

# --- 4. Main Training Loop ---
def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description='Grokking experiment on modular arithmetic')
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'],
                        help='Floating point precision: fp32 (default), fp16, or bf16')
    parser.add_argument('--video-interval', type=int, default=0,
                        help='Save video frames every N steps (0 = disabled)')
    args = parser.parse_args()

    # Register SIGINFO handler (Ctrl+T on macOS)
    signal.signal(signal.SIGINFO, siginfo_handler)

    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using MPS (Apple Silicon).")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using CUDA.")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU.")

    # Precision setup
    if args.precision == 'fp16':
        dtype = torch.float16
        print(f"🔢 Using float16 precision")
    elif args.precision == 'bf16':
        dtype = torch.bfloat16
        print(f"🔢 Using bfloat16 precision")
    else:
        dtype = torch.float32
        print(f"🔢 Using float32 precision")

    # Check for checkpoint and restore seed BEFORE creating dataset
    checkpoint_path = 'checkpoint.pt'
    checkpoint = None
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # Restore seed from checkpoint to recreate same train/val split
        if 'config' in checkpoint and 'seed' in checkpoint['config']:
            CONFIG['seed'] = checkpoint['config']['seed']
            print(f"📂 Found checkpoint, restoring seed {CONFIG['seed']}")

    torch.manual_seed(CONFIG['seed'])
    random.seed(CONFIG['seed'])  # Also seed random module for dataset shuffling

    # Data
    (train_x, train_y), (val_x, val_y) = make_dataset(CONFIG['p'], CONFIG['frac_train'])
    
    train_x, train_y = train_x.to(device), train_y.to(device)
    val_x, val_y = val_x.to(device), val_y.to(device)

    # Model
    model = CausalTransformer(
        vocab_size=CONFIG['p'],
        embed_dim=CONFIG['embed_dim'],
        num_heads=CONFIG['num_heads'],
        num_layers=CONFIG['num_layers']
    ).to(device=device, dtype=dtype)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay'],
        betas=(0.9, 0.98)
    )

    # Cosine annealing: lowers LR from 1e-3 to eta_min (not zero, so weights can keep refining)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['steps'], eta_min=1e-7
    )

    criterion = nn.CrossEntropyLoss()
    history = []
    start_step = 0

    # Checkpoint loading (checkpoint was already loaded earlier for seed restoration)
    if checkpoint is not None:
        print(f"📂 Found checkpoint at '{checkpoint_path}', resuming...")
        # Load state dict, then ensure model is in the correct dtype
        model.load_state_dict(checkpoint['model'])
        model.to(dtype=dtype)  # Ensure dtype matches current setting
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_step = checkpoint['step'] + 1
        history = checkpoint['history']
        saved_precision = checkpoint.get('precision', 'fp32')
        if saved_precision != args.precision:
            print(f"   ⚠️ Note: checkpoint was saved with {saved_precision}, now using {args.precision}")
        print(f"   Resumed from step {start_step}")
    else:
        print("📂 No checkpoint found, starting fresh.")

    # Video frame setup
    video_dir = 'video_frames'
    if args.video_interval > 0:
        os.makedirs(video_dir, exist_ok=True)
        print(f"🎬 Video mode: saving frames every {args.video_interval} steps to '{video_dir}/'")

    print("-" * 65)
    print(f"{'Step':<8} | {'Train Acc':<10} | {'Val Acc':<10} | {'Loss':<10} | {'LR':<10}")
    print("-" * 65)

    # Pre-allocate batch index tensor
    ix = torch.empty(CONFIG['batch_size'], dtype=torch.long, device=device)
    step = start_step  # Initialize for checkpoint saving in case of early interrupt

    try:
        for step in range(start_step, CONFIG['steps']):
            model.train()
            #ix.random_(0, train_x.size(0))
            #xb, yb = train_x[ix], train_y[ix]
            xb, yb = train_x, train_y
            
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Check for SIGINFO request
            global generate_images_requested
            if generate_images_requested:
                generate_images_requested = False
                print(f"\n📸 Generating images at step {step}...")
                model.eval()
                print_weight_stats(model)
                plot_results(history)
                visualize_embeddings(model, CONFIG['p'])
                visualize_sorted_embeddings(model, CONFIG['p'])
                visualize_perfect_rainbow(model, CONFIG['p'])
                visualize_power_spectrum(model, CONFIG['p'])
                visualize_attention_patterns(model, CONFIG['p'])
                visualize_unembedding(model, CONFIG['p'])
                visualize_embedding_cosine_distance(model, CONFIG['p'])
                visualize_confidence_distribution(model, CONFIG['p'], device)
                print("📸 Done! Resuming training...\n")
                model.train()

            # Eval every 100 steps (or at video interval if more frequent)
            save_frame = args.video_interval > 0 and step % args.video_interval == 0
            if step % 100 == 0 or save_frame:
                model.eval()
                with torch.inference_mode():
                    train_logits = model(train_x)
                    val_logits = model(val_x)

                    t_acc = (train_logits.argmax(-1) == train_y).float().mean().item()
                    v_acc = (val_logits.argmax(-1) == val_y).float().mean().item()
                    current_lr = scheduler.get_last_lr()[0]

                    # Only log to history at regular intervals
                    if step % 100 == 0:
                        history.append({'step': step, 'train_acc': t_acc, 'val_acc': v_acc, 'loss': loss.item(), 'lr': current_lr})

                    # Console log every 500 steps
                    if step % 500 == 0:
                        print(f"{step:<8} | {t_acc*100:<8.1f}% | {v_acc*100:<8.1f}% | {loss.item():<10.4f} | {current_lr:<10.2e}")

                    # Save video frame at specified interval
                    if save_frame:
                        save_video_frame(model, CONFIG['p'], step, device, video_dir,
                                        train_acc=t_acc, val_acc=v_acc, loss=loss.item())

    except KeyboardInterrupt:
        print("\n🛑 Training stopped by user. Generating plot for current progress...")

    print_weight_stats(model)
    plot_results(history)
    visualize_embeddings(model, CONFIG['p'])
    visualize_sorted_embeddings(model, CONFIG['p'])
    visualize_perfect_rainbow(model, CONFIG['p'])
    visualize_power_spectrum(model, CONFIG['p'])
    visualize_attention_patterns(model, CONFIG['p'])
    visualize_unembedding(model, CONFIG['p'])
    visualize_embedding_cosine_distance(model, CONFIG['p'])
    visualize_confidence_distribution(model, CONFIG['p'], device)

    # Save final video frame
    if args.video_interval > 0:
        # Get final accuracies
        model.eval()
        with torch.inference_mode():
            train_logits = model(train_x)
            val_logits = model(val_x)
            t_acc = (train_logits.argmax(-1) == train_y).float().mean().item()
            v_acc = (val_logits.argmax(-1) == val_y).float().mean().item()
            final_loss = history[-1]['loss'] if history else 0.0
        save_video_frame(model, CONFIG['p'], step, device, video_dir,
                        train_acc=t_acc, val_acc=v_acc, loss=final_loss)

    # Save checkpoint
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step': step,
        'history': history,
        'config': CONFIG,
        'precision': args.precision
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Checkpoint saved to '{checkpoint_path}'")

    # Print ffmpeg command if video frames were saved
    if args.video_interval > 0:
        frame_count = len([f for f in os.listdir(video_dir) if f.startswith('frame_') and f.endswith('.png')])
        if frame_count > 0:
            print(f"\n🎬 Video frames saved: {frame_count} frames in '{video_dir}/'")
            print("   To compile video, run:")
            print(f"   ffmpeg -framerate 30 -pattern_type glob -i '{video_dir}/frame_*.png' -c:v libx264 -pix_fmt yuv420p grokking.mp4")

if __name__ == "__main__":
    main()
