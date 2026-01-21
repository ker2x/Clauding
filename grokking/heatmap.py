import matplotlib.pyplot as plt
import torch


def analyze_neuron_waves(model, p, num_to_plot=5):
    """Find and plot the neurons with the strongest periodic signals using FFT analysis."""
    print(f"\n🌊 Analyzing Top {num_to_plot} Neuron Waves...")

    # 1. Get embedding weights (p, dim)
    W = model.token_embed.weight.detach().cpu()[:p, :].float()

    # 2. Find which neurons have the strongest periodic signal
    # Compute FFT per neuron to find the peak power
    fft_vals = torch.fft.rfft(W, dim=0)
    power = fft_vals.abs() ** 2

    # Sum power across all frequencies (excluding DC at index 0)
    # to find the "loudest" neurons
    total_power = power[1:, :].sum(dim=0)
    top_neurons = torch.topk(total_power, num_to_plot).indices

    # 3. Plotting
    plt.figure(figsize=(15, 3 * num_to_plot))

    for i, neuron_idx in enumerate(top_neurons):
        plt.subplot(num_to_plot, 1, i + 1)

        # The wave values for this specific neuron
        wave = W[:, neuron_idx].numpy()

        # Identify the dominant frequency for the label
        dom_freq = power[1:, neuron_idx].argmax().item() + 1

        plt.plot(range(p), wave, marker='o', linestyle='-', markersize=4, color='royalblue')
        plt.title(f"Neuron Index: {neuron_idx} | Dominant Frequency: {dom_freq}")
        plt.xlabel("Input Token (0 to p-1)")
        plt.ylabel("Weight Value")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('neuron_wave_analysis.png')
    print("   -> Saved 'neuron_wave_analysis.png'")
    plt.close()


def visualize_embeddings(model, p):
    print("\n🔬 Visualizing Embedding Weights...")

    # Extract weights: (vocab_size, embed_dim)
    # We detach from the graph and move to CPU for plotting
    W = model.token_embed.weight.detach().cpu().float().numpy()
    
    # 1. Heatmap of the entire embedding matrix
    plt.figure(figsize=(12, 6))
    plt.imshow(W, aspect='auto', cmap='RdBu', interpolation='nearest')
    plt.colorbar(label='Weight Value')
    plt.title(f'Embedding Matrix (The "Fourier" Brain)\nY-axis: Numbers (0-{p}) | X-axis: Embedding Dimensions')
    plt.xlabel('Neuron Dimension')
    plt.ylabel('Input Number (Token)')
    plt.savefig('embedding_heatmap.png')
    print("   -> Saved 'embedding_heatmap.png'")
    plt.close()

    # 2. Line plots of specific neurons (Columns of the matrix)
    # If the model grokked, these lines will look like perfect Sine/Cosine waves!
    plt.figure(figsize=(12, 4))

    # Pick 3 evenly spaced dimensions based on actual embed_dim
    embed_dim = W.shape[1]
    dims_to_plot = [embed_dim // 4, embed_dim // 2, 3 * embed_dim // 4]

    for i, dim in enumerate(dims_to_plot):
        plt.subplot(1, 3, i+1)
        plt.plot(W[:, dim], marker='o', markersize=2, linestyle='-')
        plt.title(f'Neuron #{dim}')
        plt.xlabel('Input Number')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('neuron_waves.png')
    print("   -> Saved 'neuron_waves.png'")
    plt.close()
