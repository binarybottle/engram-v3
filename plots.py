import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import os
from pathlib import Path
from data.bigram_frequencies_english import (
    onegrams, onegram_frequencies_array,
    bigrams, bigram_frequencies_array
)

# Create output directory
PLOT_DIR = Path('output/plots')
PLOT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Created/verified plot directory at: {PLOT_DIR.absolute()}")

def plot_letter_frequencies():
    print("Generating letter frequencies plot...")
    plt.figure(figsize=(15, 8))
    sns.set_style("whitegrid")
    
    frequencies = onegram_frequencies_array
    
    # Create scatter plot
    plt.scatter(range(len(onegrams)), frequencies * 100, s=100, alpha=0.6)
    plt.plot(range(len(onegrams)), frequencies * 100, 'b-', alpha=0.3)
    
    # Add labels for each point
    for i, (letter, freq) in enumerate(zip(onegrams, frequencies)):
        plt.annotate(f'{letter.upper()}\n{freq*100:.1f}%',
                    (i, freq*100),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center')
    
    plt.title('Letter Frequencies in English Text', fontsize=14, pad=20)
    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('Frequency (%)', fontsize=12)
    plt.xticks([])
    
    plt.tight_layout()
    output_path = PLOT_DIR / 'letter_frequencies.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved letter frequencies plot to: {output_path.absolute()}")
    plt.close()

def plot_bigram_frequencies(n_bigrams=None, filename_suffix=''):
    print(f"Generating bigram frequencies plot{' (top ' + str(n_bigrams) + ')' if n_bigrams else ''}...")
    plt.figure(figsize=(15, 8))
    sns.set_style("whitegrid")
    
    # Sort bigrams by frequency
    frequencies = bigram_frequencies_array
    sorted_indices = np.argsort(frequencies)[::-1]
    if n_bigrams:
        sorted_indices = sorted_indices[:n_bigrams]
    sorted_freqs = frequencies[sorted_indices] * 100
    sorted_bigrams = [bigrams[i] for i in sorted_indices]
    
    # Create scatter plot
    plt.scatter(range(len(sorted_freqs)), sorted_freqs, s=20, alpha=0.4)
    plt.plot(range(len(sorted_freqs)), sorted_freqs, 'b-', alpha=0.2)

    # Calculate indices for labels (20 evenly spaced points)
    if n_bigrams:
        n_labels = min(20, n_bigrams)
    else:
        n_labels = 20
    label_indices = np.linspace(0, len(sorted_freqs) - 1, n_labels).astype(int)
    
    # Add labels
    for i in label_indices:
        if i < len(sorted_freqs):
            bigram = sorted_bigrams[i]
            label = f"{bigram.upper()}\n{sorted_freqs[i]:.2f}%"
            plt.annotate(label,
                        (i, sorted_freqs[i]),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center',
                        fontsize=8)
    
    title_suffix = f" (Top {n_bigrams})" if n_bigrams else ""
    plt.title(f'Bigram Frequencies in English Text{title_suffix}', fontsize=14, pad=20)
    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('Frequency (%)', fontsize=12)
    plt.yscale('log')
    
    plt.tight_layout()
    output_path = PLOT_DIR / f'bigram_frequencies{filename_suffix}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved bigram frequencies plot to: {output_path.absolute()}")
    plt.close()

if __name__ == "__main__":
    plt.rcParams.update({'font.size': 12})
    
    # Generate plots
    plot_letter_frequencies()
    plot_bigram_frequencies(filename_suffix='_all')
    plot_bigram_frequencies(n_bigrams=100, filename_suffix='_top100')
    
    print("\nPlot generation complete!")