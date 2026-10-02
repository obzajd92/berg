import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import re

# 1. Mock Global Profile Vectors for Configuration (Baseline vs Person Vectors)
global_metrics_profile = {
    'Poem_Baseline': [120, 450, 0.85],
    'Person_A':      [140, 420, 0.70],
    'Person_B':      [95,  490, 0.92],
    'Person_R':      [118, 455, 0.83],
    'Person_S':      [135, 410, 0.65]
}
labels = ['Word_Freq', 'words_Mean', 'Token_Density']
df_ref = pd.DataFrame(global_metrics_profile, index=labels)

# Compute the base inverse covariance matrix over the global profile space to anchor Mahalanobis geometry
cov_matrix = np.cov(df_ref.values, rowvar=True) + np.eye(len(labels)) * 1e-6
inv_cov = np.linalg.inv(cov_matrix)

# 2. Sliding Window Text Processing Engine
def run_text_sliding_window(text_tokens, window_size=100, stride=20):
    """Slices tokens into rolling windows and extracts structural vector configurations."""
    windows_metrics = []
    
    for start in range(0, len(text_tokens) - window_size + 1, stride):
        window_tokens = text_tokens[start:start + window_size]
        
        # Calculate isolated rolling features
        word_freq = len(window_tokens)
        word_lengths = [len(w) for w in window_tokens]
        words_mean = np.mean(word_lengths) if word_lengths else 0
        
        # Calculate Token Density (ratio of alphanumeric tokens to total unique structures)
        unique_tokens = len(set(window_tokens))
        token_density = unique_tokens / word_freq if word_freq > 0 else 0
        
        windows_metrics.append([word_freq, words_mean, token_density])
        
    return np.array(windows_metrics)

# Simulate a processed token sequence from an external source (e.g., T.S. Eliot Gutenberg text)
np.random.seed(101)
simulated_tokens = ["word" * int(np.random.randint(1, 8)) for _ in range(1200)]

# Extract structural features frame-by-frame across consecutive segments
window_vectors = run_text_sliding_window(simulated_tokens, window_size=150, stride=30)
n_windows = len(window_vectors)

# 3. Dynamic Tracking Loop across Sequential Blocks
rolling_distances_to_baseline = []
all_window_matrices = []

for k in range(n_windows):
    # Assemble a temporary local profile combining references with current text frame configurations
    current_frame = window_vectors[k]
    combined_matrix = np.vstack([df_ref.T.values, current_frame]) # Shape: (6, 3)
    
    # Calculate full Mahalanobis pairwise interactions using scipy.spatial.distance.pdist
    cond_dist = pdist(combined_matrix, metric='mahalanobis', VI=inv_cov)
    square_dist = squareform(cond_dist)
    
    all_window_matrices.append(square_dist)
    # Track distance between baseline profile (index 0) and the incoming frame text segment (index 5)
    rolling_distances_to_baseline.append(square_dist[0, 5])

# 4. Multi-Panel Matplotlib Layout
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Panel A: Timeline tracking structural divergence from Baseline
ax1.plot(range(n_windows), rolling_distances_to_baseline, color='#1f77b4', marker='o', linewidth=2)
ax1.set_title('Rolling Mahalanobis Distance to Poem Baseline across Text Timeline', fontsize=13, fontweight='bold', pad=10)
ax1.set_xlabel('Sequential Window Index Frame', fontsize=11)
ax1.set_ylabel('Structural Distance Metric', fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.6)

# Panel B: Summary heat summary tracking step shifts
# Map distance matrix averages to see points of total volatility stabilization
matrix_means = [np.mean(m) for m in all_window_matrices]
ax2.bar(range(n_windows), matrix_means, color='#2ca02c', alpha=0.85, edgecolor='black')
ax2.set_title('Global System Variance Level (Mean Matrix Divergence Over Time)', fontsize=13, fontweight='bold', pad=10)
ax2.set_xlabel('Sequential Window Index Frame', fontsize=11)
ax2.set_ylabel('Average Divergence Scale', fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
