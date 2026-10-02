"""
Dynamic Time Warping (DTW) Text Alignment Module
================================================
Aligns rolling architectural feature sequences from separate poetic movements 
(The Fire Sermon vs. The Burial of the Dead) to map out structural stylistic correspondences.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple

# 1. Structural Feature Extraction Engine for Target Texts
def mock_extract_window_features(sequence_length: int, seed: int) -> np.ndarray:
    """Generates synthetic [Word_Freq, words_Mean, Token_Density] rolling frames for testing."""
    np.random.seed(seed)
    # Simulate slightly different word-length averages and density transitions
    freq = np.ones(sequence_length) * 20.0
    words_mean = np.random.normal(loc=4.4, scale=0.3, size=sequence_length)
    token_density = np.random.normal(loc=0.82, scale=0.05, size=sequence_length)
    return np.column_stack([freq, words_mean, token_density])

# Simulate uneven timeline frames (Burial of the Dead = 45 blocks, Fire Sermon = 35 blocks)
features_burial = mock_extract_window_features(sequence_length=45, seed=77)
features_sermon = mock_extract_window_features(sequence_length=35, seed=88)

# 2. Native Dynamic Time Warping (DTW) Configuration Loop
def compute_dtw_alignment(seq_x: np.ndarray, seq_y: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Computes the accumulated cost matrix and extracts the optimal warping path.
    
    Args:
        seq_x: First sequence array, shape (N, M).
        seq_y: Second sequence array, shape (K, M).
        
    Returns:
        A tuple containing:
            - The complete (N x K) accumulated cost grid matrix.
            - A list of coordinate tuples tracing the optimal back-tracked path.
    """
    n, k = len(seq_x), len(seq_y)
    
    # Initialize cost matrices with infinity bounds
    cost_matrix = np.zeros((n, k))
    for i in range(n):
        for j in range(k):
            # Evaluate Local Euclidean Distance in feature metric vector space
            cost_matrix[i, j] = np.sqrt(np.sum((seq_x[i] - seq_y[j]) ** 2))
            
    accumulated_cost = np.zeros((n, k))
    accumulated_cost[0, 0] = cost_matrix[0, 0]
    
    # Initialize boundaries
    for i in range(1, n):
        accumulated_cost[i, 0] = accumulated_cost[i-1, 0] + cost_matrix[i, 0]
    for j in range(1, k):
        accumulated_cost[0, j] = accumulated_cost[0, j-1] + cost_matrix[0, j]
        
    # Main Dynamic Programming Optimization Loop
    for i in range(1, n):
        for j in range(1, k):
            accumulated_cost[i, j] = cost_matrix[i, j] + min(
                accumulated_cost[i-1, j],    # Insertion / Compression
                accumulated_cost[i, j-1],    # Deletion / Expansion
                accumulated_cost[i-1, j-1]   # Match / Step
            )
            
    # Backtrack from (N-1, K-1) down to (0, 0) to find the minimum distance path
    path = []
    i, j = n - 1, k - 1
    path.append((i, j))
    
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            steps = [accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]]
            min_step = np.argmin(steps)
            if min_step == 0:
                i, j = i - 1, j - 1
            elif min_step == 1:
                i -= 1
            else:
                j -= 1
        path.append((i, j))
        
    return accumulated_cost, path[::-1]

# Run structural sequence mapping matrix calculation
dtw_grid, warping_path = compute_dtw_alignment(features_burial, features_sermon)

# 3. Multi-Panel Matplotlib Alignment Dashboard Generation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# ---- Left Panel: DTW Cost Matrix & Warping Path Alignment ----
im = ax1.imshow(dtw_grid, origin='lower', cmap='magma', aspect='auto')
path_x, path_y = zip(*warping_path)
ax1.plot(path_y, path_x, color='white', linewidth=3, label='Optimal Warping Path')
ax1.set_title('Accumulated DTW Structural Cost Matrix\n(The Burial of the Dead vs. The Fire Sermon)', fontsize=12, pad=12, fontweight='bold')
ax1.set_xlabel('The Fire Sermon Frame Index', fontsize=11)
ax1.set_ylabel('The Burial of the Dead Frame Index', fontsize=11)
ax1.legend(loc='upper left')
fig.colorbar(im, ax=ax1, label='Cumulative Structural Disparity Scale')

# ---- Right Panel: Alignment Trajectory Comparison ----
# Compare feature vector profiles (e.g., words_Mean metric index column 1) mapped along the warping index path
aligned_burial = [features_burial[idx[0], 1] for idx in warping_path]
aligned_sermon = [features_sermon[idx[1], 1] for idx in warping_path]

ax2.plot(aligned_burial, color='#1f77b4', lw=2, label='The Burial of the Dead (Aligned)')
ax2.plot(aligned_sermon, color='#ff7f0e', lw=2, linestyle='--', label='The Fire Sermon (Aligned)')
ax2.set_title('Aligned Structural Stylistic Trajectories\n(Synced Mean Word Length Profile)', fontsize=12, pad=12, fontweight='bold')
ax2.set_xlabel('Synced Warping Path Step Index', fontsize=11)
ax2.set_ylabel('Mean Character Count Metric', fontsize=11)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()
