"""
The Fire Sermon Rolling Frame-by-Frame Analyzer Engine
======================================================
Splits the isolated text of 'The Fire Sermon' into consecutive windows to track
internal stylistic shifts, lexical density changes, and multi-text covariance pivots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import re
from typing import List

# 1. Isolate and Define Raw Text Content for The Fire Sermon
the_fire_sermon_text = """
The river's tent is broken: the last fingers of leaf
Clutch and sink into the wet bank. The wind
Crosses the brown land, unheard. The nymphs are departed.
Sweet Thames, run softly, till I end my song.
The river bears no empty bottles, sandwich papers,
Silk handkerchiefs, cardboard boxes, cigarette ends
Or other testimony of summer nights. The nymphs are departed.
And their friends, the loitering heirs of city directors;
Departed, have left no addresses.
By the waters of Leman I sat down and wept...
Sweet Thames, run softly, till I end my song.
Sweet Thames, run softly, for I speak not loud or long.
But at my back in a cold blast I hear
The rattle of the bones, and chuckle spread from ear to ear.
"""

# Extract clean tokens word-by-word
fire_sermon_tokens = re.findall(r'\b\w+\b', the_fire_sermon_text.lower())

# 2. Structured Sliding Window Engine
def analyze_text_sliding_window(tokens: List[str], window_size: int = 30, stride: int = 5) -> pd.DataFrame:
    """Slides across word tokens to generate frame-by-frame structural metrics.
    
    Args:
        tokens: Flattened list of lowercase word strings.
        window_size: Number of words included in each metric window frame.
        stride: The step size/offset when shifting forward to the next frame.
        
    Returns:
        DataFrame containing sliding metric calculations for each sequence block.
    """
    frame_records = []
    n_tokens = len(tokens)
    
    # Sweep across token sequence indices
    for index, start in enumerate(range(0, n_tokens - window_size + 1, stride)):
        window_slice = tokens[start:start + window_size]
        
        # Calculate coordinate features for current text window
        word_freq = len(window_slice)
        words_mean = np.mean([len(w) for w in window_slice]) if window_slice else 0.0
        token_density = len(set(window_slice)) / word_freq if word_freq > 0 else 0.0
        
        frame_records.append({
            'Frame_Index': index,
            'Start_Token': start,
            'Word_Freq': float(word_freq),
            'words_Mean': float(words_mean),
            'Token_Density': float(token_density)
        })
        
    return pd.DataFrame(frame_records).set_index('Frame_Index')

# Generate frame records (using smaller window bounds due to short test sample text)
window_size_config = 20
stride_config = 2
df_frames = analyze_text_sliding_window(fire_sermon_tokens, window_size=window_size_config, stride=stride_config)

# 3. Compute Rolling Mahalanobis Spacing against Global Baseline
baseline_vector = np.array([20.0, 4.50, 0.85]) # Global baseline reference vector coordinates
all_features = df_frames[['Word_Freq', 'words_Mean', 'Token_Density']].values

# Calculate a stable inverse covariance matrix using the sliding frames space
cov_matrix = np.cov(all_features, rowvar=False) + np.eye(3) * 1e-6
inv_cov = np.linalg.inv(cov_matrix)

# Calculate Mahalanobis distance from baseline to each text window explicitly
rolling_distances = []
for frame in all_features:
    delta = frame - baseline_vector
    mahalanobis_dist = np.sqrt(delta.dot(inv_cov).dot(delta))
    rolling_distances.append(mahalanobis_dist)

df_frames['Mahalanobis_Distance'] = rolling_distances

# 4. Multi-Panel Matplotlib Dashboard Visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
frame_axis = df_frames.index

# Top Panel: Mahalanobis Distance Trajectory (Tracks sudden structural style switches)
ax1.plot(frame_axis, df_frames['Mahalanobis_Distance'], color='#d62728', marker='o', lw=2, label='Divergence Scale')
ax1.set_title(f"Sliding-Window Stylistic Analysis of 'The Fire Sermon' (Window={window_size_config}, Stride={stride_config})", fontsize=13, fontweight='bold', pad=10)
ax1.set_ylabel('Mahalanobis Structural Distance', fontsize=11)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='upper right')

# Bottom Panel: Feature Deconstruction
ax2.plot(frame_axis, df_frames['words_Mean'], color='#1f77b4', marker='s', lw=1.5, label='Mean Word Length')
ax2.set_ylabel('Mean Character Count', color='#1f77b4', fontsize=11)
ax2.tick_params(axis='y', labelcolor='#1f77b4')

# Instantiate a secondary mirror Y-axis for density scaling overrides
ax2_twin = ax2.twinx()
ax2_twin.plot(frame_axis, df_frames['Token_Density'], color='#2ca02c', marker='^', lw=1.5, label='Lexical Density')
ax2_twin.set_ylabel('Vocabulary Diversity Ratio', color='#2ca02c', fontsize=11)
ax2_twin.tick_params(axis='y', labelcolor='#2ca02c')

ax2.set_xlabel('Sequential Window Frame Index', fontsize=11)
ax2.grid(True, linestyle=':', alpha=0.6)

# Combine multi-axis legend labels neatly
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.show()
