import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Generate a continuous distribution of anomaly scores on text
np.random.seed(42)
n_text_segments = 300

# Most segments are regular text (low distance), with a small fraction of anomalies (high distance)
is_anomaly_ground_truth = np.random.choice([0, 1], size=n_text_segments, p=[0.92, 0.08])
distance_scores = np.zeros(n_text_segments)

for i in range(n_text_segments):
    if is_anomaly_ground_truth[i] == 1:
        distance_scores[i] = np.random.normal(loc=3.8, scale=0.8)  # Outliers
    else:
        distance_scores[i] = np.random.normal(loc=1.5, scale=0.5)  # Regular text

# 2. Convert raw distance scores to empirical p-values against a regular text baseline
# (Assuming a known normal text baseline mean=1.5, std=0.5 under the null hypothesis)
from scipy.stats import norm
p_values = 1 - norm.cdf(distance_scores, loc=1.5, scale=0.5)

# 3. Apply Benjamini-Hochberg (BH) False Discovery Rate Procedure
def apply_fdr_control(p_vals, target_fdr=0.05):
    """Computes the critical p-value threshold to control FDR at a target rate."""
    n = len(p_vals)
    sorted_indices = np.argsort(p_vals)
    sorted_p_vals = p_vals[sorted_indices]
    
    # Calculate Benjamini-Hochberg critical boundary lines: (i / n) * q
    bh_critical_values = [(i + 1) / n * target_fdr for i in range(n)]
    
    # Find the largest index i where p_i <= critical_value_i
    significant_indices = np.where(sorted_p_vals <= bh_critical_values)[0]
    
    if len(significant_indices) > 0:
        max_sig_idx = np.max(significant_indices)
        p_threshold = sorted_p_vals[max_sig_idx]
        cutoff_index_sorted = max_sig_idx
    else:
        p_threshold = 0.0
        cutoff_index_sorted = -1
        
    return p_threshold, sorted_indices, sorted_p_vals, bh_critical_values, cutoff_index_sorted

target_q = 0.05
p_thresh, sorted_idx, sorted_p, bh_crit, cutoff_idx = apply_fdr_control(p_values, target_fdr=target_q)

# 4. Separate discovered anomalies based on the adjusted FDR line
final_flags = np.zeros(n_text_segments, dtype=int)
if cutoff_idx >= 0:
    # Flag all elements up to the last significant index in sorted order
    flagged_original_indices = sorted_idx[:cutoff_idx + 1]
    final_flags[flagged_original_indices] = 1

# Calculate the actual empirical False Discovery Rate
total_flags = np.sum(final_flags)
false_flags = np.sum((final_flags == 1) & (is_anomaly_ground_truth == 0))
realized_fdr = false_flags / total_flags if total_flags > 0 else 0.0

# 5. Plot the Benjamini-Hochberg FDR Control Grid
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(range(1, n_text_segments + 1), sorted_p, label='Sorted Empirical p-values', color='blue', lw=2)
ax.plot(range(1, n_text_segments + 1), bh_crit, label=f'BH Critical Line (q = {target_q})', color='red', linestyle='--', lw=1.5)

if cutoff_idx >= 0:
    ax.axvline(x=cutoff_idx + 1, color='green', linestyle=':', label='FDR Cutoff Boundary', lw=2)
    ax.scatter(cutoff_idx + 1, sorted_p[cutoff_idx], color='green', s=100, zorder=5, label=f'Threshold p = {p_thresh:.4f}')

ax.set_title(f'FDR Configuration Grid to Minimize False Flags\nRealized False Discovery Rate: {realized_fdr*100:.1f}%', fontsize=12, pad=12, fontweight='bold')
ax.set_xlabel('Sorted Text Segment Rank Index', fontsize=11)
ax.set_ylabel('Probability Value (p-value)', fontsize=11)
ax.set_xlim([0, min(cutoff_idx * 4, n_text_segments)])  # Zoom into the critical selection zone
ax.set_ylim([0, target_q * 1.5])
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.show()
