import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# 1. Base Environment and Sliding Window Setup
metrics_data = {
    'Poem_Baseline': [120, 450, 1.2, 0.85, 4.2],
    'Person_A':      [140, 420, 1.5, 0.70, 3.9],
    'Person_B':      [95,  490, 0.9, 0.92, 4.8],
    'Person_R':      [118, 455, 1.3, 0.83, 4.1],
    'Person_S':      [135, 410, 1.6, 0.65, 3.6]
}
df_full = pd.DataFrame(metrics_data, index=['Word_Freq', 'words_Mean', 'Verse_Lag', 'Token_Density', 'Entropy'])
df = df_full.loc[['Word_Freq', 'words_Mean', 'Token_Density']]

def calc_partial_corr(data_matrix):
    corr = np.corrcoef(data_matrix)
    ridge = 1e-6
    precision = np.linalg.inv(corr + np.eye(corr.shape) * ridge)
    diag = np.sqrt(np.diag(precision))
    p_corr = -precision / np.outer(diag, diag)
    np.fill_diagonal(p_corr, 1.0)
    return p_corr

obs_pcorr = calc_partial_corr(df.values)

# 2. Simulate True vs. Decoded Word-to-Word Distance Categories
# To construct a valid confusion matrix, we map structural word distances into discrete classes:
# Class 0: Close Proximity, Class 1: Medium Proximity, Class 2: Distant Proximity
np.random.seed(42)
n_window_words = 150

# Generate simulated structural ground-truth and matched predictive model decoded states
true_distance_states = np.random.choice([0, 1, 2], size=n_window_words, p=[0.4, 0.4, 0.2])
decoded_distance_states = np.array([
    t if np.random.rand() > 0.25 else np.random.choice([0, 1, 2]) 
    for t in true_distance_states
])

# Compute raw Confusion Matrix components manually without external ML library requirements
n_classes = 3
confusion_mtx = np.zeros((n_classes, n_classes), dtype=int)
for t, p in zip(true_distance_states, decoded_distance_states):
    confusion_mtx[t, p] += 1

# 3. Permutation Loop for Partial Correlation Significance Check
n_permutations = 500
n_features, n_obs = df.shape
perm_distribution = np.zeros((n_permutations, n_features, n_features))

for p in range(n_permutations):
    perm_matrix = df.values.copy()
    for r in range(n_features):
        perm_matrix[r, :] = np.random.permutation(perm_matrix[r, :])
    perm_distribution[p] = calc_partial_corr(perm_matrix)

extreme_counts = np.sum(np.abs(perm_distribution) >= np.abs(obs_pcorr), axis=0)
empirical_p = extreme_counts / n_permutations
np.fill_diagonal(empirical_p, 0.0)

# 4. Render Layout Output using Matplotlib
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
entities = df.columns
n_entities = len(entities)

# ---- Left Panel: Permutation Masked Correlation Matrix ----
im1 = ax1.imshow(obs_pcorr, cmap='RdBu_r', vmin=-1, vmax=1)
ax1.set_title('Non-Parametric Permutation Mask\n(Alpha = 0.05 Null Check)', fontsize=13, pad=12, fontweight='bold')
ax1.set_xticks(np.arange(n_entities))
ax1.set_yticks(np.arange(n_entities))
ax1.set_xticklabels(entities, rotation=30, ha='right')
ax1.set_yticklabels(entities)

for i in range(n_entities):
    for j in range(n_entities):
        if empirical_p[i, j] >= 0.05:
            ax1.text(j, i, "X\n[ns]", ha='center', va='center', color='darkred', fontsize=10, fontweight='bold')
            ax1.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color='black', alpha=0.18))
        else:
            txt_color = 'white' if abs(obs_pcorr[i, j]) > 0.5 else 'black'
            ax1.text(j, i, f"{obs_pcorr[i, j]:.2f}", ha='center', va='center', color=txt_color, fontweight='bold')

ax1.set_xticks(np.arange(n_entities) + 0.5, minor=True)
ax1.set_yticks(np.arange(n_entities) + 0.5, minor=True)
ax1.grid(which='minor', color='white', linestyle='-', linewidth=1.5)
fig.colorbar(im1, ax=ax1, shrink=0.75)

# ---- Right Panel: Word-to-Word Distance Confusion Matrix ----
im2 = ax2.imshow(confusion_mtx, cmap='Blues')
ax2.set_title('Word-to-Word Distance Confusion Matrix\n(True vs. Decoded Proximity States)', fontsize=13, pad=12, fontweight='bold')

class_labels = ['Close', 'Medium', 'Distant']
ax2.set_xticks(np.arange(n_classes))
ax2.set_yticks(np.arange(n_classes))
ax2.set_xticklabels(class_labels, fontsize=11)
ax2.set_yticklabels(class_labels, fontsize=11)
ax2.set_xlabel('Predicted/Decoded State', fontsize=12, labelpad=8)
ax2.set_ylabel('True Baseline State', fontsize=12, labelpad=8)

# Annotate raw integer counts and percentage profiles inside matrix quadrants
for i in range(n_classes):
    for j in range(n_classes):
        cell_count = confusion_mtx[i, j]
        row_sum = np.sum(confusion_mtx[i, :])
        percentage = (cell_count / row_sum * 100) if row_sum > 0 else 0
        
        txt_color = 'white' if cell_count > np.max(confusion_mtx) * 0.5 else 'black'
        ax2.text(j, i, f"{cell_count}\n({percentage:.1f}%)", ha='center', va='center', color=txt_color, fontweight='bold')

ax2.set_xticks(np.arange(n_classes) + 0.5, minor=True)
ax2.set_yticks(np.arange(n_classes) + 0.5, minor=True)
ax2.grid(which='minor', color='white', linestyle='-', linewidth=1.5)
fig.colorbar(im2, ax=ax2, shrink=0.75)

plt.tight_layout()
plt.show()
