import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Structure the isolated target environment dataset with the updated label
metrics_data = {
    'Poem_Baseline': [120, 450, 1.2, 0.85, 4.2],
    'Person_A':      [140, 420, 1.5, 0.70, 3.9],
    'Person_B':      [95,  490, 0.9, 0.92, 4.8],
    'Person_R':      [118, 455, 1.3, 0.83, 4.1],
    'Person_S':      [135, 410, 1.6, 0.65, 3.6]
}
# Changed 'Gematria_Mean' -> 'words_Mean'
all_labels = ['Word_Freq', 'words_Mean', 'Verse_Lag', 'Token_Density', 'Entropy']
df_full = pd.DataFrame(metrics_data, index=all_labels)

# Isolate specific target rows/metrics of interest
target_metrics = ['Word_Freq', 'words_Mean', 'Token_Density']
df = df_full.loc[target_metrics]

# Helper function to compute pure partial correlations via the Precision Matrix
def calc_partial_corr(data_matrix):
    corr = np.corrcoef(data_matrix)
    ridge = 1e-6
    precision = np.linalg.inv(corr + np.eye(corr.shape) * ridge)
    diag = np.sqrt(np.diag(precision))
    p_corr = -precision / np.outer(diag, diag)
    np.fill_diagonal(p_corr, 1.0)
    return p_corr

# Calculate the authentic observable partial correlations
obs_matrix = df.values  # Rows = isolated features, Columns = environments
obs_pcorr = calc_partial_corr(obs_matrix)

# 2. Build the Non-Parametric Resampling Distribution (Permutation Loop)
np.random.seed(42)
n_permutations = 1000
n_features, n_obs = obs_matrix.shape
perm_distribution = np.zeros((n_permutations, n_features, n_features))

for p in range(n_permutations):
    perm_matrix = obs_matrix.copy()
    for r in range(n_features):
        perm_matrix[r, :] = np.random.permutation(perm_matrix[r, :])
    perm_distribution[p] = calc_partial_corr(perm_matrix)

# 3. Calculate Empirical P-Values from the Resampling Layer
extreme_counts = np.sum(np.abs(perm_distribution) >= np.abs(obs_pcorr), axis=0)
empirical_p = extreme_counts / n_permutations
np.fill_diagonal(empirical_p, 0.0)

# 4. Generate Distance Grid Configurations
names = list(df.columns)
n_entities = len(names)
dist_matrix = np.zeros((n_entities, n_entities))
for i in range(n_entities):
    for j in range(n_entities):
        dist_matrix[i, j] = np.sqrt(np.sum((df[names[i]].values - df[names[j]].values) ** 2))

# 5. Render Plot Layout using Native Matplotlib
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
entities = df.columns

# ---- Left Panel: Permutation Masked Correlation ----
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

# ---- Right Panel: Distance Tracking Matrix ----
im2 = ax2.imshow(dist_matrix, cmap='viridis')
ax2.set_title('Numerical Environment Distance\n(Isolated Metrics Grid)', fontsize=13, pad=12, fontweight='bold')
ax2.set_xticks(np.arange(n_entities))
ax2.set_yticks(np.arange(n_entities))
ax2.set_xticklabels(entities, rotation=30, ha='right')
ax2.set_yticklabels(entities)

for i in range(n_entities):
    for j in range(n_entities):
        txt_color = 'white' if dist_matrix[i, j] > np.max(dist_matrix) * 0.5 else 'black'
        ax2.text(j, i, f"{dist_matrix[i, j]:.1f}", ha='center', va='center', color=txt_color)

ax2.set_xticks(np.arange(n_entities) + 0.5, minor=True)
ax2.set_yticks(np.arange(n_entities) + 0.5, minor=True)
ax2.grid(which='minor', color='white', linestyle='-', linewidth=1.5)
fig.colorbar(im2, ax=ax2, shrink=0.75)

plt.tight_layout()
plt.show()
