import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Structure the isolated target environment dataset
metrics_data = {
    'Poem_Baseline': [120, 450, 0.85],
    'Person_A':      [140, 420, 0.70],
    'Person_B':      [95,  490, 0.92],
    'Person_R':      [118, 455, 0.83],
    'Person_S':      [135, 410, 0.65]
}
target_metrics = ['Word_Freq', 'words_Mean', 'Token_Density']
df = pd.DataFrame(metrics_data, index=target_metrics)

def calc_partial_corr(data_matrix):
    corr = np.corrcoef(data_matrix)
    ridge = 1e-6
    precision = np.linalg.inv(corr + np.eye(corr.shape) * ridge)
    diag = np.sqrt(np.diag(precision))
    p_corr = -precision / np.outer(diag, diag)
    np.fill_diagonal(p_corr, 1.0)
    return p_corr

obs_pcorr = calc_partial_corr(df.values)

# 2. Simulate Core Word Distributions with Anomaly Constraints
np.random.seed(42)
n_window_words = 200

# Base Classes: 0=Close, 1=Medium, 2=Distant, 3=Anomaly (Fails to match expected boundaries)
true_states = np.random.choice([0, 1, 2, 3], size=n_window_words, p=[0.4, 0.35, 0.18, 0.07])

# Generate a decoded stream introducing measurement noise and anomaly boundary filtering
decoded_states = []
for t in true_states:
    dice_roll = np.random.rand()
    if t == 3:  # Ground-truth structural anomaly
        decoded_states.append(3 if dice_roll > 0.2 else np.random.choice([0, 1, 2]))
    else:
        if dice_roll > 0.3:  # Correctly aligned match
            decoded_states.append(t)
        elif dice_roll > 0.1:  # Categorization noise
            decoded_states.append(np.random.choice([0, 1, 2]))
        else:  # Distance exceeds maximum allowable limits -> Flag as anomaly
            decoded_states.append(3)

decoded_states = np.array(decoded_states)

# 3. Calculate 4x4 Confusion Matrix
n_classes = 4
confusion_mtx = np.zeros((n_classes, n_classes), dtype=int)
for t, p in zip(true_states, decoded_states):
    confusion_mtx[t, p] += 1

# 4. Calculate Precision, Recall, and F1-Score for Core Classes (Excluding global anomalies if preferred)
precision_list, recall_list, f1_list = [], [], []
for c in range(n_classes):
    tp = confusion_mtx[c, c]
    fp = np.sum(confusion_mtx[:, c]) - tp
    fn = np.sum(confusion_mtx[c, :]) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

# Summary aggregates
macro_f1 = np.mean(f1_list)
macro_precision = np.mean(precision_list)

# 5. Run Permutation Loop for Signficance Check
n_permutations = 200
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

# 6. Plotting via Matplotlib API
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
entities = df.columns

# ---- Left Axes: Partial Correlation ----
im1 = ax1.imshow(obs_pcorr, cmap='RdBu_r', vmin=-1, vmax=1)
ax1.set_title('Non-Parametric Permutation Mask\n(Alpha = 0.05 Null Check)', fontsize=12, pad=12, fontweight='bold')
ax1.set_xticks(np.arange(len(entities)))
ax1.set_yticks(np.arange(len(entities)))
ax1.set_xticklabels(entities, rotation=30, ha='right')
ax1.set_yticklabels(entities)

for i in range(len(entities)):
    for j in range(len(entities)):
        if empirical_p[i, j] >= 0.05:
            ax1.text(j, i, "X\n[ns]", ha='center', va='center', color='darkred', fontsize=10, fontweight='bold')
            ax1.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color='black', alpha=0.18))
        else:
            txt_color = 'white' if abs(obs_pcorr[i, j]) > 0.5 else 'black'
            ax1.text(j, i, f"{obs_pcorr[i, j]:.2f}", ha='center', va='center', color=txt_color, fontweight='bold')
fig.colorbar(im1, ax=ax1, shrink=0.7)

# ---- Right Axes: Confusion Matrix with Anomaly Class ----
im2 = ax2.imshow(confusion_mtx, cmap='Blues')
title_str = f'Word Distance Confusion Matrix\nMacro Precision: {macro_precision:.2f} | Macro F1: {macro_f1:.2f}'
ax2.set_title(title_str, fontsize=12, pad=12, fontweight='bold')

class_labels = ['Close', 'Medium', 'Distant', 'Anomaly']
ax2.set_xticks(np.arange(n_classes))
ax2.set_yticks(np.arange(n_classes))
ax2.set_xticklabels(class_labels, fontsize=10)
ax2.set_yticklabels(class_labels, fontsize=10)
ax2.set_xlabel('Predicted/Decoded State', fontsize=11, labelpad=5)
ax2.set_ylabel('True Baseline State', fontsize=11, labelpad=5)

for i in range(n_classes):
    for j in range(n_classes):
        cell_count = confusion_mtx[i, j]
        row_sum = np.sum(confusion_mtx[i, :])
        percentage = (cell_count / row_sum * 100) if row_sum > 0 else 0
        txt_color = 'white' if cell_count > np.max(confusion_mtx) * 0.4 else 'black'
        ax2.text(j, i, f"{cell_count}\n({percentage:.1f}%)", ha='center', va='center', color=txt_color, fontweight='bold')

ax2.set_xticks(np.arange(n_classes) + 0.5, minor=True)
ax2.set_yticks(np.arange(n_classes) + 0.5, minor=True)
ax2.grid(which='minor', color='white', linestyle='-', linewidth=1.5)
fig.colorbar(im2, ax=ax2, shrink=0.7)

plt.tight_layout()
plt.show()
