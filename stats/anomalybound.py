import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# 1. Generate synthetic continuous Mahalanobis distance scores for evaluation
np.random.seed(42)
n_samples = 500

# Let's say ~10% of our analyzed words are structural anomalies
true_binary_anomalies = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])

# Anomalies generally yield higher distance scores than standard proximity states
anomaly_scores = np.zeros(n_samples)
for i in range(n_samples):
    if true_binary_anomalies[i] == 1:
        anomaly_scores[i] = np.random.normal(loc=3.5, scale=1.0) # High distance profile
    else:
        anomaly_scores[i] = np.random.normal(loc=1.2, scale=0.6) # Low normal proximity profile

# 2. Compute ROC and Precision-Recall Metrics
fpr, tpr, roc_thresholds = roc_curve(true_binary_anomalies, anomaly_scores)
roc_auc = auc(fpr, tpr)

precision, recall, pr_thresholds = precision_recall_curve(true_binary_anomalies, anomaly_scores)
avg_precision = average_precision_score(true_binary_anomalies, anomaly_scores)

# 3. Plotting the Curves side-by-side using strict Matplotlib
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ---- Left Panel: ROC Curve ----
ax1.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.2f})')
ax1.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--') # Chance baseline
ax1.set_xlim([-0.02, 1.02])
ax1.set_ylim([-0.02, 1.02])
ax1.set_xlabel('False Positive Rate (FPR)', fontsize=11, labelpad=6)
ax1.set_ylabel('True Positive Rate (TPR)', fontsize=11, labelpad=6)
ax1.set_title('Receiver Operating Characteristic (ROC)\nEvaluating Anomaly Boundary', fontsize=12, pad=12, fontweight='bold')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc="lower right", fontsize=10)

# ---- Right Panel: Precision-Recall Curve ----
ax2.plot(recall, precision, color='teal', lw=2.5, label=f'PR Curve (AP = {avg_precision:.2f})')
# Horizontal baseline representing the ratio of positive anomalies in the dataset
random_precision_baseline = np.sum(true_binary_anomalies) / n_samples
ax2.plot([0, 1], [random_precision_baseline, random_precision_baseline], color='red', lw=1.5, linestyle='--')
ax2.set_xlim([-0.02, 1.02])
ax2.set_ylim([-0.02, 1.02])
ax2.set_xlabel('Recall (Sensitivity)', fontsize=11, labelpad=6)
ax2.set_ylabel('Precision (Positive Predictive Value)', fontsize=11, labelpad=6)
ax2.set_title('Precision-Recall (PR) Curve\nEvaluating Imbalanced Text Outliers', fontsize=12, pad=12, fontweight='bold')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc="lower left", fontsize=10)

plt.tight_layout()
plt.show()
