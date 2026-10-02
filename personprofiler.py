import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Structure the English numerical environment data (Rows = Metrics, Columns = Entities)
metrics_data = {
    'Poem_Baseline': [120, 45, 14, 88, 32],
    'Person_A':      [112, 40, 15, 82, 30],
    'Person_B':      [135, 52, 11, 95, 38],
    'Person_R':      [118, 44, 13, 86, 31],  # Integrated Person R
    'Person_S':      [122, 47, 16, 89, 34]   # Added Person S for multidimensional variance
}

df = pd.DataFrame(metrics_data, index=['Word_Freq', 'Gematria_Mean', 'Verse_Lag', 'Token_Density', 'Entropy'])

# 2. Compute the Partial Correlation Matrix using the Precision Matrix method
corr_matrix = df.corr().values
ridge = 1e-6  # Safeguard configuration to ensure matrix inversion stability
precision_matrix = np.linalg.inv(corr_matrix + np.eye(corr_matrix.shape[0]) * ridge)
diag_sqrt = np.sqrt(np.diag(precision_matrix))

partial_corr = -precision_matrix / np.outer(diag_sqrt, diag_sqrt)
np.fill_diagonal(partial_corr, 1.0)
partial_corr_df = pd.DataFrame(partial_corr, index=df.columns, columns=df.columns)

# 3. Define the distance function for the difference in English numerical environments
def numerical_env_distance(vector_1, vector_2):
    """Calculates the Euclidean geometric distance between two text metric arrays."""
    return np.sqrt(np.sum((np.array(vector_1) - np.array(vector_2)) ** 2))

# 4. Generate the matrix tracking spatial distance metrics
names = list(df.columns)
distance_array = np.zeros((len(names), len(names)))

for i in range(len(names)):
    for j in range(len(names)):
        distance_array[i, j] = numerical_env_distance(df[names[i]], df[names[j]])

dist_df = pd.DataFrame(distance_array, index=names, columns=names)

# 5. Graphing the dual interactive heatmaps using matplotlib/seaborn
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot Partial Correlation Heatmap
sns.heatmap(partial_corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[0])
axes[0].set_title('Partial Correlation Matrix')

# Plot Environment Distance Heatmap
sns.heatmap(dist_df, annot=True, cmap='YlOrRd', ax=axes[1])
axes[1].set_title('Numerical Environment Distance Matrix')

plt.tight_layout()
plt.show()
