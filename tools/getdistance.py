import numpy as np
from scipy.spatial.distance import pdist, squareform

# Let's write a clean, complete script that fulfills all the requirements:
# 1. Incorporates multiple entities ("Person A", "Person B", and adding "Person R", "Person S")
# 2. Computes the partial correlation matrix to eliminate confounding variables
# 3. Defines a distance function evaluating structural numerical differences in the text's English numerical environment
# 4. Correctly conditions the linear dependency to prevent singular matrix/NaN errors.

import numpy as np
import pandas as pd

# 5 participants/environments evaluating a set of 5 textual/gematria metrics
# Columns: Entities/Environments. Rows: Numerical Features of the English text environment.
metrics_data = {
    'Poem_Baseline': [120, 45, 14, 88, 32],
    'Person_A':      [112, 40, 15, 82, 30],
    'Person_B':      [135, 52, 11, 95, 38],
    'Person_R':      [118, 44, 13, 86, 31], # Added Person R
    'Person_S':      [122, 47, 16, 89, 34]  # Added Person S
}

df = pd.DataFrame(metrics_data, index=['Word_Freq', 'Gematria_Mean', 'Verse_Lag', 'Token_Density', 'Entropy'])

# Standard correlation
corr = df.corr().values

# Calculate Partial Correlation via the Precision Matrix (Inverse of Covariance/Correlation)
# To handle potential multicollinearity safely, we add a tiny ridge adjustment
ridge = 1e-6
precision_mat = np.linalg.inv(corr + np.eye(corr.shape[0]) * ridge)
diag = np.sqrt(np.diag(precision_mat))
partial_corr_mat = -precision_mat / np.outer(diag, diag)
np.fill_diagonal(partial_corr_mat, 1.0)

partial_corr_df = pd.DataFrame(partial_corr_mat, index=df.columns, columns=df.columns)

# Define a custom distance function for the difference in English numerical environments
# We will use a Mahalanobis-like absolute variance metric or standard Euclidean distance 
def numerical_env_distance(vec1, vec2):
    return np.sqrt(np.sum((np.array(vec1) - np.array(vec2)) ** 2))

# Generate distance matrix
names = list(df.columns)
dist_matrix = np.zeros((len(names), len(names)))
#for i in range(len(names)):
#    for j in range(len(names)):
#        dist_matrix[i, j] = numerical_env_distance(df[names[i]], df[names[j]])
data_to_compare = df[names].T.value
# 2. Calculate pairwise distances (returns a condensed 1D array)
# Change 'euclidean' to 'cityblock', 'cosine', etc., if needed
condensed_dist = pdist(data_to_compare, metric='euclidean')


# 3. Convert the 1D array back into a square 2D matrix
dist_matrix = squareform(condensed_dist)


dist_df = pd.DataFrame(dist_matrix, index=names, columns=names)

print("PARTIAL_CORRELATION_MATRIX:")
print(partial_corr_df.round(3))
print("\nDISTANCE_MATRIX:")
print(dist_df.round(3))
