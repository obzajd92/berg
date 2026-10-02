import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram

# 1. Structure the 5x5 All-Pairs Normalized Pacing Distance Matrix
chapter_names = [
    'I. Burial of the Dead', 'II. A Game of Chess', 
    'III. The Fire Sermon', 'IV. Death by Water', 'V. What the Thunder Said'
]
n_chapters = len(chapter_names)

# Ingestion profile loading the pre-calculated normalized warping distance coordinates
# (Symmetric distance matrix retrieved from the compiled Scipy DTW script pipeline)
pacing_matrix = np.array([
    [0.000, 0.231, 0.312, 0.489, 0.145],
    [0.231, 0.000, 0.187, 0.521, 0.298],
    [0.312, 0.187, 0.000, 0.415, 0.364],
    [0.489, 0.521, 0.415, 0.000, 0.512],
    [0.145, 0.298, 0.364, 0.512, 0.000]
])

# Extract condensed form for valid linkage tracking
condensed_dist = squareform(pacing_matrix, checks=False)

# 2. Run Hierarchical Ward Linkage Agglomerative Clustering
# Ward's minimum variance topology minimizes total within-cluster variance vector sums
Z = linkage(condensed_dist, method='ward')

# 3. Graphing the Dual Interactive Dashboards via Pure Matplotlib
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Heatmap Plot Setup
im = ax1.imshow(pacing_matrix, cmap='magma', origin='upper')
ax1.set_title('All-Pairs Cross-Chapter Stylistic Pacing Matrix', fontsize=12, fontweight='bold', pad=12)
ax1.set_xticks(np.arange(n_chapters))
ax1.set_yticks(np.arange(n_chapters))
ax1.set_xticklabels(chapter_names, rotation=30, ha='right')
ax1.set_yticklabels(chapter_names)

for i in range(n_chapters):
    for j in range(n_chapters):
        txt_color = 'black' if pacing_matrix[i, j] > np.max(pacing_matrix) * 0.6 else 'white'
        ax1.text(j, i, f"{pacing_matrix[i, j]:.3f}", ha='center', va='center', color=txt_color, fontweight='bold')

# Dendrogram Plot Setup
dendrogram(Z, labels=chapter_names, ax=ax2, orientation='top', leaf_rotation=30, leaf_font_size=10)
ax2.set_title('Hierarchical Clustering Dendrogram\n(Ward Linkage over DTW Pacing Space)', fontsize=12, fontweight='bold', pad=12)
ax2.set_ylabel('Linkage Distance Threshold', fontsize=11)
ax2.grid(True, axis='y', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()
