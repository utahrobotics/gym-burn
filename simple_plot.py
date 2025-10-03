#!/usr/bin/env python3
"""
Simple dimensionality reduction and plotting script for ids.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Read the CSV file, skipping the header row
df = pd.read_csv('ids.csv', skiprows=1, header=None)

# Extract the 8-dimensional points (columns 2-9, since pandas is 0-indexed)
data_8d = df.iloc[:, 2:10].values
cluster_ids = df.iloc[:, 1].values

print(f"Loaded {len(data_8d)} points with {data_8d.shape[1]} dimensions")

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_8d)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
data_2d = pca.fit_transform(data_scaled)

print(f"PCA explained variance: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")

# Create the plot
plt.figure(figsize=(10, 8))

# Color by cluster ID if there are multiple clusters
unique_clusters = np.unique(cluster_ids)
if len(unique_clusters) > 1:
    scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], 
                         c=cluster_ids, 
                         cmap='tab20', 
                         alpha=0.7, 
                         s=20)
    plt.colorbar(scatter, label='Cluster ID')
else:
    plt.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.7, s=20)

plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title(f'2D PCA Visualization of 8D Data\n({len(data_2d):,} points)')
plt.grid(True, alpha=0.3)

# Save and show the plot
plt.savefig('pca_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualization complete! Plot saved as 'pca_visualization.png'")