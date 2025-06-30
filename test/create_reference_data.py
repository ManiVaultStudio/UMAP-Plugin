from umap.distances import correlation as umap_correlation_dist
import numpy as np

x1 = np.array([1, 2, 3, 4, 5])
y1 = np.array([1, 2, 3, 4, 5])  # perfect positive correlation

x2 = np.array([1, 2, 3, 4, 5])
y2 = np.array([5, 4, 3, 2, 1])  # perfect negative correlation

x3 = np.array([1, -1, 1, -1])
y3 = np.array([1, 1, -1, -1])   # zero correlation

x4 = np.array([5])
y4 = np.array([3])

x5 = np.array([1, 2])
y5 = np.array([3, 4])

x6 = np.array([0, 1, 0, 1])
y6 = np.array([1, 0, 1, 0])

x7 = np.array([1, 1.5, 3, 7])
y7 = np.array([1, 2.5, 2.9, 6])

x8 = np.array([6, 6.5, 5.9, 3])
y8 = np.array([2, 4.5, 5.2, 7])

# Compute results
print("Perfect positive correlation:", umap_correlation_dist(x1, y1))   # Expect ~0.0
print("Perfect negative correlation:", umap_correlation_dist(x2, y2))   # Expect ~2.0
print("No correlation:", umap_correlation_dist(x3, y3))                 # Expect ~1.0

print("Edge case 1:", umap_correlation_dist(x4, y4))                    # Expect ~0.0
print("Edge case 2:", umap_correlation_dist(x5, y5))                    # Expect ~0.0
print("Edge case 3:", umap_correlation_dist(x6, y6))                    # Expect ~2.0

print("Positive correlation:", umap_correlation_dist(x7, y7))           # Expect ~0.0249
print("Negative correlation:", umap_correlation_dist(x8, y8))           # Expect ~1.7209
