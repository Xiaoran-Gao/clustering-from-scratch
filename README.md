# Clustering From Scratch

Bare-bones Python implementations of classic clustering algorithms.

## Requirements

- Python 3.9+
- NumPy
- SciPy

## Algorithms

### K-Means

Partitions data into `k` clusters by repeatedly assigning points to the nearest centroid and updating centroids from the assigned points.

### Kernel K-Means

Extends K-Means with kernel functions so clusters can be separated in an implicit feature space.

### DBSCAN

Finds clusters as dense regions of points based on neighborhood distance and minimum sample thresholds. It can also identify noise points.

### Gaussian Mixture Model

Models data as a mixture of Gaussian distributions, typically using the Expectation-Maximization algorithm to estimate cluster probabilities.

## Todo

- [ ] Implement KMeans.
- [ ] Implement Kernel KMeans.
- [ ] Implement DBSCAN.
- [ ] Implement Gaussian Mixture Model with Expectation-Maximization.
- [ ] Add simple example datasets.
- [ ] Add visualizations for clustering results.
