import numpy as np
from collections import deque

class DBSCAN:
    def __init__(self, eps=0.1, min_samples=3):
        self.eps = eps
        self.min_samples = min_samples

        self.labels_ = None

    def _find_neighbors(self, X, i):
        neighbors = []
        for j in range(X.shape[0]):
            dist = np.linalg.norm(X[j] - X[i])
            if dist <= self.eps:
                neighbors.append(j)

        return neighbors

    def _is_core_point(self, neighbors):
        return len(neighbors) >= self.min_samples
    
    def fit(self, X):
        X = np.asarray(X)
        n_samples = X.shape[0]

        # Initialize
        self.labels_ = np.full(n_samples, -1) # -1 for noise
        visited = np.full(n_samples, False)
        curr_clust = 0

        for i in range(n_samples):
            if visited[i]:
                continue
            visited[i] = True

            neighbors = self._find_neighbors(X, i)
            if self._is_core_point(neighbors):
                self.labels_[i] = curr_clust
                queue = deque(neighbors)

                # Do BFS
                while len(queue) > 0:
                    j = queue.popleft()
                    if self.labels_[j] == -1:
                        self.labels_[j] = curr_clust
                    if not visited[j]:
                        visited[j] = True
                        neighbors_j = self._find_neighbors(X, j)
                        if self._is_core_point(neighbors_j):
                            queue.extend(neighbors_j)
            
                curr_clust += 1

        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_