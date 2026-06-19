import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, init="kmeans++", max_iter=100, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None

    def _initialize_centroids(self, X):
        n_samples = X.shape[0]
        centroids = np.empty((self.n_clusters, X.shape[1]))
        rng = np.random.default_rng(self.random_state)

        if self.init == "forgy":
            idx = rng.choice(n_samples, size=self.n_clusters, replace=False)
            centroids = X[idx]

        elif self.init == "kmeans++":
            centroids[0] = X[rng.integers(n_samples)]
        
            for i in range(1, self.n_clusters):
                min_dist_sq = np.full(n_samples, np.inf)
                for j in range(i):
                    min_dist_sq = np.minimum(np.sum((X - centroids[j]) ** 2, axis=1), min_dist_sq)
                probabilities = min_dist_sq / min_dist_sq.sum()
                idx = rng.choice(n_samples, p=probabilities)
                centroids[i] = X[idx]
        
        else:
            raise ValueError("Init method should be either 'forgy' or 'kmeans++'.")

        return centroids
    
    def _assign_cluster(self, X, centroids):
        # n_samples = X.shape[0]

        # dist = np.zeros((n_samples, self.n_clusters))
        # for i in range(n_samples):
        #     for j in range(self.n_clusters):
        #         dist[i, j] = np.sqrt(np.sum((X[i] - centroids[j]) ** 2))
        
        # Vectorized therefore faster
        dist = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)

        labels = np.argmin(dist, axis=1)
        return labels

    def _update_centroids(self, X, labels, old_centroids):
        new_centroids = np.empty((self.n_clusters, X.shape[1]))

        for i in range(self.n_clusters):
            if len(X[labels == i]) == 0:
                new_centroids[i] = old_centroids[i]
            else:
                new_centroids[i] = X[labels == i].mean(axis=0)
        
        return new_centroids

    def _is_converged(self, old_centroids, new_centroids):
        return np.linalg.norm(new_centroids - old_centroids) < self.tol

    def _compute_inertia(self, X, labels):
        inertia = 0
        for i in range(self.n_clusters):
            inertia += np.sum((X[labels == i] - self.centroids_[i]) ** 2)
        
        return inertia

    def fit(self, X):
        X = np.asarray(X)
        if X.shape[0] < self.n_clusters:
            raise ValueError("Not enough data for current number of clusters.")

        # Initialize centroids
        new_centroids = self._initialize_centroids(X)

        for i in range(self.max_iter):
            # Assign cluster for each data point
            labels = self._assign_cluster(X, new_centroids)
            old_centroids = new_centroids.copy()
            
            # Update centroids
            new_centroids = self._update_centroids(X, labels, old_centroids)

            # Check convergence (i.e. if the new centroids are close enough to the old ones)
            if self._is_converged(old_centroids, new_centroids):
                break
        
        self.centroids_ = new_centroids
        self.labels_ = self._assign_cluster(X, self.centroids_)

        # Compute inertia
        self.inertia_ = self._compute_inertia(X, self.labels_)

        return self

    def predict(self, X):
        if self.centroids_ is None:
            raise ValueError("Model is not fitted.")
        
        X = np.asarray(X)
        labels = self._assign_cluster(X, self.centroids_)
        return labels

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_