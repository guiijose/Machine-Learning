import numpy as np

class RBFNetwork:
    def __init__(self, centers, sigma=1.0):
        self.centers = centers
        self.sigma = sigma
        self.weights = None

    def _rbf(self, x, center):
        return np.exp(-np.linalg.norm(x - center) ** 2 / (2 * self.sigma ** 2))

    def _calculate_interpolation_matrix(self, X):
        G = np.zeros((X.shape[0], len(self.centers)))
        for i, x in enumerate(X):
            for j, center in enumerate(self.centers):
                G[i, j] = self._rbf(x, center)
        return G

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        G = self._calculate_interpolation_matrix(X)
        self.weights = np.linalg.pinv(G).dot(y)

    def predict(self, X):
        X = np.array(X, dtype=float)
        G = self._calculate_interpolation_matrix(X)
        predictions = G.dot(self.weights)
        return np.where(predictions >= 0.5, 1, 0)
    



