import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

class RBFNetwork:
    def __init__(self, centers=None, n_centers=None, sigma=None, regularization=0.0):
        """
        Classe RBF (Radial Basis Function Network).
        :param centers: Centros (centróides) fornecidos manualmente (array-like). Se None, serão calculados automaticamente.
        :param n_centers: Número de centros a serem utilizados se centers não for fornecido (usado com KMeans).
        :param sigma: Parâmetro sigma para a função RBF. Se None, será calculado automaticamente.
        :param regularization: Termo de regularização (Ridge). Padrão 0 (sem regularização).
        """
        self.centers = centers
        self.n_centers = n_centers
        self.sigma = sigma
        self.regularization = regularization
        self.weights = None

    def _rbf(self, x, center):
        """Calcula a ativação da função RBF (base radial) para um ponto x e um centro."""
        print(f"center shape: {center.shape}")
        print(f"x shape: {x.shape}")
        return np.exp(-np.linalg.norm(x - center) ** 2 / (2 * self.sigma ** 2))

    def _calculate_interpolation_matrix(self, X):
        """Calcula a matriz de interpolações G para os dados de entrada X."""
        G = np.zeros((X.shape[0], len(self.centers)))
        for i, x in enumerate(X):
            for j, center in enumerate(self.centers):
                G[i, j] = self._rbf(x, center)
        return G

    def fit(self, X, y):
        """
        Ajusta os pesos da rede RBF aos dados de entrada X e saídas y.
        :param X: Dados de entrada (n_amostras x n_features).
        :param y: Saídas esperadas (n_amostras x 1).
        """
        # Definir os centros (centróides) se não foram fornecidos
        if self.centers is None:
            if self.n_centers is None:
                raise ValueError("Deve fornecer 'n_centers' ou 'centers'.")
            kmeans = KMeans(n_clusters=self.n_centers, random_state=42)
            kmeans.fit(X)
            self.centers = kmeans.cluster_centers_

        # Calcular sigma automaticamente, se não fornecido
        if self.sigma is None:
            dists = cdist(self.centers, self.centers)
            self.sigma = np.mean(dists)

        # Construir a matriz G de ativações
        G = self._calculate_interpolation_matrix(X)

        # Resolver pesos com regularização (Ridge Regression)
        reg_identity = self.regularization * np.eye(G.shape[1])
        self.weights = np.linalg.inv(G.T @ G + reg_identity) @ G.T @ y

    def predict(self, X):
        """
        Realiza previsões (binárias ou contínuas) com base nos dados de entrada X.
        :param X: Dados de entrada (n_amostras x n_features).
        :return: Previsões binárias (0 ou 1).
        """
        G = self._calculate_interpolation_matrix(X)
        predictions = G.dot(self.weights)
        return np.where(predictions >= 0.5, 1, 0)

    def predict_proba(self, X):
        """
        Calcula as probabilidades de saída (regressão).
        :param X: Dados de entrada (n_amostras x n_features).
        :return: Saídas contínuas (valores entre 0 e 1).
        """
        G = self._calculate_interpolation_matrix(X)
        return G.dot(self.weights)