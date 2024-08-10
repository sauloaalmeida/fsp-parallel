import pytest
import numpy as np
import fsp.methods.kmeans.scipy_single_thread as kmeans_scipy_st
import fsp.methods.kmeans.sklearn_multi_thread as kmeans_sklearn_mt

class TestKmeansMethods:

    def setup_method(self, method):
        np.random.seed(42)
        observations=100
        features=10
        epsilon = 1e-10
        self.X = np.random.rand(observations,features)

    def test_kmeans_scipy_st(self):
        labels, centroids = kmeans_scipy_st.kmeans(X=self.X, k=3, random_state=42)
        print("indices:\n",labels)
        print("centroids:\n",centroids)

    def test_kmeans_sklearn_mt(self):
        labels, centroids = kmeans_sklearn_mt.kmeans(X=self.X, k=3, random_state=42)
        print("indices:\n",labels)
        print("centroids:\n",centroids)