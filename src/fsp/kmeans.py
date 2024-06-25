from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans2, kmeans, vq
from cuml import KMeans as cuMlKmeans
import cupy as cp
import numpy as np

RANDOM_STATE_SEED = 42


#remocao do parametro n_jobs
#https://scikit-learn.org/0.23/auto_examples/release_highlights/plot_release_highlights_0_23_0.html#scalability-and-stability-improvements-to-kmeans
def kmeans_sklearn_mt(_X, _initial_centroids):
    kmeans_sklearn = KMeans(n_clusters=_initial_centroids.shape[0], init=_initial_centroids, n_init=1, max_iter=100, random_state=RANDOM_STATE_SEED)
    kmeans_sklearn.fit(_X)
    return kmeans_sklearn.cluster_centers_, kmeans_sklearn.labels_

def kmeans_scipy(_X, _initial_centroids):
    centroids_scipy, _ = kmeans(_X, _initial_centroids, iter=1)
    idx_scipy, distortion = vq(_X, centroids_scipy)
    return centroids_scipy, idx_scipy

def kmeans_cuml(_X, _initial_centroids):
    kmeans_float = cuMlKmeans(n_clusters=_initial_centroids.shape[0])
    in1 = cp.array(_X, dtype=cp.float32)
    kmeans_float.fit(in1)

    return kmeans_float.cluster_centers_, kmeans_float.labels_

def main():

    np.random.seed(RANDOM_STATE_SEED)
    X = np.random.rand(1000000,100)
    k = 4

    #get k random indices from data
    initial_centroids_indices = np.random.choice(X.shape[0], k, replace=False)

    #Initialize this observations as set of centroids
    initial_centroids = X[initial_centroids_indices]

    scipy_centroids, scipy_idx = kmeans_scipy(X,initial_centroids)

    sklearn_centroids, sklearn_idx = kmeans_sklearn_mt(X,initial_centroids)

    cuml_centroids, cuml_idx = kmeans_cuml(X,initial_centroids)

    # Results
    # print("Initial centroids:\n", initial_centroids)
    # print("Scipy centroids:\n", scipy_centroids)
    # print("Scipy idx:\n", scipy_idx)
    # print("Sklearn centroids:\n", sklearn_centroids)
    # print("Sklearn idx:\n", sklearn_idx)
    # print("cuML centroids:\n", cuml_centroids)
    # print("cuML idx:\n", cuml_idx)

    # print("Centroids similar:", np.allclose(scipy_centroids, sklearn_centroids))
    # print("Labels similar:", np.array_equal(scipy_idx, sklearn_idx))

    # kmeans2_scipy(X,4)

    # # Check if centroids are similar
    # centroids_scipy_sorted = np.sort(centroids_scipy, axis=0)
    # centroids_sklearn_sorted = np.sort(centroids_sklearn, axis=0)


def main2():
    # Generate synthetic data
    np.random.seed(42)
    data = np.random.rand(2000, 2)  # 2000 points in 2D

    # Number of clusters
    k = 3

    # Fixed random seed for reproducibility
    random_state = 42

    # Custom initialization using random points from the data
    np.random.seed(random_state)
    initial_centroids_indices = np.random.choice(data.shape[0], k, replace=False)
    initial_centroids = data[initial_centroids_indices]

    # Ensure identical initial centroids
    initial_centroids_copy = np.copy(initial_centroids)

    # K-Means using scipy kmeans2 with custom initial centroids
    centroids_scipy, _ = kmeans2(data, initial_centroids, iter=100, minit='matrix')
    idx_scipy, _ = vq(data, centroids_scipy)

    # K-Means using sklearn with custom initial centroids
    kmeans_sklearn = KMeans(n_clusters=k, init=initial_centroids_copy, n_init=1, max_iter=100, tol=1e-4, random_state=random_state)
    kmeans_sklearn.fit(data)
    centroids_sklearn = kmeans_sklearn.cluster_centers_
    idx_sklearn = kmeans_sklearn.labels_

    # Results
    print("Initial centroids:\n", initial_centroids)
    print("Scipy centroids:\n", centroids_scipy)
    print("Scipy labels:\n", idx_scipy)
    print("Sklearn centroids:\n", centroids_sklearn)
    print("Sklearn labels:\n", idx_sklearn)

    # Check if centroids are similar
    centroids_scipy_sorted = np.sort(centroids_scipy, axis=0)
    centroids_sklearn_sorted = np.sort(centroids_sklearn, axis=0)

    print("Centroids similar:", np.allclose(centroids_scipy_sorted, centroids_sklearn_sorted))

    # Check if labels are the same
    print("Labels similar:", np.array_equal(idx_scipy, idx_sklearn))

main()