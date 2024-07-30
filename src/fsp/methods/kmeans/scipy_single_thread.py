from scipy.cluster.vq import kmeans, vq

def kmeans_scipy_CPU(_X, _k):
    centroids_scipy, _ = kmeans(_X, _k, iter=1)
    idx_scipy, distortion = vq(_X, centroids_scipy)
    return idx_scipy, centroids_scipy