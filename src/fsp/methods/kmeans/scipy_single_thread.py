from scipy.cluster.vq import vq
from scipy.cluster.vq import kmeans as kmeans_scipy

def kmeans(X, k, random_state):
    centroids_scipy, _ = kmeans_scipy(obs=X, k_or_guess=k, iter=1, seed=random_state)
    idx_scipy, distortion = vq(obs=X, code_book=centroids_scipy)
    
    return idx_scipy, centroids_scipy