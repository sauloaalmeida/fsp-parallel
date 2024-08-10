from cuml import KMeans
import cupy as cp

def kmeans(X, k, random_state):

    random = random_state
    if random_state == None:
        random = 1

    in1 = cp.array(X, dtype=cp.float32)
    kmeans = KMeans(n_clusters=int(k), max_iter=1000, random_state=random, n_init=1)
    kmeans.fit(in1)
    return kmeans.labels_.get(), kmeans.cluster_centers_.get()