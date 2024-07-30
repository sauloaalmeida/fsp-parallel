from cuml import KMeans

def kmeans(_X, _k, random_state):
    in1 = cp.array(_X, dtype=cp.float32)
    kmeans = KMeans(n_clusters=_k, max_iter=1000, random_state=random_state, n_init=1)
    kmeans.fit(in1)
    return kmeans.labels_, kmeans.cluster_centers_,