from sklearn.cluster import KMeans

def kmeans(X, k, random_state):
    kmeans = KMeans(n_clusters=k, max_iter=1000, random_state=random_state, n_init=1).fit(X)
    return kmeans.labels_, kmeans.cluster_centers_