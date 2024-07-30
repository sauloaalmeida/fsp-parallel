import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import squareform

def pdist_dm1(A):
    return squareform(pairwise_distances(X=A, metric='sqeuclidean',  n_jobs = -1), checks=False)

def cdist_dm1(A, B):
    return pairwise_distances(X = A, Y= B, metric='sqeuclidean',  n_jobs = -1)

# def pdist_dm2(A, V):
#     return squareform(pairwise_distances(X=A, metric='mahalanobis',  n_jobs = -1, VI=np.linalg.inv(np.diag(V))), checks=False)

# def cdist_dm2(A, B, Va, Vb, ha2, hb2):
#     return pairwise_distances(X = A, Y= B,  n_jobs = -1, metric='mahalanobis', VI=np.linalg.inv(np.diag(ha2*Va+hb2*Vb)))

def cdist(A, B):
    return pairwise_distances(X = A, Y= B,  n_jobs = -1)