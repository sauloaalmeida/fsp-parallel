import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import squareform

def pdist_dm1(A):
    return squareform(pairwise_distances(X=A, metric='sqeuclidean',  n_jobs = -1), checks=False)

def cdist_dm1(A, B):
    return pairwise_distances(X = A, Y= B, metric='sqeuclidean',  n_jobs = -1)

def cdist(A, B):
    return pairwise_distances(X = A, Y= B,  n_jobs = -1)