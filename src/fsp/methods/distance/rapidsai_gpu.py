import cupy as cp
from cuml.metrics import pairwise_distances
from scipy.spatial.distance import squareform

def cdist(A, B):
    return pairwise_distances(A,B)

def pdist_dm1(A):
    return squareform(pairwise_distances(A, metric="sqeuclidean"), checks=False)

def cdist_dm1(A, B):
    return pairwise_distances(A,B, metric="sqeuclidean")

