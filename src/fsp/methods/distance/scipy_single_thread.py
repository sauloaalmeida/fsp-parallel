import numpy as np
from scipy.spatial import distance

def pdist_dm1(A):
    return distance.pdist(A,'sqeuclidean')

def cdist_dm1(A, B):
    return distance.cdist(A,B,'sqeuclidean')

def cdist(A, B):
    return distance.cdist(A,B)
