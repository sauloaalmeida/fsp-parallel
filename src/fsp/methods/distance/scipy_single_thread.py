import numpy as np
from scipy.spatial import distance

def pdist_dm1(A):
    return distance.pdist(A,'sqeuclidean')

def cdist_dm1(A, B):
    return distance.cdist(A,B,'sqeuclidean')

# def pdist_dm2(A, V):
#     return distance.pdist(A, 'mahalanobis', VI=np.linalg.inv(np.diag(V)))

# def cdist_dm2(A, B, Va, Vb, ha2, hb2):
#     return distance.cdist(A,B, 'mahalanobis', VI=np.linalg.inv(np.diag(ha2*Va+hb2*Vb)))

def cdist(A, B):
    return distance.cdist(A,B)
