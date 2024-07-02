import sys
import inspect
import time
import torch
import cupy as cp
import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans2, kmeans, vq
from cuml import KMeans as cuMlKmeans
from kmeans_pytorch import kmeans as torchKMeans


RANDOM_STATE_SEED = 42

def printExecutionDetail(_startTime, _X, _k, _name, _idExec):
    elipsedTime = time.time() - _startTime
    print(f"{_idExec},{_name},{_k},{_X.shape[1]},{_X.shape[0]},{elipsedTime}")


#remocao do parametro n_jobs
#https://scikit-learn.org/0.23/auto_examples/release_highlights/plot_release_highlights_0_23_0.html#scalability-and-stability-improvements-to-kmeans
def kmeans_sklearn_mt_CPU(_X, _k, _idExec):
    start = time.time()
    kmeans_sklearn = KMeans(n_clusters=_k, n_init=1, max_iter=100, random_state=RANDOM_STATE_SEED)
    kmeans_sklearn.fit(_X)
    printExecutionDetail(start,_X,_k,inspect.stack()[0][3],_idExec)
    return kmeans_sklearn.cluster_centers_, kmeans_sklearn.labels_

def kmeans_scipy_CPU(_X, _k, _idExec):
    start = time.time()
    centroids_scipy, _ = kmeans(_X, _k, iter=1)
    idx_scipy, distortion = vq(_X, centroids_scipy)
    printExecutionDetail(start,_X,_k,inspect.stack()[0][3],_idExec)
    return centroids_scipy, idx_scipy

def kmeans_cuml_GPU(_X, _k, _idExec):
    start = time.time()
    kmeans_float = cuMlKmeans(n_clusters=_k)
    in1 = cp.array(_X, dtype=cp.float32)
    kmeans_float.fit(in1)
    printExecutionDetail(start,_X,_k,inspect.stack()[0][3],_idExec)
    return kmeans_float.cluster_centers_, kmeans_float.labels_

def __kmeans_torch(_X, _k, _device):
    data = torch.tensor(_X)
    cluster_ids_x, cluster_centers = torchKMeans(
        X=data, tqdm_flag=False, num_clusters=_k, distance='euclidean', device=torch.device(_device)
    )
    return cluster_centers, cluster_ids_x

def kmeans_torch_GPU(_X, _k, _idExec):
    if not torch.cuda.is_available():
        raise Exception("Sorry, No GPU available")

    start = time.time()
    cluster_centers, cluster_ids_x = __kmeans_torch(_X, _k, 'cuda')  # Use GPU
    printExecutionDetail(start,_X,_k,inspect.stack()[0][3],_idExec)
    return cluster_centers, cluster_ids_x

def kmeans_torch_CPU(_X, _k, _idExec):
    start = time.time()
    cluster_centers, cluster_ids_x = __kmeans_torch(_X, _k, 'cpu')
    printExecutionDetail(start,_X,_k,inspect.stack()[0][3],_idExec)
    return cluster_centers, cluster_ids_x


def main():

    _idExec = sys.argv[1]
    _functionName = sys.argv[2]
    _k = int(sys.argv[3])
    _featuresAmount = int(sys.argv[4])
    _observationsAmount = int(sys.argv[5])

    np.random.seed(RANDOM_STATE_SEED)
    X = np.random.rand(_observationsAmount,_featuresAmount)
    k = _k
    idExec = _idExec

    eval(f"{_functionName}(X,k,idExec)")


main()


    # scipy_centroids, scipy_idx = kmeans_scipy(X,k)

    # sklearn_centroids, sklearn_idx = kmeans_sklearn_mt(X,k)

    # cuml_centroids, cuml_idx = kmeans_cuml(X,k)

    # torchCPU_centroids, torchCPU_idx = kmeans_torch_CPU(X,k)

    # torchGPU_centroids, torchGPU_idx = kmeans_torch_GPU(X,k)

    # Results
    # print("Initial centroids:\n", initial_centroids)
    # print("Scipy centroids:\n", scipy_centroids)
    # print("Scipy idx:\n", scipy_idx)
    # print("Sklearn centroids:\n", sklearn_centroids)
    # print("Sklearn idx:\n", sklearn_idx)
    # print("cuML centroids:\n", cuml_centroids)
    # print("cuML idx:\n", cuml_idx)

    # print("Centroids similar:", np.allclose(scipy_centroids, sklearn_centroids))
    # print("Labels similar:", np.array_equal(scipy_idx, sklearn_idx))

    # kmeans2_scipy(X,4)

    # # Check if centroids are similar
    # centroids_scipy_sorted = np.sort(centroids_scipy, axis=0)
    # centroids_sklearn_sorted = np.sort(centroids_sklearn, axis=0)

