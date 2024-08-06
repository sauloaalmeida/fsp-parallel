import torch
import scipy
import cupy as cp
import numpy as np
from cuml.metrics import pairwise_distances as pairwise_distances_cuml
from scipy.spatial import distance
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import squareform
from pylibraft.common import Handle
from pylibraft.distance import pairwise_distance

def cdist_scipy(_a,_b):
    return distance.cdist(_a,_b)

def cdist_sklearn_st(_a,_b):
    return pairwise_distances(X = _a, Y= _b, n_jobs = 1)

def cdist_sklearn_mt(_a,_b):
    return pairwise_distances(X = _a, Y= _b,  n_jobs = -1)

def cdist_torch_CPU(_a,_b):
    return torch.cdist(torch.from_numpy(_a), torch.from_numpy(_b))

def cdist_torch_GPU(_a,_b):
    if torch.cuda.is_available():
        device = torch.device('cuda')  # Use GPU
    else:
        device = torch.device('cpu')  # Fallback to CPU if GPU is not available
        print("----->>>> running over CPU")

    # Convert NumPy array to PyTorch tensor and move to GPU
    torch_tensor_a = torch.tensor(_a, dtype=torch.float32).to(device)
    torch_tensor_b = torch.tensor(_b, dtype=torch.float32).to(device)

    # Calculate pairwise distances using torch.cdist
    return torch.cdist(torch_tensor_a,torch_tensor_b)

def cdist_rapids_GPU(_a, _b):
    in1 = cp.array(_a, dtype=cp.float32)
    in2 = cp.array(_b, dtype=cp.float32)
    output = pairwise_distance(in1, in2)
    return cp.asarray(output)

def main():

    np.random.seed(42)
    a = np.random.rand(10000,100)

    cdistScipyResult = cdist_scipy(a,a)

    cdistSklearnResult = cdist_sklearn_st(a,a)

    cdistSklearnResult = cdist_sklearn_mt(a,a)

    cdistTorchResultCPU = cdist_torch_CPU(a,a)

    cdistTorchResultGPU = cdist_torch_GPU(a,a)

    cdistRapidsResultGPU = cdist_rapids_GPU(a,a)

def check_difference():
    np.random.seed(42)
    a = np.random.rand(3,3)

    print(f"scify cdist:\n{distance.cdist(a,a)}")
    print(f"scify pdist:\n{distance.pdist(a)}")

    print(f"sklearn single thread cdist:\n{pairwise_distances(X=a, Y=a)}")
    print(f"sklearn single thread pdist:\n{squareform(pairwise_distances(X=a), checks=False)}")

    print(f"sklearn multi-thread cdist:\n{pairwise_distances(X=a, Y=a, n_jobs=-1)}")
    print(f"sklearn multi-thread pdist:\n{squareform(pairwise_distances(X=a, n_jobs=-1), checks=False)}")

    torch_sensor_a = torch.from_numpy(a)
    torch_sensor_b = torch.from_numpy(a)

    print(f"pytorch multi-thread cdist:\n{torch.cdist(torch_sensor_a, torch_sensor_a)}")
    print(f"pytorch multi-thread pdist:\n{squareform(torch.cdist(torch_sensor_a, torch_sensor_a), checks=False)}")


check_difference()