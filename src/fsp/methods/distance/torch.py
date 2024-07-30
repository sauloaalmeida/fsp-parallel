import torch
import numpy as np

def _initDevice(deviceName):
    if deviceName == 'cuda':
        if torch.cuda.is_available():
            return torch.device(deviceName)
        else:
            print("Trying to using GPU but it's not available. Using CPU!")

    return torch.device(deviceName)


def _createTensor(data, dev):
    if dev.type == 'cuda':
        return torch.tensor(data, dtype=torch.float32).to(dev)
    else:
        print("Criando cpu tensor")
        return torch.tensor(data, dtype=torch.float64).to(dev)

def cdist(A, B, deviceName="cpu"):
    dev = _initDevice(deviceName)
    tensorA = _createTensor(A, dev)
    tensorB = _createTensor(B, dev)
    return torch.cdist(tensorA,tensorB)

def cdist_dm1(A, B, deviceName="cpu"):
    return torch.pow(cdist(A, B, deviceName),2)

def pdist_dm1(A, deviceName="cpu"):
    dev = _initDevice(deviceName)
    tensorA = _createTensor(A, dev)
    return torch.pow(torch.nn.functional.pdist(tensorA),2)

def pdist_dm2(A, V, deviceName="cpu"):
    dev = _initDevice(deviceName)
    tensorA = _createTensor(A, dev)
    tensorV = _createTensor(V, dev)

    #create inverse matrix of diagonal
    VI = torch.linalg.inv(torch.diag(tensorV))

    # total number points
    n = A.shape[0]

    #getting only the matrix upper area
    i_upper = torch.triu_indices(n, n, 1)

    # Calculate diff for each pair
    delta = tensorA[i_upper[0]] - tensorA[i_upper[1]]

    #Calculate matrix product between delta and VI
    m1 = torch.mm(delta, VI)

    # Calculate scalar product between m1 and delta
    m2 = torch.sum(m1 * delta, dim=1)

    # Get square values
    return torch.sqrt(m2)

def cdist_dm2(A, B, Va, Vb, ha2, hb2, deviceName="cpu"):
    dev = _initDevice(deviceName)
    tensorA = _createTensor(A, dev)
    tensorB = _createTensor(B, dev)
    tensorVa = _createTensor(Va, dev)
    tensorVb = _createTensor(Vb, dev)
    tensorHa2 = _createTensor(ha2, dev)
    tensorHb2 = _createTensor(hb2, dev)

    #create inverse matrix of diagonal
    VI = torch.linalg.inv(torch.diag(ha2*Va+hb2*Vb))

