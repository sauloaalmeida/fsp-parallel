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

