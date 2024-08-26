import torch
import numpy as np
import fsp.methods.distance.util as util

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

def _createMainTensorsAndDevice(A,B,deviceName):
    #init device
    dev = _initDevice(deviceName)

    #create tensorA and bensorB
    tensorA = _createTensor(A, dev)
    tensorB = _createTensor(B, dev)

    #create tensors with variance of A and B
    Va = torch.var(tensorA, dim=0, correction=0)
    Vb = torch.var(tensorB, dim=0, correction=0)

    return  tensorA, tensorB, Va, Vb, dev

def cdist(A, B, deviceName="cpu"):
    tensorA, tensorB, _ = _createMainTensorsAndDevice(A, B, deviceName)
    return torch.cdist(tensorA,tensorB)

def dm_case1(A,B, deviceName="cpu"):

    #Initializing not vectorized values
    Na, d, Nb, ha2, hb2, logha2, loghb2 = util.calculateConstValuesDmCase1(A, B)

    #initalizing tensors and device used (cpu or gpu)
    tensorA, tensorB, Va, Vb, dev = _createMainTensorsAndDevice(A, B, deviceName)

    meanVa = torch.mean(Va)
    meanVb = torch.mean(Vb)

    sum_ab = torch.sum(torch.exp( -torch.pow(torch.cdist(tensorA, tensorB),2)/(2*ha2*meanVa+2*hb2*meanVb)))

    if sum_ab == 0:
        return torch.tensor(0).cpu();
    else:
        sum_a = torch.sum(torch.exp( -torch.pow(torch.nn.functional.pdist(tensorA),2)/(4*ha2*meanVa)))
        sum_b = torch.sum(torch.exp( -torch.pow(torch.nn.functional.pdist(tensorB),2)/(4*hb2*meanVb)))
        D_CS = -d*torch.log(_createTensor(2, dev)) - (d/2)*( logha2 + loghb2 + torch.log(meanVa) + torch.log(meanVb) ) + d*torch.log(ha2*meanVa+hb2*meanVb) - 2*torch.log(sum_ab) +  torch.log(Na + 2*sum_a) +  torch.log(Nb + 2*sum_b)
        return D_CS.cpu()

def dm_case2(A,B, deviceName="cpu"):

    #initalizing tensors and device used (cpu or gpu)
    Na, d, Nb, ha2, hb2, logha2, loghb2 = util.calculateConstValuesDmCase2(A, B)

    #initalizing tensors and device used (cpu or gpu)
    tensorA, tensorB, Va, Vb, dev = _createMainTensorsAndDevice(A, B, deviceName)

    #define a minimum value to variance tensors
    Va[Va<10**(-12)] = 10**(-12)
    Vb[Vb<10**(-12)] = 10**(-12)

    w = torch.sqrt(ha2*Va+hb2*Vb)
    sum_ab = torch.sum(torch.exp( -torch.pow(torch.cdist(tensorA/w, tensorB/w),2))**2/2)

    if sum_ab == 0:
        return torch.tensor(0).cpu();
    else:
        logprodVa = torch.sum(torch.log(Va))
        logprodVb = torch.sum(torch.log(Vb))
        logprodVab = torch.sum(torch.log(ha2*Va+hb2*Vb))

        sum_a = torch.sum(torch.exp( -torch.pow(torch.nn.functional.pdist(tensorA/torch.sqrt(Va)),2)**2/(4*ha2)))
        sum_b = torch.sum(torch.exp( -torch.pow(torch.nn.functional.pdist(tensorB/torch.sqrt(Vb)),2)**2/(4*hb2)))

        D_CS = -d*torch.log(_createTensor(2, dev)) - (d/2)*( logha2 + loghb2 ) - (1/2)*( logprodVa + logprodVb ) + logprodVab -2*torch.log(sum_ab) + torch.log(Na + 2*sum_a) + torch.log(Nb + 2*sum_b)

        return D_CS.cpu()

