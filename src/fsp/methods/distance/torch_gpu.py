import fsp.methods.distance.torch as method_torch

def dm_case1(A,B):
    return method_torch.dm_case1(A,B,"cuda")

def dm_case2(A,B):
    return method_torch.dm_case2(A,B,"cuda")

def cdist(A, B):
    return method_torch.cdist(A,B,"cuda").cpu().numpy()