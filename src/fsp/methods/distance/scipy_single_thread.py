import numpy as np
import fsp.methods.distance.util as util
from scipy.spatial import distance

def dm_case1(A, B):
    #Initializing not vectorized values
    Na, d, Nb, ha2, hb2, logha2, loghb2 = util.calculateConstValuesDmCase1(A, B)

    #calculating variance
    Va = np.var(A, axis=0)
    Vb = np.var(B, axis=0)

    meanVa = np.mean(Va)
    meanVb = np.mean(Vb)

    sum_ab = np.sum(np.exp( -distance.cdist(A, B)/(2*ha2*meanVa+2*hb2*meanVb) ))

    if sum_ab == 0:
        return 0;
    else:
        sum_a  = np.sum(np.exp( -distance.pdist(A,'sqeuclidean')/(4*ha2*meanVa)))
        sum_b  = np.sum(np.exp( -distance.pdist(B,'sqeuclidean')/(4*hb2*meanVb)))
        return -d*np.log(2) - (d/2)*( logha2 + loghb2 + np.log(meanVa) + np.log(meanVb) ) + d*np.log(ha2*meanVa+hb2*meanVb) - 2*np.log(sum_ab) +  np.log(Na + 2*sum_a) +  np.log(Nb + 2*sum_b)

def dm_case2(A, B):
    #Initializing not vectorized values
    Na, d, Nb, ha2, hb2, logha2, loghb2 = util.calculateConstValuesDmCase2(A, B)

    Va = np.var(A, axis=0)
    Vb = np.var(B, axis=0)

    Va[Va<10**(-12)] = 10**(-12)
    Vb[Vb<10**(-12)] = 10**(-12)

    w = np.sqrt(ha2*Va+hb2*Vb)
    sum_ab = np.sum(np.exp( -distance.cdist(A/w, B/w, metric='sqeuclidean'))**2/2)

    if sum_ab == 0:
        return 0;
    else:
        logprodVa = np.sum(np.log(Va))
        logprodVb = np.sum(np.log(Vb))
        logprodVab = np.sum(np.log(ha2*Va+hb2*Vb))

        sum_a = np.sum(np.exp( - distance.pdist(A/np.sqrt(Va), metric='sqeuclidean')**2/(4*ha2)))
        sum_b = np.sum(np.exp( - distance.pdist(B/np.sqrt(Vb), metric='sqeuclidean')**2/(4*hb2)))

        return -d*np.log(2) - (d/2)*( logha2 + loghb2 ) - (1/2)*( logprodVa + logprodVb ) + logprodVab -2*np.log(sum_ab) + np.log(Na + 2*sum_a) + np.log(Nb + 2*sum_b)

def cdist(A, B):
    return distance.cdist(A,B)


# def pdist_dm1(A):
#     return distance.pdist(A,'sqeuclidean')

# def cdist_dm1(A, B):
#     return distance.cdist(A,B,'sqeuclidean')

