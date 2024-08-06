import numpy as np
import fsp.options as options

from scipy.spatial import distance

def dm_case1(A,B, distance_method):
    #Define Na and Nb, d
    Na, d = A.shape
    Nb, _ = B.shape

    #Calculate the variance of A and B
    Va = np.var(A, axis=0)
    Vb = np.var(B, axis=0)

    ha2 = ( 4/((2*d+1)*Na) )**(2/(d+4))
    hb2 = ( 4/((2*d+1)*Nb) )**(2/(d+4))

    logha2 = (2/(d+4))*( np.log(4) - np.log(2*d+1) - np.log(Na) )
    loghb2 = (2/(d+4))*( np.log(4) - np.log(2*d+1) - np.log(Nb) )

    meanVa = np.mean(Va)
    meanVb = np.mean(Vb)

    sum_a  = np.sum(np.exp( -distance_method.pdist_dm1(A)/(4*ha2*meanVa)))
    sum_b  = np.sum(np.exp( -distance_method.pdist_dm1(B)/(4*ha2*meanVa)))
    sum_ab = np.sum(np.exp( -distance_method.cdist_dm1(A, B)/(2*ha2*meanVa+2*hb2*meanVb) ))

    if sum_ab == 0:
        D_CS = 0
    else:
        D_CS = -d*np.log(2) - (d/2)*( logha2 + loghb2 + np.log(meanVa) + np.log(meanVb) ) + d*np.log(ha2*meanVa+hb2*meanVb) - 2*np.log(sum_ab) +  np.log(Na + 2*sum_a) +  np.log(Nb + 2*sum_b)

    return D_CS

def dm_case2(A,B, distance_method):
    #Define Na and Nb, d
    Na, d = A.shape
    Nb, _ = B.shape

    #Calculate the variance of A and B
    Va = np.var(A, axis=0)
    Vb = np.var(B, axis=0)

    Va[Va<10**(-12)] = 10**(-12)
    Vb[Vb<10**(-12)] = 10**(-12)

    ha2 = ( 4/((d+2)*Na) )**(2/(d+4))
    hb2 = ( 4/((d+2)*Nb) )**(2/(d+4))

    logha2 = (2/(d+4))*( np.log(4) - np.log(d+2) - np.log(Na) )
    loghb2 = (2/(d+4))*( np.log(4) - np.log(d+2) - np.log(Nb) )

    logprodVa = np.sum(np.log(Va))
    logprodVb = np.sum(np.log(Vb))
    logprodVab = np.sum(np.log(ha2*Va+hb2*Vb))

    w = np.sqrt(ha2*Va+hb2*Vb)
    sum_a = np.sum(np.exp( - distance_method.pdist_dm1(A/np.sqrt(Va))**2/(4*ha2)))
    sum_b = np.sum(np.exp( - distance_method.pdist_dm1(B/np.sqrt(Vb))**2/(4*hb2)))
    sum_ab = np.sum(np.exp( - distance_method.cdist_dm1(A/w,B/w))**2/2)

    if sum_ab == 0:
        D_CS = 0;
    else:
        D_CS = -d*np.log(2) - (d/2)*( logha2 + loghb2 ) - (1/2)*( logprodVa + logprodVb ) + logprodVab -2*np.log(sum_ab) + np.log(Na + 2*sum_a) + np.log(Nb + 2*sum_b)

    return D_CS

def Divergence_Measure(A,B,opt):

    if opt.dm_case == 1:
        return dm_case1(A, B, opt.getDistanceMethod())
    elif opt.dm_case == 2:
        return dm_case2(A, B, opt.getDistanceMethod())