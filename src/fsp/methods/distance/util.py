import numpy as np

def getShapes(A,B):
    #Define Na and Nb, d
    Na, d = A.shape
    Nb, _ = B.shape

    return Na, d, Nb

def calculateConstValuesDmCase1(A, B):

    #get shapes of datasets
    Na, d, Nb = getShapes(A,B)

    #define values not vectorized (consts for this execution)
    ha2 = ( 4/((2*d+1)*Na) )**(2/(d+4))
    hb2 = ( 4/((2*d+1)*Nb) )**(2/(d+4))

    logha2 = (2/(d+4))*( np.log(4) - np.log(2*d+1) - np.log(Na) )
    loghb2 = (2/(d+4))*( np.log(4) - np.log(2*d+1) - np.log(Nb) )

    return Na, d, Nb, ha2, hb2, logha2, loghb2

def calculateConstValuesDmCase2(A, B):

    #get shapes of datasets
    Na, d, Nb = getShapes(A,B)

    #define values not vectorized (consts for this execution)
    ha2 = ( 4/((d+2)*Na) )**(2/(d+4))
    hb2 = ( 4/((d+2)*Nb) )**(2/(d+4))

    logha2 = (2/(d+4))*( np.log(4) - np.log(d+2) - np.log(Na) )
    loghb2 = (2/(d+4))*( np.log(4) - np.log(d+2) - np.log(Nb) )

    return Na, d, Nb, ha2, hb2, logha2, loghb2