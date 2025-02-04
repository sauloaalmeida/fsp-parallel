import numpy as np
import fsp.options as options

from scipy.spatial import distance

def Divergence_Measure(A,B,opt):

    if opt.dm_case == 1:
        return opt.getDistanceMethod().dm_case1(A, B)
    elif opt.dm_case == 2:
        return opt.getDistanceMethod().dm_case2(A, B)