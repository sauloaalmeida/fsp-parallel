import pytest
import pandas as pd
import numpy as np
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn import datasets
from fsp.options import Options
from fsp.fsp import fsp
from fsp.fsp import fsp_predict

import platform
import os
import numpy
import pandas
import scipy
import sklearn
class TestFsp:
    def test_fsp_default_execution(self):

        X,y = datasets.load_iris(return_X_y=True)

        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=20, random_state=42)

        option = Options(Standardize=0, return_full_history=1, return_full_dm=0, s_parameter=0.3, p_parameter=0.01, kmeans_random_state=0)

        fsp_output = fsp(X_train, y_train,opt=option)

        assert fsp_output['Mu'] == None
        assert fsp_output['Sigma'] == None
        assert fsp_output['ClassNames'].tolist() == [0,1,2]
        assert fsp_output['opt'] == option
        assert fsp_output['InsampleError'] == 0.023076923076923078
        assert fsp_output['NumIterations'] == 18

        assert pd.DataFrame(fsp_output['H']).shape == (19, 16)

        y_pred, y_pred_Proportion = fsp_predict(fsp_output, X_test)

        assert y_test.tolist() == [1,0,2,1,1,0,1,2,1,1,2,0,0,0,0,1,2,1,1,2]
        assert y_pred.shape == (20,)
        assert y_pred.tolist() == [1,0,2,1,1,0,1,2,2,1,2,0,0,0,0,1,2,1,1,2]
        assert y_pred_Proportion.shape == (20,)
        assert np.mean(y_test == y_pred) == 0.95
        assert y_pred_Proportion.tolist() == [1., 1., 1., 1., 1., 1., 1., 1., 0.7142857142857143 ,1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
