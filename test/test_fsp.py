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

        option = Options(kmeans_random_state = 42)

        fsp_output = fsp(X_train, y_train, opt = option)

        assert fsp_output['Mu'] == None
        assert fsp_output['Sigma'] == None
        assert fsp_output['ClassNames'].tolist() == [0,1,2]
        assert fsp_output['opt'] == option
        assert fsp_output['InsampleError'] == 0.023076923076923078
        assert fsp_output['NumIterations'] == 29

        assert pd.DataFrame(fsp_output['H']).shape == (10, 16)

        y_pred, y_pred_Proportion = fsp_predict(fsp_output, X_test, opt=option)

        assert y_test.tolist() == [1,0,2,1,1,0,1,2,1,1,2,0,0,0,0,1,2,1,1,2]
        assert y_pred.shape == (20,)
        #TODO: assert y_pred.tolist() == [1,0,2,1,1,0,1,2,2,1,2,0,0,0,0,1,2,1,1,2]
        assert y_pred_Proportion.shape == (20,)
        #TODO: assert np.mean(y_test == y_pred) == 0.95
        #TODO: assert y_pred_Proportion.tolist() == [1., 1., 1., 1., 1., 1., 0.9565217391304348, 1., 0.625, 0.9565217391304348, 0.9166666666666666, 1., 1., 1., 1., 1., 1., 0.9565217391304348, 0.9565217391304348, 0.9166666666666666]

    def test_fsp_dm_case1_dist_scipy_st_kmeans_skylearn_mt(self):

        X,y = datasets.load_iris(return_X_y=True)

        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=20, random_state=42)

        option = Options(dm_case=1,
                        return_full_history=True,
                        return_full_dm=True,
                        s_parameter=0.3,
                        p_parameter=0.01,
                        kmeans_random_state=0)

        fsp_output = fsp(X_train, y_train,opt=option)

        assert fsp_output['Mu'] == None
        assert fsp_output['Sigma'] == None
        assert fsp_output['ClassNames'].tolist() == [0,1,2]
        assert fsp_output['opt'] == option
        assert fsp_output['InsampleError'] == 0.023076923076923078
        assert fsp_output['NumIterations'] == 18

        assert len(fsp_output['H']['dm']) == 19

        assert fsp_output['H']['dm'][0][0].tolist() == [44.82400966459582, 115.32846043223826, 3.6528890346393412]
        assert fsp_output['H']['dm'][9][0].tolist() == [0., 0., 3.8127838639057305]
        assert fsp_output['H']['dm'][18][0].tolist() == [0., 0., 0.]

        assert pd.DataFrame(fsp_output['H']).shape == (19, 16)

        y_pred, y_pred_Proportion = fsp_predict(fsp_output, X_test, opt=option)

        assert y_test.tolist() == [1,0,2,1,1,0,1,2,1,1,2,0,0,0,0,1,2,1,1,2]
        assert y_pred.shape == (20,)
        assert y_pred.tolist() == [1,0,2,1,1,0,1,2,2,1,2,0,0,0,0,1,2,1,1,2]
        assert y_pred_Proportion.shape == (20,)
        assert np.mean(y_test == y_pred) == 0.95
        assert y_pred_Proportion.tolist() == [1., 1., 1., 1., 1., 1., 1., 1., 0.7142857142857143 ,1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]

    def test_fsp_dm_case1_dist_sklearn_mt_kmeans_skylearn_mt(self):

        X,y = datasets.load_iris(return_X_y=True)

        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=20, random_state=42)

        option = Options(dm_case=1,
                        return_full_history=True,
                        return_full_dm=True,
                        s_parameter=0.3,
                        p_parameter=0.01,
                        kmeans_random_state=0,
                        distance_method=2)

        fsp_output = fsp(X_train, y_train,opt=option)

        assert fsp_output['Mu'] == None
        assert fsp_output['Sigma'] == None
        assert fsp_output['ClassNames'].tolist() == [0,1,2]
        assert fsp_output['opt'] == option
        assert fsp_output['InsampleError'] == 0.023076923076923078
        assert fsp_output['NumIterations'] == 18

        assert len(fsp_output['H']['dm']) == 19

        assert fsp_output['H']['dm'][0][0].tolist() == [44.82400966459582, 115.32846043223826, 3.6528890346393412]
        assert fsp_output['H']['dm'][9][0].tolist() == [0., 0., 3.8127838639057305]
        assert fsp_output['H']['dm'][18][0].tolist() == [0., 0., 0.]

        assert pd.DataFrame(fsp_output['H']).shape == (19, 16)

        y_pred, y_pred_Proportion = fsp_predict(fsp_output, X_test, opt=option)

        assert y_test.tolist() == [1,0,2,1,1,0,1,2,1,1,2,0,0,0,0,1,2,1,1,2]
        assert y_pred.shape == (20,)
        assert y_pred.tolist() == [1,0,2,1,1,0,1,2,2,1,2,0,0,0,0,1,2,1,1,2]
        assert y_pred_Proportion.shape == (20,)
        assert np.mean(y_test == y_pred) == 0.95
        assert y_pred_Proportion.tolist() == [1., 1., 1., 1., 1., 1., 1., 1., 0.7142857142857143 ,1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]

    def test_fsp_dm_case2_dist_scipy_st_kmeans_skylearn_mt(self):

        X,y = datasets.load_iris(return_X_y=True)

        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=20, random_state=42)

        option = Options(return_full_history=True,
                        return_full_dm=True,
                        s_parameter=0.3,
                        p_parameter=0.01,
                        kmeans_random_state=0)

        fsp_output = fsp(X_train, y_train,opt=option)

        assert fsp_output['Mu'] == None
        assert fsp_output['Sigma'] == None
        assert fsp_output['ClassNames'].tolist() == [0,1,2]
        assert fsp_output['opt'] == option
        assert fsp_output['InsampleError'] == 0.023076923076923078
        assert fsp_output['NumIterations'] == 18

        assert len(fsp_output['H']['dm']) == 19

        #assert fsp_output['H']['dm'][0][0].tolist() == [55.95265107450104, 137.76852954539308, 4.553404433542719]
        #assert fsp_output['H']['dm'][9][0].tolist() == [0., 0., 3.0939923488322814]
        assert fsp_output['H']['dm'][18][0].tolist() == [0., 0., 0.]

        assert pd.DataFrame(fsp_output['H']).shape == (19, 16)

        y_pred, y_pred_Proportion = fsp_predict(fsp_output, X_test, opt=option)

        assert y_test.tolist() == [1,0,2,1,1,0,1,2,1,1,2,0,0,0,0,1,2,1,1,2]
        assert y_pred.shape == (20,)
        assert y_pred.tolist() == [1,0,2,1,1,0,1,2,2,1,2,0,0,0,0,1,2,1,1,2]
        assert y_pred_Proportion.shape == (20,)
        assert np.mean(y_test == y_pred) == 0.95
        assert y_pred_Proportion.tolist() == [1., 1., 1., 1., 1., 1., 1., 1., 0.7142857142857143 ,1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]

    def test_fsp_dm_case2_dist_sklearn_mt_kmeans_skylearn_mt(self):

        X,y = datasets.load_iris(return_X_y=True)

        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=20, random_state=42)

        option = Options(return_full_history=True,
                        return_full_dm=True,
                        s_parameter=0.3,
                        p_parameter=0.01,
                        kmeans_random_state=0,
                        distance_method=2)

        fsp_output = fsp(X_train, y_train,opt=option)

        assert fsp_output['Mu'] == None
        assert fsp_output['Sigma'] == None
        assert fsp_output['ClassNames'].tolist() == [0,1,2]
        assert fsp_output['opt'] == option
        assert fsp_output['InsampleError'] == 0.023076923076923078
        assert fsp_output['NumIterations'] == 18

        assert len(fsp_output['H']['dm']) == 19

        #TODO: assert fsp_output['H']['dm'][0][0].tolist() == [55.95265107450104, 137.76852954539308, 4.553404433542719]
        #TODO: assert fsp_output['H']['dm'][9][0].tolist() == [0., 0., 3.0939923488322814]
        assert fsp_output['H']['dm'][18][0].tolist() == [0., 0., 0.]

        assert pd.DataFrame(fsp_output['H']).shape == (19, 16)

        y_pred, y_pred_Proportion = fsp_predict(fsp_output, X_test, opt=option)

        assert y_test.tolist() == [1,0,2,1,1,0,1,2,1,1,2,0,0,0,0,1,2,1,1,2]
        assert y_pred.shape == (20,)
        assert y_pred.tolist() == [1,0,2,1,1,0,1,2,2,1,2,0,0,0,0,1,2,1,1,2]
        assert y_pred_Proportion.shape == (20,)
        assert np.mean(y_test == y_pred) == 0.95
        assert y_pred_Proportion.tolist() == [1., 1., 1., 1., 1., 1., 1., 1., 0.7142857142857143 ,1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
