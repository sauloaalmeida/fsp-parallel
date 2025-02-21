import time
import os
import numpy as np
import pandas as pd
from pathlib import Path
from fsp.options import Options
from fsp.fsp import fsp
from fsp.fsp import fsp_predict

def fspSingleEvaluate(X_train, y_train, X_test, y_test, opt):

    startTrainingTime = time.time()
    mdl = fsp(X=X_train, y=y_train, opt=opt)
    elipsedTrainingTime = time.time() - startTrainingTime

    startPredict1Time = time.time()
    y_pred1, _ = fsp_predict(mdl, X_test, 1)
    elipsedPredict1Time = time.time() - startPredict1Time

    startPredict2Time = time.time()
    y_pred2, _ = fsp_predict(mdl, X_test, 2)
    elipsedPredict2Time = time.time() - startPredict2Time

    return elipsedTrainingTime, elipsedPredict1Time, elipsedPredict2Time,  np.mean(y_test != y_pred1),  np.mean(y_test != y_pred2)


def load_data(datasetName):
    #setup used folders
    projectRootAbsPath = Path('/home/saulo/fsp-parallel')
    datasetAbsDirPath = projectRootAbsPath / "data" / "processed" / f"{datasetName}.csv"

    X_y = pd.read_csv(datasetAbsDirPath , header=None).values

    # Adjust class labels if necessary
    if np.min(np.unique(X_y[:, -1].astype(int))) == 1:
        X_y[:, -1] -= 1

    #return loaded Data
    return X_y


def createOption(distanceMethod):
    return Options(Standardize=True, initial_k=1, p_parameter=0.01, h_threshold=1, dm_case=1, dm_threshold=3, update_s_parameter=True, s_parameter=0.15, distance_method=distanceMethod, kmeans_random_state=None)