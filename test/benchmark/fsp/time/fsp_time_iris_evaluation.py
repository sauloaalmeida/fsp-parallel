import sys
sys.path.insert(1, '/home/saulo/workspace/projetos-python/fsp-python-gpu/src')
import inspect
import time
import os
import numpy as np
import pandas as pd
from pathlib import Path
from fsp.options import Options
from fsp.evaluate.fspaccuracy import fsp_evaluating_accuracy

def load_iris_data():
    #setup used folders
    projectRootAbsPath = Path('/home/saulo/workspace/projetos-python/fsp-python-gpu')
    datasetAbsDirPath = projectRootAbsPath / "test" / "benchmark" / "fsp" / "Datasets" / "Iris.csv"

    X_y = pd.read_csv(datasetAbsDirPath , header=None).values
    X_y = X_y[:,2:5]

    # Adjust class labels if necessary
    if np.min(np.unique(X_y[:, -1].astype(int))) == 1:
        X_y[:, -1] -= 1

    #return loaded Data
    return X_y


def createOption(distanceMethod):

    return Options(Standardize=False, initial_k=1, p_parameter=0.01, h_threshold=1, dm_case=2, dm_threshold=3, update_s_parameter=True, s_parameter=0.1, distance_method=distanceMethod, kmeans_random_state=None)


def fspSerialLeaveOneOutTimeEvaluating():

    #loading data
    X_y = load_iris_data()

    return fsp_evaluating_accuracy(dataset_name=X_y,
                                   opt=[createOption(distanceMethod=1),
                                        createOption(distanceMethod=3),
                                        createOption(distanceMethod=4)],
                                   KFold=0,
                                   NumberOfRuns=100)


def main():

    results = fspSerialLeaveOneOutTimeEvaluating()

    for result in results:
        print(f'Media  predict1={result["MeanStd_accuracies_predict1"]}, Media  predict2={result["MeanStd_accuracies_predict2"]}')

main()
