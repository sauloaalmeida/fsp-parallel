import sys
sys.path.insert(1, '/home/saulo/fsp-parallel/src')
import inspect
import time
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import RepeatedStratifiedKFold
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
    datasetAbsDirPath = projectRootAbsPath / "test" / "benchmark" / "fsp" / "Datasets" / f"{datasetName}.csv"

    X_y = pd.read_csv(datasetAbsDirPath , header=None).values

    # Adjust class labels if necessary
    if np.min(np.unique(X_y[:, -1].astype(int))) == 1:
        X_y[:, -1] -= 1

    #return loaded Data
    return X_y


def createOption(distanceMethod):
    return Options(Standardize=True, initial_k=1, p_parameter=0.01, h_threshold=1, dm_case=1, dm_threshold=3, update_s_parameter=True, s_parameter=0.15, distance_method=distanceMethod, kmeans_random_state=None)


def fspSerialLeaveOneOutEvaluating(idExec, datasetName="Iris", distanceMethod=1, numRepeats=10):

    #loading data
    X_y = load_data(datasetName)

    #create Options instance
    opt = createOption(distanceMethod=distanceMethod)
    print(opt)

    #train test split for one (the first) element of leave one out cross validation
    crossValidationLiveOneOut = LeaveOneOut()
    for repeat in range(numRepeats):
        for i, (train_indexes, test_indexes) in (enumerate(crossValidationLiveOneOut.split(X_y[:, :-1], X_y[:, -1]))):
            X_train = X_y[train_indexes, :-1]
            X_test = X_y[test_indexes, :-1]
            y_train = X_y[train_indexes, -1].astype(int)
            y_test = X_y[test_indexes, -1].astype(int)

            elipsedTrainingTime, elipsedPredict1Time, elipsedPredict2Time, erroPred1, erroPred2 = fspSingleEvaluate(X_train, y_train, X_test, y_test, opt)
            print(f"{idExec},{numRepeats},{len(X_y)},{repeat*len(X_y)+i},{(repeat*len(X_y)+i)//len(X_y)},{(repeat*len(X_y)+i)%len(X_y)},{datasetName},{distanceMethod},{elipsedTrainingTime},{elipsedPredict1Time},{elipsedPredict2Time},{erroPred1},{erroPred2}")

def fspSerialSingleEvaluating(idExec, datasetName="Iris", distanceMethod=1, numRepeats=10):

    #loading data
    X_y = load_data(datasetName)

    #create Options instance
    opt = createOption(distanceMethod=distanceMethod)
    print(opt)

    for repeat in range(numRepeats):
        #train test split for one (the first) element of leave one out cross validation
        crossValidationLiveOneOut = LeaveOneOut()
        _, (train_indexes, test_indexes) = next(enumerate(crossValidationLiveOneOut.split(X_y[:, :-1], X_y[:, -1])))

        X_train = X_y[train_indexes, :-1]
        X_test = X_y[test_indexes, :-1]
        y_train = X_y[train_indexes, -1].astype(int)
        y_test = X_y[test_indexes, -1].astype(int)

        elipsedTrainingTime, elipsedPredict1Time, elipsedPredict2Time, erroPred1, erroPred2 = fspSingleEvaluate(X_train, y_train, X_test, y_test, opt)
        print(f"{idExec},{numRepeats},{len(X_y)},{repeat},{repeat},1,{datasetName},{distanceMethod},{elipsedTrainingTime},{elipsedPredict1Time},{elipsedPredict2Time},{erroPred1},{erroPred2}")


def fspSerialKFoldEvaluating(idExec, datasetName, distanceMethod, numRepeats, kFoldSize):

    #loading data
    X_y = load_data(datasetName)

    #create Options instance
    opt = createOption(distanceMethod=distanceMethod)
    print(opt)

    #create cross validation strategy
    repeatedStratifiedKFoldCrossValidation = RepeatedStratifiedKFold(n_splits=kFoldSize, n_repeats=numRepeats)

    #iterate over folders spliting, training and predicting
    for i, (train_indexes, test_indexes) in enumerate(repeatedStratifiedKFoldCrossValidation.split(X_y[:, :-1], X_y[:, -1])):
        X_train = X_y[train_indexes, :-1]
        X_test = X_y[test_indexes, :-1]
        y_train = X_y[train_indexes, -1].astype(int)
        y_test = X_y[test_indexes, -1].astype(int)

        elipsedTrainingTime, elipsedPredict1Time, elipsedPredict2Time, erroPred1, erroPred2 = fspSingleEvaluate(X_train, y_train, X_test, y_test, opt)
        print(f"{idExec},{numRepeats},{kFoldSize},{i},{i//kFoldSize},{i%kFoldSize},{datasetName},{distanceMethod},{elipsedTrainingTime},{elipsedPredict1Time},{elipsedPredict2Time},{erroPred1},{erroPred2}")

def main():

    _idExec, _datasetName, _distanceMethod, _numRepeats, _kFoldSize = inputValidation(sys.argv)

    if(_kFoldSize == -1):
        fspSerialSingleEvaluating(_idExec, _datasetName, _distanceMethod, _numRepeats)
    elif (_kFoldSize == 0):
        fspSerialLeaveOneOutEvaluating(_idExec, _datasetName, _distanceMethod, _numRepeats)
    else:
        fspSerialKFoldEvaluating(_idExec, _datasetName, _distanceMethod, _numRepeats, _kFoldSize)

main()
