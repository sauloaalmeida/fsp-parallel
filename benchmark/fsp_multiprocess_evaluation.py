import sys
import torch
sys.path.insert(1, '/home/saulo/fsp-parallel/src')
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
import util.evaluation_util as evaluation_util
from fsp.options import Options
from fsp.fsp import fsp
from fsp.fsp import fsp_predict


def inputValidation(args):

    _kFoldSize = 10
    _datasetName = 'ElectricalFaultDetection'
    _numRepeats = 10
    _numProccess = 32

    print(args)

    if (len(args) <= 2 or len(args) > 6):
        raise Exception("Execution call must be between 2 and 5 arguments:"
                        "\n1 - ExecutionId (string),"
                        "\n2 - DatasetName (String),"
                        "\n3 - NumberRepeats - Optional (Int) - Without parameter: 10, n - Number of repeats"
                        "\n4 - KFoldSize - Optional (Int) - Without parameter: 10, n - Number of folds (must be at least 2)."
                        "\n5 - Number of Proccess will be used"
                       )

    _idExec = args[1]
    _datasetName = args[2]

    #validating number of repeats (if exists)
    if(len(args) > 3):
        try:
            _numRepeats = int(args[3])
        except ValueError:
            raise Exception("Error validating the number of repeats. If informed, number of repeats must be an integer.")

        if(_numRepeats < 1):
            raise ValueError("Error validating the number of repeats. Number of repeats must be at least 1.")

    #validating kfold size (if exists)
    if(len(args) > 4):
        try:
            _kFoldSize = int(args[4])
        except ValueError:
            raise Exception("Error validating the number of Folds. If informed, number of folds must be an integer.")

        if(_kFoldSize < 2):
            raise ValueError("Error validating the number of Folds. Minimum fold size is 2 for KFold cross validation.")


    #validating numProccess size (if exists)
    if(len(args) > 5):
        try:
            _numProccess = int(args[5])
        except ValueError:
            raise Exception("Error validating the number of proccess. If informed, number of proccess must be an integer.")

        if(_numProccess < 2):
            raise ValueError("Error validating the number of proccess. Minimum proccess amount is 2 for parallel execution.")


    return _idExec, _datasetName, _numRepeats, _kFoldSize, _numProccess

def fspMultiproccessKFoldEvaluating(_idExec, _datasetName, _numRepeats, _kFoldSize, _numProccess):

    #Set number of procces
    torch.set_num_threads(_numProccess)

    #loading data
    X_y = evaluation_util.load_data(_datasetName)

    #create Options instance
    opt = evaluation_util.createOption(distanceMethod=3)
    print(opt)

    #create cross validation strategy
    repeatedStratifiedKFoldCrossValidation = RepeatedStratifiedKFold(n_splits=_kFoldSize, n_repeats=_numRepeats)

    #iterate over folders spliting, training and predicting
    for i, (train_indexes, test_indexes) in enumerate(repeatedStratifiedKFoldCrossValidation.split(X_y[:, :-1], X_y[:, -1])):
        X_train = X_y[train_indexes, :-1]
        X_test = X_y[test_indexes, :-1]
        y_train = X_y[train_indexes, -1].astype(int)
        y_test = X_y[test_indexes, -1].astype(int)

        elipsedTrainingTime, elipsedPredict1Time, elipsedPredict2Time, erroPred1, erroPred2 = evaluation_util.fspSingleEvaluate(X_train, y_train, X_test, y_test, opt)
        print(f"{_idExec},{_numRepeats},{_kFoldSize},{i},{i//_kFoldSize},{i%_kFoldSize},{_datasetName},{_numProccess},{elipsedTrainingTime},{elipsedPredict1Time},{elipsedPredict2Time},{erroPred1},{erroPred2}")


def main():

    _idExec, _datasetName, _numRepeats, _kFoldSize, _numProccess = inputValidation(sys.argv)

    print(f'id={_idExec}, database={_datasetName}, repeats={_numRepeats}, kfold={_kFoldSize}, proccess={_numProccess}')

    fspMultiproccessKFoldEvaluating(_idExec, _datasetName, _numRepeats, _kFoldSize, _numProccess)

main()
