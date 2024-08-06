import pytest
import time
from pathlib import Path
from fsp.options import Options
from fsp.evaluate.hyperparam_optimization import fsp_HyperparameterTuning_skopt
from fsp.evaluate.fspaccuracy import fsp_evaluating_accuracy
import joblib

def testVersionJoblib():
    print('Joblib Lib Version:', joblib.__version__)

def test_hyperparam_optimization():
    fsp_HyperparameterTuning_skopt("Iris")

def test_fsp_evaluating_accuracy():
    fsp_evaluating_accuracy(dataset_name="Iris", opt=[Options(),Options(distance_method=3),Options(distance_method=4)], KFold=0, NumberOfRuns=10)