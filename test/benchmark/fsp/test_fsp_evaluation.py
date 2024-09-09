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
    fsp_evaluating_accuracy(dataset_name=["Iris"], opt=[Options(),Options(distance_method=3),Options(distance_method=4)], KFold=0, NumberOfRuns=10)

def test_benchmark1_fsp_evaluating_accuracy():
    fsp_evaluating_accuracy(dataset_name=None, opt=Options.list1(), KFold=0, NumberOfRuns=10)

def test_benchmark2_fsp_evaluating_accuracy():
    opt1 = Options(Standardize=True, initial_k=45, p_parameter=0.01, h_threshold=4, dm_case=1, dm_threshold=3, update_s_parameter=True, s_parameter=0.15, distance_method=1)
    opt2 = Options(Standardize=True, initial_k=45, p_parameter=0.01, h_threshold=4, dm_case=1, dm_threshold=3, update_s_parameter=True, s_parameter=0.15, distance_method=3)
    opt3 = Options(Standardize=True, initial_k=45, p_parameter=0.01, h_threshold=4, dm_case=1, dm_threshold=3, update_s_parameter=True, s_parameter=0.15, distance_method=4)

    fsp_evaluating_accuracy(dataset_name=["ElectricalFaultDetection"], opt=[opt1, opt2, opt3], KFold=0, NumberOfRuns=10)

def test_benchmark3_fsp_evaluating_accuracy():
    opt1 = Options(Standardize=True, initial_k=45, p_parameter=0.01, h_threshold=4, dm_case=1, dm_threshold=3, update_s_parameter=True, s_parameter=0.15, distance_method=1)
    fsp_evaluating_accuracy(dataset_name=["Iris"], opt=[opt1, opt2, opt3], KFold=0, NumberOfRuns=1)