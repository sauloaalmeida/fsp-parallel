import pytest
import time
from pathlib import Path
from fsp.options import Options
from fsp.evaluate.hyperparam_optimization import fsp_HyperparameterTuning_skopt
import joblib

def testVersionJoblib():
    print('Joblib Lib Version:', joblib.__version__)

def test_hyperparam_optimization():
    fsp_HyperparameterTuning_skopt("Iris")