import pytest
import time
import os
from pathlib import Path
from fsp.options import Options
from fsp.model_selection.fspaccuracy import verify_accuracy_fsp_from_optlabels
from fsp.model_selection.fspaccuracy import verify_accuracy_fsp


absPath = Path(".").cwd() / "test" / "benchmark" / "fsp"


def test_accuracy_fsp():
    result, outputFile = verify_accuracy_fsp(dataset_filenames=["Iris"], opts={"deterministicResult":Options(kmeans_random_state=42)}, input_dataset_path=absPath, output_report_path=absPath)
    assert os.path.exists(outputFile)
    assert len(result) == 1
    assert sum(1 for _ in open(outputFile)) == 2

    #testing result values
    assert result[0]['DatasetFileName'] == 'Iris'
    assert result[0]['optLabel'] == 'deterministicResult'
    assert result[0]['MeanStd_accuracies_predict1'].tolist() == [96.66666666666666, 1.1102230246251565e-14]
    assert result[0]['MeanStd_accuracies_predict2'].tolist() == [94.66666666666667, 0.]
    assert result[0]['KFold'] == 150
    assert result[0]['NumberOfRuns'] == 10
    assert result[0]['accuracies_predict1'] == [0.9666666666666667, 0.9666666666666667, 0.9666666666666667, 0.9666666666666667, 0.9666666666666667, 0.9666666666666667, 0.9666666666666667, 0.9666666666666667, 0.9666666666666667, 0.9666666666666667]
    assert result[0]['accuracies_predict2'] == [0.9466666666666667, 0.9466666666666667, 0.9466666666666667, 0.9466666666666667, 0.9466666666666667, 0.9466666666666667, 0.9466666666666667, 0.9466666666666667, 0.9466666666666667, 0.9466666666666667]

    os.remove(outputFile)
    assert not os.path.exists(outputFile)

def test_accuracy_fsp_one_base_one_option():
    result, outputFile = verify_accuracy_fsp_from_optlabels(dataset_filenames=["Test1"], opt_labels=["opt1s0"], input_dataset_path=absPath, output_report_path=absPath)
    assert os.path.exists(outputFile)
    assert len(result) == 1
    assert sum(1 for _ in open(outputFile)) == 2
    os.remove(outputFile)
    assert not os.path.exists(outputFile)

def test_accuracy_fsp_one_base_two_options():
    result, outputFile = verify_accuracy_fsp_from_optlabels(dataset_filenames=["Test1"], opt_labels=["opt1s0","opt12s1"], input_dataset_path=absPath, output_report_path=absPath)
    assert os.path.exists(outputFile)
    assert len(result) == 2
    assert sum(1 for _ in open(outputFile)) == 3
    os.remove(outputFile)
    assert not os.path.exists(outputFile)

def test_accuracy_fsp_two_base_one_option():
    result, outputFile = verify_accuracy_fsp_from_optlabels(dataset_filenames=["Test1","Test2"], opt_labels=["opt1s0"], input_dataset_path=absPath, output_report_path=absPath)
    assert os.path.exists(outputFile)
    assert len(result) == 2
    assert sum(1 for _ in open(outputFile)) == 3
    os.remove(outputFile)
    assert not os.path.exists(outputFile)


def test_accuracy_fsp_two_base_two_options():
    result, outputFile = verify_accuracy_fsp_from_optlabels(dataset_filenames=["Test1","Test2"], opt_labels=["opt1s0","opt7s0"], input_dataset_path=absPath, output_report_path=absPath)
    assert os.path.exists(outputFile)
    assert len(result) == 4
    assert sum(1 for _ in open(outputFile)) == 5
    os.remove(outputFile)
    assert not os.path.exists(outputFile)