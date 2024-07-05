import pytest
import time
import os
from pathlib import Path
from fsp.model_selection.fspaccuracy import verify_accuracy_fsp

absPath = Path(".").cwd() / "test" / "benchmark" / "fsp"

def test_accuracy_fsp_one_base_one_option():
    result, outputFile = verify_accuracy_fsp(dataset_filenames=["Test1"], opt_labels=["opt1s0"], input_dataset_path=absPath, output_report_path=absPath)
    assert os.path.exists(outputFile)
    assert len(result) == 1
    assert sum(1 for _ in open(outputFile)) == 2
    os.remove(outputFile)
    assert not os.path.exists(outputFile)

def test_accuracy_fsp_one_base_two_options():
    result, outputFile = verify_accuracy_fsp(dataset_filenames=["Test1"], opt_labels=["opt1s0","opt12s1"], input_dataset_path=absPath, output_report_path=absPath)
    assert os.path.exists(outputFile)
    assert len(result) == 2
    assert sum(1 for _ in open(outputFile)) == 3
    os.remove(outputFile)
    assert not os.path.exists(outputFile)

def test_accuracy_fsp_two_base_one_option():
    result, outputFile = verify_accuracy_fsp(dataset_filenames=["Test1","Test2"], opt_labels=["opt1s0"], input_dataset_path=absPath, output_report_path=absPath)
    assert os.path.exists(outputFile)
    assert len(result) == 2
    assert sum(1 for _ in open(outputFile)) == 3
    os.remove(outputFile)
    assert not os.path.exists(outputFile)


def test_accuracy_fsp_two_base_two_optiona():
    result, outputFile = verify_accuracy_fsp(dataset_filenames=["Test1","Test2"], opt_labels=["opt1s0","opt7s0"], input_dataset_path=absPath, output_report_path=absPath)
    assert os.path.exists(outputFile)
    assert len(result) == 4
    assert sum(1 for _ in open(outputFile)) == 5
    os.remove(outputFile)
    assert not os.path.exists(outputFile)