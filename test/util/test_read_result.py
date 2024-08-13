from pathlib import Path
import pytest
import os
import numpy as np
import pandas as pd
import fsp.util.readresult as read_result

class TestReadResult:

    def test_read_result_iris(self):
        projectRootAbsPath = Path(".").cwd()
        resultsAbsDirPath = projectRootAbsPath / "test" / "benchmark" / "fsp" / "Results"

        # Define the dataset results file name
        dataset_results_FullFileName = resultsAbsDirPath / f"Iris_fsp_evaluating_accuracy_KFold0.npz"

        print(os.path.exists(dataset_results_FullFileName))

        # Define as configurações globais do pandas
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
        pd.set_option('display.expand_frame_repr', False)

        result = read_result.load_result(dataset_results_FullFileName)
        print(pd.DataFrame(result) )

    def test_read_result_bands(self):
        projectRootAbsPath = Path(".").cwd()
        resultsAbsDirPath = projectRootAbsPath / "test" / "benchmark" / "fsp" / "Results"

        # Define the dataset results file name
        dataset_results_FullFileName = resultsAbsDirPath / f"Bands_fsp_evaluating_accuracy_KFold0.npz"

        print(os.path.exists(dataset_results_FullFileName))

        # Define as configurações globais do pandas
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
        pd.set_option('display.expand_frame_repr', False)

        result = read_result.load_result(dataset_results_FullFileName)
        print(pd.DataFrame(result) )