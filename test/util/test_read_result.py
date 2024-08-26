from pathlib import Path
import pytest
import os
import numpy as np
import pandas as pd
import fsp.util.readresult as read_result
import matplotlib
import matplotlib.pyplot as plt

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
        dataset_results_FullFileName = resultsAbsDirPath / f"Iris_fsp_evaluating_accuracy_KFold0.npz"

        print(os.path.exists(dataset_results_FullFileName))

        # Define as configurações globais do pandas
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
        pd.set_option('display.expand_frame_repr', False)

        optionsList = []
        resultData = np.zeros((64,5))

        results = read_result.load_result(dataset_results_FullFileName)

        for result in results:

            #descarta os resultados de kemeans diferentes do sklearn
            if result['opt'].kmeans_method != 2:
                continue

            #Verifica qual os parametros do opt
            #se nao existir, adiciona o opt na lista de existentes
            if result['opt'].getParamsDetails() not in optionsList:
                optionsList.append(result['opt'].getParamsDetails())

            # retorna sua posicao
            optIndexUsed = optionsList.index(result['opt'].getParamsDetails())

            #adiciona o tempo gasto na matriz de resposta
            #usando o metodo de distancia e os parametros do opt
            resultData[optIndexUsed, (result['opt'].distance_method -1)] = result['ElapsedTime']

        print(resultData)

        xpoints = np.arange(1, 65)

        for i in range(5):
            ypoints = resultData[:,i]
            plt.plot(xpoints, ypoints, label=str(i))

        yMatlab = np.array([4.1950,0.6186,0.6171,2.6362,3.6240,1.0864,1.8872,3.4870,2.3945,0.5943,0.4441,1.2614,3.5554,1.0153,1.7052,4.9829,1.5267,0.5766,0.4378,1.2884,3.0807,1.1758,1.5773,3.5714,1.4992,0.5400,0.4785,1.1526,2.6467,1.1420,1.5789,2.9916,2.2579,0.5723,0.4282,1.1556,3.4819,1.0387,1.4993,2.9973,2.3080,0.5908,0.4368,1.2190,3.3932,1.1174,1.6733,2.7902,1.4275,0.5783,0.4429,1.1155,2.6563,1.0196,1.5101,3.0122,1.4470,0.5542,0.4324,1.1940,2.3993,1.0322,1.6697,3.1024])
        plt.plot(xpoints, yMatlab, label="Matlab")

        plt.legend()
        plt.show()


