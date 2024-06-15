from sklearn.datasets import load_iris
from fsp.fsp import Fsp
import pandas as pd


def preProccessIrisDataset():
    # Load the iris dataset
    iris_data = load_iris()
    return iris_data.data[50:,[2,3]], iris_data.target[50:] - 1 #converting classes to 0 based aprouch

def test_fsp():

    #instancia o um dataset
    X, y = preProccessIrisDataset()

    #instancia o algoritmo
    fsp = Fsp(X,y)

    #treina a o fsp
    result = fsp.fit()

    #Exibe em formato pandas
    T = pd.DataFrame(result['rmH'])
    print(T)
