from sklearn.datasets import load_iris
from fsp.fsp import Fsp
from fsp.options import Options
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from numpy.random import RandomState
import numpy as np
import pandas as pd

def preProccessIrisDataset():
    # Load the iris dataset
    iris_data = load_iris()
    #return iris_data.data[50:,[2,3]], iris_data.target[50:] - 1 #converting classes to 0 based aprouch
    return iris_data.data, iris_data.target


def test_fsp():

    # Define as configurações globais do pandas
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)
    pd.set_option('display.expand_frame_repr', False)

    #instancia o um dataset
    X, y = preProccessIrisDataset()

    #prepara os datasets de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, random_state=42)

    #instancia o fsp
    fsp = Fsp(opt = Options(return_full_history=True,dm_threshold=0.1,p_parameter=0.01,kmeans_random_state=42))

    #treina a o fsp
    model = fsp.fit(X_train,y_train)

    print("\nrmH\n",pd.DataFrame(model['rmH']))
    print('\nHistorico H\n',pd.DataFrame(model['H']))

    #testa o fsp
    y_pred, y_pred_Proportion = fsp.predict_case1(X_test,model['rmH'],X_train,y_train)
    print("y_test =",y_test)
    print("y_pred.shape",y_pred.shape)
    print("\ny_pred\n",y_pred)
    print("\ny_pred_Proportion.shape",y_pred_Proportion.shape)
    print("\ny_pred_Proportion\n",y_pred_Proportion)

    erro = np.mean(y_test != y_pred)
    print("Erro de classificação:", erro)

def test_kmeans():

    #instancia o um dataset
    X, y = preProccessIrisDataset()

    #print(X)
    #print(y)
    rs = RandomState(42)

    kmeans = KMeans(n_clusters=4,max_iter= 1000, random_state=rs,  n_init=1).fit(X)
    idx = kmeans.labels_
    C = kmeans.cluster_centers_

    print(idx)
    print(C)

