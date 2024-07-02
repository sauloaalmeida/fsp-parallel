import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd


def loadData():
    return pd.read_csv('saida1.txt', sep=',', header=None)

def main():
    #Observations
    x = np.array([10,100,1000,10000])

    #features
    y = np.array([10,50,100])

    #create meshgrid
    x, y = np.meshgrid(x, y)

    df = loadData()
    z = df.values[:,-1]
    z = z.reshape(x.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plote os dados
    ax.plot_surface(x, y, z, cmap='viridis')

    # Adicione rótulos aos eixos
    ax.set_xlabel('Observations')
    ax.set_ylabel('Features')
    ax.set_zlabel('Elipsed time')

    # Mostre o gráfico
    plt.show()



main()
