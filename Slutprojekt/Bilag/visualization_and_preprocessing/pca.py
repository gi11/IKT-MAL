
#%%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import pandas as pd
from pandas.plotting import scatter_matrix
import sklearn.preprocessing as sklpre
import math
import os
import sys
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot


data_csv_path = os.path.abspath(os.path.dirname(sys.argv[0])) + "\SpotifyDBData_preprocessed_stdscaled.csv"
SpotifyDBData = pd.read_csv(data_csv_path, sep=',', header=0)

#%%
def createPCA(data, N_comps = 2):
    pca =  PCA(n_components=N_comps)
    pca.fit(data)
    return pca

YData = SpotifyDBData['genre']
XData = SpotifyDBData.drop(columns=['genre', 'mode', 'time_signature'])

UniqueGenres = set(YData)

# --------------------------- 2 dim -------------------------------------#
ourPCA_2dim = createPCA(XData, N_comps=2)
X_reduced_2dim = ourPCA_2dim.fit_transform(XData)

dataSplitprGenre = {genre: [] for genre in UniqueGenres}

for i, x in enumerate(X_reduced_2dim): 
    dataSplitprGenre[YData[i]].append(x)

LastDict_2dim = {genre: np.array(dataSplitprGenre[genre]) for genre in UniqueGenres}

# --------------------------- 3 dim -------------------------------------#

ourPCA_3dim = createPCA(XData, N_comps=3)
X_reduced_3dim = ourPCA_3dim.fit_transform(XData)

dataSplitprGenre = {genre: [] for genre in UniqueGenres}

for i, x in enumerate(X_reduced_3dim): 
    dataSplitprGenre[YData[i]].append(x)

LastDict_3dim = {genre: np.array(dataSplitprGenre[genre]) for genre in UniqueGenres}

#%%
def plot_two_component_genres(genres, dict_2dim, sizeOfFig):
    plt.figure(figsize=(sizeOfFig, sizeOfFig))
    for genre in genres:
        plt.scatter(dict_2dim[genre][:,0], dict_2dim[genre][:,1], s=2, label=str(genre))
    plt.axis('equal')
    plt.legend()
    plt.show()
    plt.xlabel("pAxis 1")
    plt.ylabel("pAxis 2")
    
plot_two_component_genres(UniqueGenres,LastDict_2dim, 15)
#plot_two_component_genres(list(UniqueGenres)[0:5], LastDict_2dim, 15)
#plot_two_component_genres(list(UniqueGenres)[6:10], LastDict_2dim, 15)
#plot_two_component_genres(list(UniqueGenres)[11:15], LastDict_2dim, 15)
#plot_two_component_genres(list(UniqueGenres)[16:20],LastDict_2dim, 15)
#plot_two_component_genres(list(UniqueGenres)[21:25], LastDict_2dim, 15)
plot_two_component_genres(['Soundtrack','Classical','Opera'], LastDict_2dim, 15)
    

plot_two_component_genres(['Jazz','Soul'], LastDict_2dim, 15)

plot_two_component_genres(['Dance','Electronic'], LastDict_2dim, 15)
    
#%%
def plot_three_component_genres(genres, dict_3dim, sizeofFig):
    fig = pyplot.figure()
    ax = Axes3D(fig)
    for genre in genres:
        ax.scatter(dict_3dim[genre][:,0], dict_3dim[genre][:,1], dict_3dim[genre][:,2], s=2, label=str(genre))
    ax.set_xlabel("pAxis 1")
    ax.set_ylabel("pAxis 2")
    ax.set_zlabel("pAxis 3")
    pyplot.legend()
    pyplot.show()
    
#plot_three_component_genres(UniqueGenres,LastDict_3dim, 15)
plot_three_component_genres(list(UniqueGenres)[0:5], LastDict_3dim, 15)
#plot_three_component_genres(list(UniqueGenres)[6:10], LastDict_3dim, 15)
#plot_three_component_genres(list(UniqueGenres)[11:15], LastDict_3dim, 15)
#plot_three_component_genres(list(UniqueGenres)[16:20],LastDict_3dim, 15)
#plot_three_component_genres(list(UniqueGenres)[21:25], LastDict_3dim, 15)
