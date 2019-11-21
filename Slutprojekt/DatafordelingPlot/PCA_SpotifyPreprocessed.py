# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:32:41 2019

@author: danie
"""

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


data_csv_path = os.path.abspath(os.path.dirname(sys.argv[0])) + "\SpotifyDBData_preprocessed.csv"
SpotifyDBData = pd.read_csv(data_csv_path, sep=',', header=0)

#%%
def createPCA(data, N_comps = 2):
    pca =  PCA(n_components=N_comps)
    pca.fit(data)
    return pca

YData = SpotifyDBData['genre']
XData = SpotifyDBData.drop(columns=['genre', 'mode', 'time_signature'])

ourPCA = createPCA(XData, N_comps=2)
X_reduced = ourPCA.fit_transform(XData)

UniqueGenres = set(YData)

dataSplitprGenre = {genre: [] for genre in UniqueGenres}

for i, x in enumerate(X_reduced): 
    dataSplitprGenre[YData[i]].append(x)

LastDict = {genre: np.array(dataSplitprGenre[genre]) for genre in UniqueGenres}

#%%
def plot_two_component_genres(genres, sizeOfFig):
    plt.figure(figsize=(sizeOfFig, sizeOfFig))
    for genre in genres:
        plt.scatter(LastDict[genre][:,0], LastDict[genre][:,1], s=2, label=str(genre))
    # plt.axis('equal')
    plt.legend()
    plt.show()
    
plot_two_component_genres(UniqueGenres, 15)
plot_two_component_genres(list(UniqueGenres)[0:5], 15)
plot_two_component_genres(list(UniqueGenres)[6:10], 15)
    
