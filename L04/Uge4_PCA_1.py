# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 10:56:47 2019

@author: Gill
"""
#%% Imports and data preparation
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import matplotlib
import matplotlib.pyplot as plt

# Load digit data and split into training and test data
digits = load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#%% Plot single digit
def plot_digit(digit, cmpsize=64):
    s = int(np.sqrt(cmpsize))
    image = digit.reshape(s, s)
    plt.imshow(image, cmap = matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()

digit_index = 123
plot_digit(X[digit_index])
print(y[digit_index])

#%% Create PCAs with different number of components
def create_pca(data, cmps):
    pca = PCA(n_components=cmps)
    pca.fit(data)
    return pca

dimensions = [2, 4, 8, 9, 16, 25, 32]
pca = {n: create_pca(X_train, n) for n in dimensions}

X_reduced = {n: pca[n].fit_transform(X_train) for n in dimensions}


#%% Plot the standard deviation and explained variance pr dimension
def plot_pca_dim_variance(pca):
    varians = pca.explained_variance_
    s = np.sqrt(varians)
    
    plt.plot(s)
    plt.title('Standard afvigelser (spredning) langs de forskellige dimensioner')
    plt.xlabel('Dimension nummer')
    
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_)/np.sum(pca.explained_variance_))
    plt.title('Explained variance - procentdel')
    plt.xlabel('Dimension nummer')

#%% Split data into which digit it "should" represent
def mkdict1(x_reduced): # Method 1
    xdict={i:[] for i in range(10)}
    for i, x in enumerate(x_reduced):
        xdict[y_train[i]].append(x)
    return {i:np.array(xdict[i]) for i in range(10)}

def mkdict2(x_reduced): # Method 2
    return {
        digit: np.array([x for xi, x in enumerate(x_reduced) if y_train[xi] == digit]) 
        for digit in range(10)
    }
 
#%% Time the two different mothods of creating dict

from timeit import timeit
r1 = timeit(lambda: mkdict1(X_reduced[2]), number=100)
r2 = timeit(lambda: mkdict2(X_reduced[2]), number=100)
print(f"mkdict1 = {r1}")
print(f"mkdict2 = {r2}")

#%% Plot the two-component representation of digits and color by 'actual' digit
npdict = mkdict1(X_reduced[2])
def plot_two_component_digits(digits):
    for dnum in digits:
        plt.scatter(npdict[dnum][:,0], npdict[dnum][:,1], s=2, label=str(dnum))
    plt.axis('equal')
    plt.legend()
    plt.show()

plot_two_component_digits([1, 2, 3])
plot_two_component_digits([4, 5, 6])
plot_two_component_digits([7, 8, 9])
plot_two_component_digits([1, 3])
plot_two_component_digits([8, 9])

#%% Recreate / recover original
X_recovered = {n: pca[n].inverse_transform(X_reduced[n]) for n in dimensions}

#%% Plot 16-component PCA representation of digits as 4x4 grid
for i in range(10):
    plot_digit(X_train[i])
    print(y_train[i])
    plot_digit(X_reduced[16][i], 16)

#%%
digit_index = 5

plot_digit(X[digit_index])
print(y[digit_index])
plot_digit(X_recovered[16][digit_index])