# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:23:46 2019

@author: Jacob Hausted
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import pandas as pd
import math
#%%
spotifyDBData = pd.read_csv('C:/Users/Jacob Hausted/ML/EgetProjekt/SpotifyFeatures.csv', sep=',',header=0)

#%%
# Let try and look at popularity - this would make sense if this a normal distribution
trackPopularity = spotifyDBData['popularity']

# Find the statistical properties of the popularity
mu = np.mean(trackPopularity)
sigma = np.std(trackPopularity)
sigma2 = np.var(trackPopularity)
median = np.median(trackPopularity)

# Create an object to plot into
fig, ax = plt.subplots(1, 1, figsize=[10,6])

# Now plot the popularity data to get an idea of it's distribution
ax.hist(trackPopularity,bins=100, density=True, label='Track Popularity') # Normalises the histogram
plt.xlabel('Popularity rating')
plt.ylabel('Normalised Counts')
ax.axvline(mu, color='b', label = "Mean")
ax.axvline(median, color='r', label="Median")

# Try and fit a gausian distribution
xarr = np.linspace(np.max(trackPopularity), np.min(trackPopularity), 500)
ax.plot(xarr, norm.pdf(xarr, mu, sigma), label='True gausian PDF')
ax.legend()


#%%
# All feature names to be used as titles for each subplot
FeatureNames = spotifyDBData.head()
Genres = spotifyDBData['genre'] # Series containing genres of each track
UniqueGenres = Genres.unique()  # Contains the name of each genre included in
                                # DB

for val in UniqueGenres:        # Verify 
    print(val) 
    
# 1. Create sub figure with 4*5 places
fig, ax = plt.subplots(6, 5, figsize=[40,10])
left = 0.06  # the left side of the subplots of the figure
right = 0.98   # the right side of the subplots of the figure
bottom = 0.06  # the bottom of the subplots of the figure
top =  0.94    # the top of the subplots of the figure
wspace = 0.3  # the amount of width reserved for space between subplots,
              # expressed as a fraction of the average axis width
hspace = 0.5  # the amount of height reserved for space between subplots,
              # expressed as a fraction of the average axis height
plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)

plotDict1 = {
    0: 0,
    1: 0, 
    2: 0,
    3: 0,
    4: 0,
    5: 1,
    6: 1,
    7: 1,
    8: 1,
    9: 1,
    10: 2,
    11: 2,
    12: 2,
    13: 2,
    14: 2,
    15: 3,
    16: 3,
    17: 3,
    18: 3,
    19: 3,
    20: 4,
    21: 4,
    22: 4,
    23: 4,
    24: 4,
    25: 5      
}

plotDict2 = {
    0: 0,
    1: 1, 
    2: 2,
    3: 3,
    4: 4,
    5: 0,
    6: 1,
    7: 2,
    8: 3,
    9: 4,
    10: 0,
    11: 1,
    12: 2,
    13: 3,
    14: 4,
    15: 0,
    16: 1,
    17: 2,
    18: 3,
    19: 4,
    20: 0,
    21: 1,
    22: 2,
    23: 3,
    24: 4,
    25: 0      
}

for index, genre in enumerate(UniqueGenres):
    # 2. Make key-value pair with genre as key, popularity as value
    gSpecificTrack = spotifyDBData.loc[spotifyDBData['genre'] == genre]
    popularity = gSpecificTrack['popularity']
    ax[plotDict1[index],plotDict2[index]].hist(popularity, bins=40, density=True, label='Track Popularity')
    #ax[plotDict1[index],plotDict2[index]].set_xlabel('Popularity rating')
    #ax[plotDict1[index],plotDict2[index]].set_ylabel('Normalised Counts')
    title = genre + ", N = " + str(len(popularity))
    ax[plotDict1[index],plotDict2[index]].set_title(title) 

#%% How about danceability ? Can it be used to predict popularity?
PD = spotifyDBData[['danceability','energy']]
corrcoef = np.corrcoef(PD.T) # obs: rækker=variable, kolonner=samples (modsat normalt..)
