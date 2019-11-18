# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:23:46 2019

@author: Jacob Hausted
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
#%%
data_csv_path = os.path.abspath(os.path.dirname(sys.argv[0])) + "\SpotifyFeatures.csv"
spotifyDBData = pd.read_csv(data_csv_path, sep=',', header=0)

#%%
# Lets look at popularity - this ould be expected to resemble a normal distribution
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
Genres = spotifyDBData['genre'] # Series containing genres of each track
UniqueGenres = Genres.unique()  # Contains the name of each genre included in DB

print(UniqueGenres) # Verify 

cols = 4   # How many subplots pr row
width = 15 # Width of figure
prop = 1/3 # Subplot proportions, height/width ratio of subfigures

rows = int(len(UniqueGenres)/cols)+1
height = (rows/cols)*width*prop

fig, ax = plt.subplots(rows, cols, figsize=(width,height))
plt.subplots_adjust(wspace=0.2, hspace=1)
for index, genre in enumerate(UniqueGenres):
    row, col = int(index/cols), index % cols
    genre_tracks = spotifyDBData.loc[spotifyDBData['genre'] == genre]
    popularity = genre_tracks['popularity']
    title = genre + ", N = " + str(len(popularity))
    ax[row,col].hist(popularity, bins=40, density=True, label='Track Popularity')
    ax[row,col].set_title(title)

#%%
# Apply One Hot Encoding to the genre feature
onehotenc = pd.get_dummies(spotifyDBData, columns=["genre"])

# Rename/format genre column (lowercase, no special symbols)
replacements = {' ':'-', '&':'n', '’':''}
onehotenc.columns = map(lambda s: s.lower().translate(str.maketrans(replacements)), onehotenc.columns)

print(list(onehotenc.columns)) # Check column names



#%% split data pr genre

genres = list(filter(lambda name: "genre_" in name, onehotenc.columns))
audiofeature_cols = ['popularity', 'acousticness', 'danceability', 'energy', 
                 'instrumentalness', 'liveness', 'loudness', 'speechiness', 
                 'tempo', 'valence']       


minmaxscaler = sklpre.MinMaxScaler()

df_scaled = pd.DataFrame(onehotenc)
df_scaled[audiofeature_cols] = pd.DataFrame(
        minmaxscaler.fit_transform(df_scaled[audiofeature_cols]), 
        index=df_scaled[audiofeature_cols].index,
        columns=df_scaled[audiofeature_cols].columns)

#df_genre = {}
#genremean = {}
#genremedian = {}
#for genre in genres:
#    onegenre = df_scaled.loc[df_scaled[genre] == 1]
#    onegenre.reset_index(inplace=True, drop=True)
#    gidx = genre.lower()
#    
#    df_genre[gidx] = onegenre.filter(audiofeature_cols)
#    
#    genremean[gidx] = df_genre[gidx].mean().values.flatten().tolist()
#    genremedian[gidx] = df_genre[gidx].median().values.flatten().tolist()
#    genremin[gidx] = df_genre[gidx].min().values.flatten().tolist()
#    genremax[gidx] = df_genre[gidx].max().values.flatten().tolist()
    
tracks = {genre: df_scaled.loc[df_scaled[genre] == 1]
                    .reset_index(drop=True)
                    .filter(audiofeature_cols)
            for genre in genres}
    
audiofeatures = {genre.replace("genre_",""): tracks[genre].describe()
                 for genre in genres}

# Mean and meadian
afeatures_mean = {genre: tracks[genre].mean().values.flatten().tolist()
                  for genre in genres}
afeatures_median = {genre: tracks[genre].median().values.flatten().tolist()
                    for genre in genres}

#%% 

def radar_subplot(categories, data, title=None, subplotpos=(1,1,1)):
    # Dimension angles
    N = len(categories)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    
    # Create polar subplot
    ax = plt.subplot(subplotpos[0],subplotpos[1],subplotpos[2], polar=True)
    ax.set_rlabel_position(0)
    ax.set_title(title)
    plt.xticks(angles, categories, color='grey', size=10)
    plt.yticks([0.2,0.4,0.6,0.8], ["0.2","0.4","0.6","0.8"], color="grey", size=10)
    plt.ylim(0,1)
    
    # Plot one line/shape pr item in data
    plotdata = data if isinstance(data[0],list) else [data]
    angles += angles[:1]
    for values in plotdata:
        plotvalues = values + values[:1]  # Line finishes same place as it starts
        ax.plot(angles, plotvalues, linewidth=1, linestyle='solid')
        ax.fill(angles, plotvalues, 'b', alpha=0.1)

def plot_audiofeatures(genres, cols=3, width=18, hspace=0.3):
    rows = int(len(genres)/cols)+1
    height = width*(rows/cols)    
    plt.figure(figsize=(width,height))
    plt.subplots_adjust(hspace=hspace)
    for idx, genre in enumerate(genres):
        genre = genre if genre.startswith("genre_") else "genre_" + genre
        categories = list(tracks[genre])
        subplot = (rows, cols, idx+1)
        layers = [afeatures_mean[genre], afeatures_median[genre]]
        title = genre.replace("genre_","").capitalize()
        radar_subplot(categories, layers, title, subplot)
        
plot_audiofeatures(["blues", "classical", "comedy", "country", "electronic", 
                    "folk", "jazz", "opera", "rap", "rock", "reggae", "soul"])

#%% Radar diagrams above works well - what we are actually doing is looking for distinctions between the genres
# Could have been achieved by plotting mean on y, genre on x for each feature.. Tiresome, which is why radar diagrams
# was used

def plot_nonOptimalFeaturePlot(fname, genres, width=10, hspace=0.3):
    plt.figure(figsize=(width, width))
    means = []
    xtickPos = []
    for i, genre in enumerate(genres):
        genre = genre if genre.startswith("genre_") else "genre_" + genre
        findex = audiofeature_cols.index(fname)
        means.append(afeatures_mean[genre][findex])
        xtickPos.append(i)    
    
    plt.bar(xtickPos, means)
    plt.xticks(xtickPos, [genre.capitalize() for genre in genres])
    
features = ["popularity",]

plot_nonOptimalFeaturePlot("popularity", ["blues", "classical", "comedy", "country", "electronic", 
                    "folk", "jazz", "opera", "rap", "rock", "reggae", "soul"])
    
    
#%%
    
attributes = ["tempo", "popularity", "acousticness", "danceability", "energy"]
axs = scatter_matrix(onehotenc[attributes], figsize=(20,20), alpha=0.01)


#%%
att2 = ["popularity","instrumentalness", "liveness", "speechiness", "valence"]
axs = scatter_matrix(onehotenc[att2], figsize=(20,20), alpha=0.01)

#%% How about danceability ? Can it be used to predict popularity?
PD = spotifyDBData[['danceability','energy']]
corrcoef = np.corrcoef(PD.T) # obs: rækker=variable, kolonner=samples (modsat normalt..)


#%%
print("===============================================================================================")
print("================================= PERFORMING DATA CLEANING ====================================")
print("===============================================================================================")

genres.remove("genre_a-capella")
# Removing ALL samples with a popularity of 0
for genre in genres:
    tracks[genre] = tracks[genre][tracks[genre].popularity > 0.01]
    
cols = 4   # How many subplots pr row
width = 15 # Width of figure
prop = 1/3 # Subplot proportions, height/width ratio of subfigures

rows = int(len(UniqueGenres)/cols)+1
height = (rows/cols)*width*prop

fig, ax = plt.subplots(rows, cols, figsize=(width,height))
plt.subplots_adjust(wspace=0.2, hspace=1)
for index, genre in enumerate(genres):
    row, col = int(index/cols), index % cols
    #genre_tracks = spotifyDBData.loc[spotifyDBData['genre'] == genre]
    popularity = tracks[genre]['popularity']
    title = genre + ", N = " + str(len(popularity))
    ax[row,col].hist(popularity, bins=50, density=True, label='Track Popularity')
    ax[row,col].set_title(title)











#
#
#testDF = tracks["genre_classical"]
#print(testDF.size)
#filteredDF = testDF[testDF.popularity != 0]
#print(filteredDF.size)
#trackPopularity = filteredDF['popularity']
#
## Find the statistical properties of the popularity
#mu = np.mean(trackPopularity)
#sigma = np.std(trackPopularity)
#sigma2 = np.var(trackPopularity)
#median = np.median(trackPopularity)
#
## Create an object to plot into
#fig, ax = plt.subplots(1, 1, figsize=[10,6])
#
## Now plot the popularity data to get an idea of it's distribution
#ax.hist(trackPopularity,bins=100, density=True, label='Track Popularity') # Normalises the histogram
#plt.xlabel('Popularity rating')
#plt.ylabel('Normalised Counts')
#ax.axvline(mu, color='b', label = "Mean")
#ax.axvline(median, color='r', label="Median")
#
## Try and fit a gausian distribution
#xarr = np.linspace(np.max(trackPopularity), np.min(trackPopularity), 500)
#ax.plot(xarr, norm.pdf(xarr, mu, sigma), label='True gausian PDF')
#ax.legend()
#
#
#
#
#
#
