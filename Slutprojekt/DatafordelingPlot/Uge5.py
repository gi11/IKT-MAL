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

#%%
att3 = ["popularity","duration_ms"]
axs = scatter_matrix(onehotenc[att3], figsize=(20,20), alpha=0.01)

#%%
att4 = ["duration_ms","genre"]
axs = scatter_matrix(spotifyDBData[att4], figsize=(20,20), alpha=0.01)
#%% How about danceability ? Can it be used to predict popularity?
PD = spotifyDBData[['danceability','energy']]
corrcoef = np.corrcoef(PD.T) # obs: rækker=variable, kolonner=samples (modsat normalt..)


#%% ---------------------- REMOVING ALL THE SONGS WITH A 0 POPULARITY RATING --------------------------
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
    ax[row,col].hist(popularity, bins=40, density=True, label='Track Popularity')
    ax[row,col].set_title(title)

# THAT'S MORE LIKE IT ! GREAT CLEANING!

#%% -------------- CHECK THE DURATION FEATURE IF SCALING CAN BE DONE ------------------------------
cols = 4   # How many subplots pr row
width = 15 # Width of figure
prop = 1/3 # Subplot proportions, height/width ratio of subfigures

rows = int(len(UniqueGenres)/cols)+1
height = (rows/cols)*width*prop

fig, ax = plt.subplots(rows, cols, figsize=(width,height))
plt.subplots_adjust(wspace=0.2, hspace=1)
for index, genre in enumerate(UniqueGenres):
    if genre != "A Capella":
        row, col = int(index/cols), index % cols
        genre_tracks = spotifyDBData.loc[spotifyDBData['genre'] == genre]
        duration = genre_tracks['duration_ms']
        title = genre + ", N = " + str(len(duration))
        ax[row,col].hist(duration, bins=40, density=True, label='Track Popularity')
        ax[row,col].set_title(title)
    
        sub_mean = np.mean(duration)
        sub_median = np.median(duration)
        sub_max = np.max(duration)
        print(sub_max)
    
        ax[row,col].axvline(sub_mean, color='b', label = "Mean")
        ax[row,col].axvline(sub_median, color='r', label="Median")
        ax[row,col].axvline(sub_max, color='y', label="Max")
        
        sub_max = 0
    
# Should we just filter out the outliers of duration as well? Could one go about
# it differently, e.g is there to manny to simply assign them to the mean? tendency
# seems to be that duration for each genre has a nice distribution. 
# Would it be beneficial to convert units - e.g from ms to s? Decrease the size
# of the numbers before performing the scaling? 

# It has been decided to drop this feature, as the correlation 
        
        
#%% What should one do about the Key feature? 
def uniqueFeatureLabels(df, fName):
    UniqueLabels = df[fName].unique()  
    print("Unique labels for ", fName,":", len(UniqueLabels))
    
uniqueFeatureLabels(spotifyDBData, "key") # = 12

# Is it feasible to one hot encode this ? This would add an additional 11 features
# - This combined with the genre encoding could cause feature overload? Would our model
# Have enough samples to learn? Should this feature be dropped completely?

# It has been decided to drop this feature - as our samplesize is only 230000, minus the ones
# with 0 popularity rating. For fear of inproper training of the model due to the sheer size
# of the feature list, this feature will be dropped. 

# ---------------- What should one do about mode? ---------------------------------
uniqueFeatureLabels(spotifyDBData, 'mode') # = 2
# Only expands to one addtional feature - should be one hot encoded. 
# Decided to include this. Use one hot encoding. 

#%%
# ------------ What should one do about time_signature feature? -------------------
uniqueFeatureLabels(spotifyDBData, 'time_signature') # = 5
# Is it feasible to encode into 4 addtional features? ... 
# Clean this data - 1/4 and 0/4 should be removed. 
# 0/4 does not exit? Drop this. 
fig, ax = plt.subplots(1, 1, figsize=(width,height))
UniqueTS = spotifyDBData['time_signature'].unique()
N_uTS = []
for TS in UniqueTS:
    inSignature = spotifyDBData[spotifyDBData['time_signature'] == TS]
    N_uTS.append(len(inSignature))
    print("For time signature: ", TS, ", samples: ",  len(inSignature))
    

ax.hist(N_uTS, bins=5, density=True, label='Track Popularity')
    

#%% Last part, when above is done, create a new script without plots that only performs data cleaning and feature scaling
# Let it store the result as a new csv file from which the next step in the pipeline can read from...

# REMEMBER:
# OneHotEncoder CANNOT BE USED FOR Y VALUES - USE LABELBINARIZER INSTEAD!

