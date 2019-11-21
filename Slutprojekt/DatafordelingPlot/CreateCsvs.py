#%%
import numpy as np
from scipy.stats import norm
import pandas as pd
from pandas.plotting import scatter_matrix
import sklearn.preprocessing as sklpre
import math
import os
import sys

#%% Load data from csv
data_csv_path = os.path.dirname(__file__) + "/SpotifyFeatures.csv"
spotifyDBData = pd.read_csv(data_csv_path, sep=',', header=0)

# print(spotifyDBData)

# %% Remove unused features/columns
remove_cols = ["artist_name", "track_name", "track_id", "duration_ms", "key"]
spotifyDBData = spotifyDBData.drop(columns=remove_cols)

# spotifyDBData.count()

# %% Remove tracks with invalid time signatures

spotifyDBData['time_signature'] = spotifyDBData['time_signature'].astype('str')

spotifyDBData = spotifyDBData[spotifyDBData['time_signature'] !=  "0/4" ]
spotifyDBData = spotifyDBData[spotifyDBData['time_signature'] !=  "1/4" ]

spotifyDBData.count()


#%% Remove A-capella genre

spotifyDBData = spotifyDBData[spotifyDBData.genre != "A Capella"]
spotifyDBData.count()

#%% Apply minmax scaling to audio features

minmaxscaler = sklpre.MinMaxScaler()
audiofeature_cols = ['popularity', 'acousticness', 'danceability', 'energy', 
                 'instrumentalness', 'liveness', 'loudness', 'speechiness', 
                 'tempo', 'valence']

df_scaled = pd.DataFrame(spotifyDBData)
df_scaled[audiofeature_cols] = pd.DataFrame(
        minmaxscaler.fit_transform(df_scaled[audiofeature_cols]), 
        index=df_scaled[audiofeature_cols].index,
        columns=df_scaled[audiofeature_cols].columns)

df_scaled.count()


#%% Apply std scaling to audio features

from sklearn.preprocessing import StandardScaler

stdscaler = sklpre.MinMaxScaler()
df_scaled_std = pd.DataFrame(spotifyDBData)
df_scaled_std[audiofeature_cols] = pd.DataFrame(
        stdscaler.fit_transform(df_scaled_std[audiofeature_cols]), 
        index=df_scaled_std[audiofeature_cols].index,
        columns=df_scaled_std[audiofeature_cols].columns)


df_scaled_std.count()

#%% Remove tracks too unpopular
df_scaled = df_scaled[df_scaled.popularity > 0.01]
df_scaled_std = df_scaled_std[df_scaled_std.popularity > 0.01]


df_scaled.count()
df_scaled_std.count()

#%% One hot encode "genre","mode" and"time_signature" features
onehotenc_all = pd.get_dummies(df_scaled, columns=["genre", "mode", "time_signature"])
onehotenc_mode_tsig = pd.get_dummies(df_scaled, columns=["mode", "time_signature"])

#%%
# Rename/format onehot encoded columns (lowercase, no special symbols)
replacements = {' ':'-', '&':'n', '’':'', '/':'-'}
onehotenc_all.columns = map(lambda s: s.lower().translate(str.maketrans(replacements)), onehotenc_all.columns)
onehotenc_mode_tsig.columns = map(lambda s: s.lower().translate(str.maketrans(replacements)), onehotenc_mode_tsig.columns)
df_scaled.columns = map(lambda s: s.lower().translate(str.maketrans(replacements)), df_scaled.columns)
df_scaled_std.columns = map(lambda s: s.lower().translate(str.maketrans(replacements)), df_scaled_std.columns)
print(list(onehotenc_all.columns)) # Check column names
print(list(onehotenc_mode_tsig.columns)) # Check column names
print(list(df_scaled.columns)) # Check column names

# %%
df_scaled.to_csv(os.path.dirname(__file__) + '/spotifyDBData_preprocessed_minmaxscaled.csv', index=False)
df_scaled_std.to_csv(os.path.dirname(__file__) + '/spotifyDBData_preprocessed_stdscaled.csv', index=False)

onehotenc_all.to_csv(os.path.dirname(__file__) + '/spotifyDBData_preprocessed_onehotenc.csv', index=False)
onehotenc_mode_tsig.to_csv(os.path.dirname(__file__) + '/spotifyDBData_preprocessed_mode-tsig_onehotenc.csv', index=False)

# %%

