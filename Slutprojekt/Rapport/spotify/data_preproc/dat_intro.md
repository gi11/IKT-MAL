# The Spotify Tracks dataset

The dataset for this report is found on kaggle, where the original data has been gathered directly from spotify via their API. The dataset includes 16 features with around ~233 thousand songs. The features included in the dataset can be divided into 2 main categories, namely identification/general features and sound/feel features. The features included in the dataset are as follows:

### Identification and general features: 
- Genre
- Artist name
- Track name
- Track ID 
- Popularity 

### Sound and feel:
- Acousticness  
- Danceability 
- Duration in ms 
- Energy 
- Instrumentalness 
- Key 
- Liveness 
- Loudness 
- Mode 
- Speechiness 
- Tempo 
- Time signature 
- Valence

It should be noted here that the data has already been preprocessed by spotify before we attempt to do anything with it. This can easily be seen from the popularity feature being a scalar, and thus, must be generated from some algorithm translating the yearly/monthly/daily listens into some entity. With that said, some processing for the models must be performed, be that cleaning or scaling as will be seen from the next parts of the report. 

It should be noted that for the classifier there is a fear of the dataset being too small, as with 26 unique genres, 233 thousand samples could in reality be a too small training set for the model to learn. 

!include dat_Visualisation.md

!include dat_Cleaning.md

!include dat_FScaling.md