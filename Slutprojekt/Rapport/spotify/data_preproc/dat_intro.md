# The Spotify Tracks dataset

The dataset for this project has been found on kaggle, and contains data gathered directly from spotify's API[^2]. The dataset includes around ~233 thousand songs with 16 features. These features can be divided into two main categories, namely identification/general features and sound/feel features, as follows: 

| |                 |           | |                     |           |                   |           | |
| - | ---           | --        | - | ---               | --        | ---               | --        | - |
| | __General__     |           | | __Sound and feel__  |           |                   |           | |
| | - Genre         | _string_  | | - Acousticness      | _0 - 1_   | - Loudness        | _< 0_ | |
| | - Artist Name   | _string_  | | - Danceability      | _0 - 1_   | - Mode            | _string_  | |
| | - Track Name    | _string_  | | - Duration in ms    | _> 0_     | - Speechiness     | _0 - 1_   | |
| | - Track ID      | _string_  | | - Energy            | _0 - 1_   | - Tempo           | _string_  | |
| | - Popularity    | _0 - 100_ | | - Instrumentalness  | _0 - 1_   | - Time signature  | _string_  | |
| |                 |           | | - Key               | _string_  | - Valence         | _0 - 1_   | |
| |                 |           | | - Liveness          | _0 - 1_   |                   |           | |

It should be noted here that the data has already been processed to some degree by spotify before we attempt to do anything with it. For instance, the popularity feature is always a value between 0 and 100, and thus, must be generated from some algorithm translating the yearly/monthly/daily listens into some entity. With that said, some preprocessing must still be performed, be that cleaning or scaling, before the data is suitable for the models. \newline
It should also be noted that the group has concerns about the size of the dataset, relative to the number of classes. With 26 unique genres, 233 thousand samples might not be sufficient for some models. 

[^2]: https://developer.spotify.com/documentation/web-api/

!include dat_Visualisation.md

<!-- \newpage -->
!include dat_Cleaning.md

!include dat_FScaling.md
