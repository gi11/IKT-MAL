## Visualising the data

As a first step in preprocessing the data, sufficient knowledge of the dataset and its features should be established. A special focus will be put on popularity and genre, since those are the features ($\mathbf{y_{true}}$) to be predicted by the models. 

### Feature distributions

Starting off with the popularity, a histogram is made using matplotlib to gain insight of its distribution. 

![Histogram of the popularity feature, for the whole dataset](img/popularity_hist.png){#popularity_hist width=48%}

It can be seen that the distribution resembles a normal distribution, but deviates significantly in some places. The most notable difference being the big spike in the first bin (a popularity rating of 0), which could have several explanations. It might be that there are simply just a lot of unpopular songs on the plaform, as the data would seem to indicate. Another explanation could be that the popularity score is not calculated by spotify before some criteria is met, e.g. a track has been on the platform for period of time. It depends on how and when the score is computed, when the tracks were put on the platform, etc - all information that the group does not currently have.

Continuing, since the dataset included multiple genres, an assumption of each genre having it's own distribution seems somewhat probable, and as such, the popularity across each genre is plotted. 

![Histograms of popularity for each genre](img/popularity_pr_genre.png){#popularity_pr_genre width=85%}

From Figure \ref{popularity_pr_genre}, it is seen that the assumption holds somewhat true. Furthermore, a few key insights is gained, namely that the "A Capella" genre only has 119 samples, and only a few genres contains the outliers with a popularity rating of 0, which was seen in the general histogram. Indirectly, seeing that the feature distribution is somewhat different across genres, this could be examined further. Here, one could examine the means/variances across genres of different features and plotting these against each other, with an example below of the mean of the popularity for each genre.

![Popularity mean for each genre](img/mean_Pop_Genres_wide_all.png){#popularity_mean_pr_genre width=67%}

This process however, would get rather tiresome to do for each feature. As a better alternative, a radar plot is used instead for each genre, with each plot containing multiple feature means. This type of plot has the advantage of providing a more comprehensive overview of the differences and similarities across genres. It should be noted here, that the data has been scaled to value between 0 and 1, which is also done in order to improve our models efficiency later. The radar plots are shown below in Figure \ref{radar_plot} 

![Mean and Median values for each numerical feature, pr genre visualised on radar plots](img/genre_radar.png){#radar_plot width=80%}

From Figure \ref{radar_plot}, a distinction between the shapes of the radarplots can be seen, which is good, because in order for classifier to distinguish between genres, their features should differ in some way. There are however some issues, which can be seen if comparing for example Rock, Raggae and Country. The shape in the radar plot of these genres is very similar, and as such, it must be expected that trying to classify tracks included in these genres will be harder. This thought will be expanded upon later in the report, by applying principal component analysis (PCA) to the data.

### Correlation between features

Now, in moving back to the regressor to be used for predicting popularity, the correlations between popularity and other features should be analysed. Turning the beforementioned analysis around, where popularity was argued to be somewhat related to genre, the genre feature can be seen as an important feature to predict the popularity from - this leaves the rest of the features to be analysed. To easily visualize the results of these correlations, scikit learn's `scatter_matrix()` is used. Using only some of the features here as an example, the results of running the `scatter_matrix()` is seen below: 

![Scatter matrix of the popularity, tempo, acousticness, danceability and energy feature](img/ScatterMat1.png){#scatter_matrix width=67%}

Looking at Figure \ref{scatter_matrix}, the popularity does not seem to be correlated much with the other features, which is unfortunate. This result leaves the expectation for the regressor rather low - if there is no correlation, no mathmatical connection between the two can be made, and thus no predictions can be made. However, this method is limited to comparing two of the feature dimensions directly with each other, showing that one of the features cannot reliably predict the value of another. But some combination of multiple features might provide a better basis for predictions, and it is hoped that a regressor might uncover these if they exist.
