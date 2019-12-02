# Discussion and Conclusion

In this section, the results is discussed and reflected upon. The classification model/pipeline and regression model/pipeline will be discussed seperately. 

## A note on underfitting and overfitting

Before discussing the results of the models chosen in this project, one should stop and ask whether the models are even good? Do they have the means to perform the task? These questions relate to whether the models are underfitted or overfitted - and how can one be sure they are neither? 

If we are with underfitting, it can be hard to tell whether a model has been so. The issue here is that there is a risk that the chosen model does not have a high enough complexity to make a satisfactory "goodness of fit" for the dataset. The issue here lies mainly with bias. Increasing the number of features alleviates this problem as it icreases the complexity. Likewise, overfitting is when the model has too high of a complexity for the dataset. So, there seems to be a midpoint that is "just-right" for model complexity, how does one determine this? 

For underfitting, although not done explicitly for this project, the _"bias-variance-tradeoff"_[^2] could be used. When performing our searches, it is key that we are not only looking at an accuracy score, as this has no relation to under/overfitting. Instead, by using the F1 score(classification), and R2(regression), we are able to somewhat measure the goodness of fit while also tuning the penalizing factors(minimizing variance = reduces overfitting). Furthermore, by using cross-validation we also introduce an early way to identify the overfitting issue as this is an indicator of how well the model will react to unseen data. Lastly, if the models training scores and test scores varies by a big amount, this might be an indicator of overfitting aswell. 

Feature reduction(or the posibility of it) could be done in order to minimize bias, but was unfortunately forgotten to implement in the pipeline. Here, a plot 

[^2]:https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff#k-nearest_neighbors

## Classification

For the classification models, the group is very satisfied with the results, even though a f1 score of 37% and 40% does not seem like much. If compared to the randomcase, or the case of dummy identifier, we would have expected a yield of $\frac{1}{25}$. Furthermore, the plots of the correct guesses pr. genre, supports the theory that was discussed during the PCA analysis - that the some genres, comedy especially, was easier to predict correctly than those more closely grouped together. This theory however gives reason to another hypothesis - genres is inheretently subjective(earlier in the report the group stated that _"Die Zauberflöte"_ was opera, but this is just the groups opinion), and as such, their labels does not give reason to a clearly divided group each. Assuming the beforementionend question true, no classifier should be able to predict genres with certainty, as the truth would be that the same songs could be classified as more than one genre.

Looking at the dataset with the intention of genre classification, it must be noted that a lot of features was found to be uncorrelated to the desired prediction. This was mainly from the fact that the dataset in reality contained a lot of labelled features - some of which was Unique(and therefore had to be removed), and some which had almost 0 correlation with the genres(e.g _"Mode"_). 

## Regression

Since the idea with the dataset originally was to only produce a regressor able to predict the popularity of songs, the project has definitely given the group some experience in regards to the reality of the ML world. Even though no correlation could immeadiately be made by the group in the preprocessing between popularity and other features, the results after the search was no less than suprising. A mean error of around 7,5% was much less than expected. The theory here is that by removing the samples with a popularity of 0 greatly increased the correlation in some way not obvious to the group. 


## General for the project

Looking at the roadmap of getting to the models, the group must conclude that the pipeline is not as automated as first hoped. Likewise, the data processing took a lot longer than originally anticipated, but in turn, yielded a great deal more knowledge of the dataset. Seeing as this is the first time the group is attempting to implement an _"End-to-end ML"_ project, it must be concluded that a lot of first time errors was made - e.g trying to include that dataprocessing part of the pipeline with the visualization just made the code bloated and unreadable. Although not entirely succesfull in the implementation of an automated pipeline, the group is satisfied with the results. 

It should be noted that for the dataset, a lot of samples was removed when sorting out the samples which had a popularity rating of 0. The effect of this is not entirely clear, as one could guess that the(espicially the regressor) would have issues estimating unpopular songs. Likewise, it could be expected that by excluding these samples, we are actually hindering the effectiveness of the classifier, as popularity was seen to be somewhat correlated with the genre. 

The project could definitely be further developed as it holds some aspects which has not been explored by the group in this project. Since the data is gathered from the spotify API, the pipeline would be expected to greatly benefit from an added part the would gather more songs. This would lend itself to the use case that when new songs were added to the spotify platform, the models could be used to predict either their popularity or genre. 

Another aspect is the underlying _"Timbre vectors"_ of the data. From Spotify's API documentation it seems that the features worked upon in this project is extracted from these, which in essence holds the digital signal processing made on the songs when they are added to spotify. This however would require much more work than done for this project, which is why the group early on decided not to explore this aspect, although interresting. 