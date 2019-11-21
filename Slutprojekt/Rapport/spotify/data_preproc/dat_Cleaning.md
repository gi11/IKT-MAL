## Cleaning the data

From the visualisation of the data, a few key points was seen, in regards to the data: 

- The popularity feature contains a significant amount of samples with a rating of 0.
- The "A Capella" genre only contains 119 samples
- The duration feature is in a too high range, and contains significant outliers
- Both the genre, key, time signature and mode are all labeled traits

Since the group has no idea what causes a popularity rating of 0 - is it a new song without any data yet? Is there a threshold of listens/checks that causes the popularity rating? Is it just unpopuler with no interrest? - it has been decided to drop the samples with this rating from the dataset. This does however cause some implications, as our models ability to predict unpopular songs will be severely hindered. Futhermore, dropping samples from the groups already somewhat small dataset is not desireable as it also hinders the models ability to train and gather information. 
The "A Capella" genre does not contain enough samples to provide valuable information about the genre itself, no model will be able to learn from 119 samples, and certainly not distinguish between other genres from it(outliers will have too much effect as no real distribution can be made). 

The duration feature was difficult to tackle, as one could no simply scale it to a range of 0 - 1, since the outliers would cause too much of a skew in the range and thus, the feature would face truncation issues. One possible solution the group though of would be to scale the unit, either to second and minutes as this would leviate the issue of the huge range found in the feature. This, however, did not help the outlier issue. To overcome the outlier issue, a manual boundry should be set in place, which would discard the outliers, and remove the from the dataset. The boundry could be found from analysis of the distribution plots, but again, this solution would cause further reduction in the dataset. Therefore, since the correlation to popularity was already low, and the distribution mean and meadians across genres was so similar, the feature is removed. 

For the labeled features, some encoding must be performed. For this, it has been decided to use one hot encoding. This again does not come without a price, as this effectively increases the amount of features used in the models, and thus, it must be expected that the size of the dataset should be sufficient to still be able to learn properly. Thus, looking at just the genre, we are looking to expand the dataset with an addiational 24 features - this is problematic. The mode exapands to 2 features, and time_signature to 3(the time signatures found in the dataset was 0/4, 1/4, 3/4, 4/4, 5/4 - 0/4 and 1/4 was decided to add to 4/4, as there really is no _music_ in either of those). Key would expand to an additonal 12, and there was no found any real correlation, thus the feature was decided to be dropped. The rest was encoded. __OBS__: The genre has only been one hot encoded for the regressor, as alot of scikit learns classifiers contains _labelBinarizer()_[^1] inside them - thus, they do not expect their y-values to encoded beforehand. In the case of them expecting it to,  _labelBinarizer()_ should be used for in order to satisfy the the models of scikit learn. 

Performing the above cleaning, the popularity across the genres can be plotted again to verify that the 0 rated samples has indeed been filtered out: 




[^1]:https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html#sklearn.preprocessing.LabelBinarizer