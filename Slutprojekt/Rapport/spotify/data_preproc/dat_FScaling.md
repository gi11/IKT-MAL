## Feature scaling and PCA

After the data cleaning, the features should be scaled in order to improve the models efficiency. If a model uses the Euclidean Distance between points in the feature space of our datamatrix, the distance is governed by range of our features. As such, a feature with high range will contribute more than those with low range, which will skew the final distance, as the features will no longer contribute equally(which is what we want). This is especially seen if we are to use the K-nearest Neighbours model, as this is exactly based on the Euclidean Distance. 

Scaling should be performed carefully, as stated before with the duration feature, truncation issues and outlier issues must be taken into account. If scaling is done properly, there should not be any downsides in form of performance of the algorithms, as those who works on non-scaled data does not generally suffer from receiving scaled input. Furthermore, in order to further investigate the the tendencies noted from the radarplots, a PCA decomposition into a 2D or 3D plane would be beneficial.

Now, there are several feature scaling methods to choose from, but for the groups project, only standardized scaling has been used:
$$
z = \frac{x-\mu}{\sigma}
$$
The group decided on this scaling method due to the desire to use PCA on the dataset. The advantage for using standarized scaling in comparisson to just normalizing the data is that the PCA maximises the variance of the feature projections onto it's component axes. In order to ensure an equal weight of each feature, min max scaling(normalization) should not be used, as the outlier effect of this scaling method would degrade the performance of the PCA, in the sense that min max scaling ensures no guarantee that the variances of the features is dimensioned equally. 

With the scaling done, the relation tendencies from the radarplots can be further investigated using PCA, plotting both in 2 dimensions and 3: 

![PCA plot using matplotlibs scatter, both in 2D and 3D. Only a sample of the results used is shown here. Both 2D and 3D contains the same samples, but reduced differently (Plots cropped to keep size at acceptable level). _Left_: 2D Plots, _Right_: 3D Plots. ](img/PCA.png){#PCA_plot width=100%}

From the PCA plots, the hypothesis of some genres overlapping more than others is verified, and as such, it must be expected that the classifier will have varying results, being better at distinguishing some genres than others. Some interresting things to note from the plots is the fact that alot of the more "_mainstream_" genres, such as _Rock, Pop, Rap, Hip-Hop_ are all grouped together. Likewise, both _Classical, Opera, Soundtrack_ and especially _Comedy_ are somewhat distinguishable. _Classical, Opera, Soundtrack_ are grouped somewhat together, as they all overlap in one place, which from intuition makes somewhat sense, as _Soundtracks_ often uses _Classic_ pieces, and _Opera_ and _Classical_ often use elements from each other (Mozarts pieces are all classified as classical on spotify, but "_Die Zauberföte_" is a mix of opera and classic). 
