## Feature scaling

After the data cleaning, the features should be scaled in order to improve the models efficiency. The improvement is done as the models who use the Euclidean Distance between points in the feature space of our datamatrix, the distance is governed by range of our features. As such, a feature with high range will contribute more than those with low range, which will skew the final distance, as the features will no longer contribute equally(which is what we want). This is especially seen if we are to use the K-nearest Neighbours model, as this is exactly based on the Euclidean Distance. 

Scaling should be performed carefully, as stated before with the duration feature, truncation issues and outlier issues must be taken into account. If scaling is done properly, there should not be any downsides in form of performance of the algorithms, as those who works on non-scaled data does not generally suffer from receiving scaled input. 

Now, the scaling used by the group has been mostly that of min-max scaling, as this seemed like an easy way to scale the data. It should be noted that this method does not center the data. 