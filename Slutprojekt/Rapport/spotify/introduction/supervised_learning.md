## Supervised learning explained

In this project, a supervised learning approach is used to train the selected models, see Figure \ref{map_supervised} for a detailed map-view af this approach.

![Map-overview of model-fitting using supervised learning](img/ml_supervised_map.png){#map_supervised width=78%}

The collection of data samples containing all the information on all the different features denoted $\mathbf{X}$, is one of two components defining the entire set. The other denoted $\mathbf{y_{true}}$ is a list of true values, one corresponding to each sample in $\mathbf{X}$. $\mathbf{y_{true}}$ can either a be a label or an numerical value depending on whether the problem is classification or regression.

When the data have been aquired and preprocessed, it is split into two sets; one for training the model and another for performance validation. The components in these sets are denoted $\mathbf{X_{train}}$, $\mathbf{y_{train}}$, $\mathbf{X_{true}}$ and $\mathbf{y_{true}}$ for the training and validation (test) set respectively. Splitting the data into two sets is an important step, because in order for the validation to be meaningful, the model has to be applied to data it has not parsed before. This can also help to identify whether or not the models ability to generalize has been compromised by over-fitting to the training data. \newline
<!-- This is the only way the models ablity to generalize and make predictions can be analyzed meaningfully. -->
<!-- This is an important part of the process, as it can be used to identify whether or not the models ability to generalize has been been compromised by over-fitting to the training data. -->
The next step after splitting the data is running the training loop. Here the model ($\mathbf{h}$) is fed the data and then tasked to make predictions ($\mathbf{y_{pred}}$) on each of the training samples. It then evaluates this result compared to the samples $\mathbf{y_{true}}$ using a cost function ($\mathbf{J}$). Often this function is chosen to be the least square solution. Training the model is an iterative process where the goal is to minimize the cost function for the entire set of training samples. In each iteration of the training loop the model parameters ($\mathbf{\theta}$) are tweaked, until the cost function reaches its minimum and the ideal model parameters have been found.

When the training process is done, the model is tasked with making predictions on the test set (unseen samples). These predictions are compared to $\mathbf{y_{true}}$ but instead of applying a cost function, a score metric is used instead to get a meaningful representation of how well the model performs. The goal is to reach as high a score as possible in contrast to the finding a minimum for the cost function.
