
# Genre Classification

One of the main goals of setting up the ML-pipeline with this dataset, is to investigate the prospect of predicting a genre, based on the audiofeatures of a track. Again, it must be underlined that this a supervised classification task, as the desired output is one destinct category of a predefined set of possible options.

### Algorithm Selection
Before training can begin, an algorithm, or set of algorithms, must be selected for the job.
The choice of algorithm has been affected by which algorithms the group has had previous experience with, and the primary basis of comparison has been their success at performing similar classification tasks.
Due to the very good results with the K-Nearest Neighbors (KNN) algorithm in the MNIST search quest exercise for O3, this was the initial choice for the genre-classification task.

In addition, the group has chosen a second type of model, Fully Connected Neural Networks, for comparison. This algorithm seems like a good complementary alternative to the K-nearest Neighbors algorithm, primarily due to its flexibility and scalability. 
Since there are more ways to configure a Neural Network, it could also prove more difficult and time-comsuming to find the optimal initial configuration of the network. But this also means that it has a good chance of being able to cover any areas where the K-Nearest Neighbors is insifficient, making it a good fall-back candidate

### Data processing and structure

The input for the classification model consists of all the scaled audiofeatures, along with one-hot encoded "mode" and "time_signature" features. This gives the model a total of 15 features to use a basis for assigning a genre to each input.

The classes themselves are created from the "genre" column of the dataset, which is encoded through a LabelEncoder, assigning an integer to each possible value, replacing the original string. This means that the classes will be represented by integer values ranging from 0-24, and this data will be the y-values for our model.

Before the model is trained, the data is split into training and validation sets. The model is fitted to the training-set, and afterwards the validation-set is used to test how the model responds to previously unknown samples. This is an important part of the process, as it can be used to identify whether or not the models ability to generalize has been been compromised by over-fitting to the training data.
The proportion of the dataset assigned to the testset was selected to be $\approx 0.3$.

## K-Nearest Neighbors

As mentioned, the KNN classifier has been selected as the primary candidate for genre-prediction. The configuration of the model is described below, followed by the results.

### Hyperparameter Search

In order to get the best possible performance, the model is initialized multiple times with different combinations of hyperparameters, so that the best values for each hyperparameter can be found by evaluating the performance of each combination.
The first search performed on the KNN model considers the hyperparameters shown in Table \ref{table_hyperparameters} :


Name                Value  
-------             ------ 
`n_neighbors`       `[3, 4, 5]`
`weights`           `['uniform', 'distance']`
`algorithm`         `['ball_tree', 'kd_tree', 'brute']`
`p`                 `[2, 3, 4]`

Table: KNN hyperparameter-space to search \label{table_hyperparameters}

As the `n_neighbors` was set relatively low in this first search, larger values for this parameter were used in a later search, where the values 10, 20, 50 and 100 were tested for `n_neighbors`. After searching through the selected parameter space, the best hyperparameters were found to be: 

- `n_neighbors=100`
- `weights='uniform'`
- `algorithm='ball_tree'`  
- and `p=2`

These are the final parameters given to the K-Nearest Neigbor model.

### Results

Initializing a model with the found hyperparameters and fitting it to the training data, has yielded the results shown below:

#### Performance metrics

These performance metrics for the KNN model are taken from sklearn's classification report, which contains the following four metrics:

- *Precision* - The fraction of positives that were true positive, defined as $\frac{tp}{tp+fp}$
- *Recall* - The fraction of positive samples the model was able to find, defined as $\frac{tp}{tp + fn}$
- *F1-score* - The weighted mean of *recall* and *precision* (both are equally weighted in this case)
- *Support* - The number of samples in the validation set

In combination, these four metrics give a good picture of the models overall performance.

|    | Precision    | Recall  | F1-score   | Support |
|-----|-----|-----|-----|-----|
|        Accuracy |           |          |     0.38  |   66778 |
|      Macro avg. |      0.38 |     0.38 |     0.37  |   66778 |
|   Weighted avg. |      0.37 |     0.38 |     0.37  |   66778 |


Table: Classification results (summary) \label{table_summary}

| Class/Genre       | Precision    | Recall  | F1-score   | Support |
|-----|-----|-----|-----|-----|
|     Alternative |      0.23 |     0.20 |     0.21  |    2687 |
|           Anime |      0.51 |     0.38 |     0.44  |    2660 |
|           Blues |      0.36 |     0.30 |     0.32  |    2640 |
|Children’s Music |      0.26 |     0.19 |     0.22  |    3289 |
|       Classical |      0.53 |     0.51 |     0.52  |    2423 |
|          Comedy |      0.95 |     0.93 |     0.94  |    2708 |
|         Country |      0.25 |     0.42 |     0.31  |    2563 |
|           Dance |      0.20 |     0.17 |     0.18  |    2595 |
|      Electronic |      0.48 |     0.48 |     0.48  |    2736 |
|            Folk |      0.24 |     0.31 |     0.27  |    2692 |
|         Hip-Hop |      0.27 |     0.40 |     0.32  |    2783 |
|           Indie |      0.16 |     0.11 |     0.13  |    2835 |
|            Jazz |      0.36 |     0.33 |     0.34  |    2813 |
|           Movie |      0.57 |     0.27 |     0.37  |    1609 |
|           Opera |      0.67 |     0.86 |     0.75  |    2389 |
|             Pop |      0.30 |     0.40 |     0.34  |    2861 |
|             R&B |      0.20 |     0.17 |     0.18  |    2718 |
|             Rap |      0.24 |     0.18 |     0.21  |    2793 |
|          Reggae |      0.40 |     0.38 |     0.39  |    2588 |
|       Reggaeton |      0.45 |     0.58 |     0.51  |    2636 |
|            Rock |      0.25 |     0.32 |     0.28  |    2744 |
|             Ska |      0.52 |     0.45 |     0.48  |    2762 |
|            Soul |      0.21 |     0.11 |     0.14  |    2739 |
|      Soundtrack |      0.53 |     0.69 |     0.60  |    2821 |
|           World |      0.37 |     0.36 |     0.36  |    2694 |

Table: Classification results (pr. class/genre) \label{table_genres}

#### Distribution of predictions pr. class

The models performance is examined further by plotting how the models predictions for the test data is distributed for each genre (true and false positives).

![Distribution of correct and incorrect predictions pr genre](img/kneighbors_predictions_bar.png){#kneighbors_results_bar width=100%}

### Discussion