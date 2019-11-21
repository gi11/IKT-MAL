## Performance metrics

I order to validate the  trained model's over-all ablity to generalize anyting about the data we analyze it's prediction on the data samples set-aside  for testing. This is done scoring the predictions similar to in-training but instead of a cost function a scoring metric is used. It is desired that the value of this metric will be as high as possible.

### Classifier: F1-score

For the classification problem several possible scoring metrics are availabe; accuracy, precission, recall and f1-score. Validating the model using an accuracy () score wont be enough on its own as accuracy  can be missleading, eg. cases where the classes in the training set is not equally distributed.