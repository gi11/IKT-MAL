## End-to-End Machine Learning study process

Gennerally the chapter outlines the following key steps in developing such a model:

- __Getting the Data and looking at the Big Picture:__
    Defining the problem, we wish to apply a ML model to solve, finding and analyzing the dataset to get an overview and initial insight into any underlying patterns and relationships between features in the data set. 

- __Prepare the Data for Machine Learning Algorithms:__
    From the data analysis the necessary data cleaning and feature scaling are planned and performed to prepare it for the selected model. Most ML models perform poorly on accidental null-data and features with a high variance. To this end we can drop bad samples or features and apply normalization or standardization to sanitize the data.

- __Selecting, Training and Fine-Tuning a Model:__
    When the data has been prepared, it is time to train a appropiate model and validate the result. This is done utilizing a iterative process where the model is optimized according to a cost function and then its validated using a score metric.

- __Launching, Monitoring and Maintaining the System:__
    This is the eventual goal of the entire process. When the model has been optimized then it time to launch it. Hopefully it will perform well and make accuarate predictions. However it is important to keep monitoring and maintaining the model as it generally tends to rot over time as the input data might change and therefore retraining on fresh data might be necessary.