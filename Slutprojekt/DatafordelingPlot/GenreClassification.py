#%%

import numpy as np
from scipy.stats import norm
import pandas as pd
from pandas.plotting import scatter_matrix
import sklearn.preprocessing as sklpre
import math
import os
import sys

#%%

data_csv_path = os.path.dirname(__file__) + "/spotifyDBData_preprocessed.csv"
spotifyDBData = pd.read_csv(data_csv_path, sep=',', header=0)


# %%
lb = sklpre.LabelBinarizer()
lb.fit(spotifyDBData['genre'])



# %%
