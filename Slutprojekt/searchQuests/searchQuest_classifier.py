#%%

from time import time
import numpy as np

from sklearn import svm
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn import datasets

import sys
import os
# sys.path.append("../../include/")
sys.path.append(os.path.dirname(__file__) + "/../../include/")

#%%
from libitmal import dataloaders_v3 as itmaldataloaders

#%%
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

data_csv_path = os.path.dirname(__file__) + "\\..\\DatafordelingPlot\\spotifyDBData_preprocessed.csv"
spotifyDBData = pd.read_csv(data_csv_path, sep=',', header=0)
input_features = [
    # "popularity",
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence",
    # "mode",
    # "time_signature",
]

def getSpotifyBinarizedXY():
    X = np.array(spotifyDBData[input_features])

    lb = LabelBinarizer()
    # lb.fit(np.array(spotifyDBData['genre']))
    y = lb.fit_transform(np.array(spotifyDBData['genre']))

    print("Binarized Label to the following classes")
    print(lb.classes_)
    print(lb.classes_.shape[0])

    return X, y

def getSpotifyIntLabeledXY():
    X = np.array(spotifyDBData[input_features])

    label_encoder = LabelEncoder() 
    y = label_encoder.fit_transform(spotifyDBData['genre']) 
    # print(y.unique())

    return X, y


#%%
currmode="N/A" # GLOBAL var!

def SearchReport(model): 
    
    def GetBestModelCTOR(model, best_params):
        def GetParams(best_params):
            r=""          
            for key in sorted(best_params):
                value = best_params[key]
                t = "'" if str(type(value))=="<class 'str'>" else ""
                if len(r)>0:
                    r += ','
                r += f'{key}={t}{value}{t}'  
            return r            
        try:
            p = GetParams(best_params)
            return type(model).__name__ + '(' + p + ')' 
        except:
            return "N/A(1)"
        
    print("\nBest model set found on train set:")
    print()
    print(f"\tbest parameters={model.best_params_}")
    print(f"\tbest '{model.scoring}' score={model.best_score_}")
    print(f"\tbest index={model.best_index_}")
    print()
    print(f"Best estimator CTOR:")
    print(f"\t{model.best_estimator_}")
    print()
    try:
        print(f"Grid scores ('{model.scoring}') on development set:")
        means = model.cv_results_['mean_test_score']
        stds  = model.cv_results_['std_test_score']
        i=0
        for mean, std, params in zip(means, stds, model.cv_results_['params']):
            print("\t[%2d]: %0.3f (+/-%0.03f) for %r" % (i, mean, std * 2, params))
            i += 1
    except:
        print("WARNING: the random search do not provide means/stds")
    
    global currmode                
    assert "f1_micro"==str(model.scoring), f"come on, we need to fix the scoring to be able to compare model-fits! Your scoreing={str(model.scoring)}...remember to add scoring='f1_micro' to the search"   
    return f"best: dat={currmode}, score={model.best_score_:0.5f}, model={GetBestModelCTOR(model.estimator,model.best_params_)}", model.best_estimator_ 

def ClassificationReport(model, X_test, y_test, target_names=None):
    assert X_test.shape[0]==y_test.shape[0]
    print("\nDetailed classification report:")
    print("\tThe model is trained on the full development set.")
    print("\tThe scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, model.predict(X_test)                 
    print(classification_report(y_true, y_pred, target_names))
    print()
    
def FullReport(model, X_test, y_test, t):
    print(f"SEARCH TIME: {t:0.2f} sec")
    beststr, bestmodel = SearchReport(model)
    #ClassificationReport(model, X_test, y_test)    
    print(f"CTOR for best model: {bestmodel}\n")
    print(f"{beststr}\n")
    return beststr, bestmodel
    
def LoadAndSetupData(mode, test_size=0.3):
    assert test_size>=0.0 and test_size<=1.0
    
    def ShapeToString(Z):
        n = Z.ndim
        s = "("
        for i in range(n):
            s += f"{Z.shape[i]:5d}"
            if i+1!=n:
                s += ";"
        return s+")"

    global currmode
    currmode=mode
    print(f"DATA: {currmode}..")
    
    if mode=='moon':
        X, y = itmaldataloaders.MOON_GetDataSet(n_samples=5000, noise=0.2)
        itmaldataloaders.MOON_Plot(X, y)
    elif mode=='mnist':
        X, y = itmaldataloaders.MNIST_GetDataSet(fetchmode=False)
        if X.ndim==3:
            X=np.reshape(X, (X.shape[0], -1))
    elif mode=='iris':
        X, y = itmaldataloaders.IRIS_GetDataSet()
    elif mode=='spotify_binarized':
        X, y = getSpotifyBinarizedXY()
    elif mode=='spotify_intlabels':
        X, y = getSpotifyIntLabeledXY()
    else:
        raise ValueError(f"could not load data for that particular mode='{mode}'")
        
    print(f'  org. data:  X.shape      ={ShapeToString(X)}, y.shape      ={ShapeToString(y)}')

    assert X.ndim==2
    assert X.shape[0]==y.shape[0]
    # assert y.ndim==1 or (y.ndim==2 and y.shape[1]==0)    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0, shuffle=True
    )
    
    print(f'  train data: X_train.shape={ShapeToString(X_train)}, y_train.shape={ShapeToString(y_train)}')
    print(f'  test data:  X_test.shape ={ShapeToString(X_test)}, y_test.shape ={ShapeToString(y_test)}')
    print()
    
    return X_train, X_test, y_train, y_test

print('OK')

#%%
from sklearn.neighbors import KNeighborsClassifier

# Setup data
# 'iris', 'moon', 'mnist', 'spotify_classes', 'spotify_intlabels'
X_train, X_test, y_train, y_test = LoadAndSetupData('spotify_intlabels')



# Setup search parameters
model = KNeighborsClassifier(n_jobs=-1)

tuning_parameters = {
    'n_neighbors':[3,4,5],
    'weights':('uniform', 'distance'),
    'algorithm':('ball_tree', 'kd_tree', 'brute'),
    'p':[2,3,4],   
}

# tuning_parameters = {
#     'n_neighbors':[5,10,15,20]
#     # 'weights':('uniform'),
#     # 'algorithm':('kd_tree'),
#     # 'p':[2],    
# }

CV=5
VERBOSE=0

# Run Randomized Search - RandomizedSearchCV for the model
start = time()
random_tuned = RandomizedSearchCV(
    model, 
    tuning_parameters, 
    random_state=42, 
    n_iter=8, 
    cv=CV, 
    scoring='f1_micro', 
    verbose=10, 
    n_jobs=-1, 
    iid=True)
random_tuned.fit(X_train, y_train)
t = time()-start

# Report result
b0, m0= FullReport(random_tuned , X_test, y_test, t)
print(b0)

# %%


# %%
