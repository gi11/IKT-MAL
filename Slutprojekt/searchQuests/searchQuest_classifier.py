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
sys.path.append("../../include/")
sys.path.append(os.path.dirname(__file__) + "/../../include/")
sys.path.append(os.path.dirname(__file__) + "/../")
# sys.path.append(os.path.dirname(__file__) + "\\..\\..\\include")

#%%
from libitmal import dataloaders as itmaldataloaders

#%%
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

data_csv_path = os.path.dirname(__file__) + "\\..\\DatafordelingPlot\\spotifyDBData_preprocessed.csv"
data_csv_path_onehot = os.path.dirname(__file__) + "\\..\\DatafordelingPlot\\spotifyDBData_preprocessed_onehotenc.csv"
data_csv_path_onehot_mode_tsig = os.path.dirname(__file__) + "\\..\\DatafordelingPlot\\spotifyDBData_preprocessed_mode-tsig_onehotenc.csv"

def loadPreprocessed():
    return pd.read_csv(data_csv_path, sep=',', header=0)

def loadPreprocessedOnehotenc():
    return pd.read_csv(data_csv_path_onehot, sep=',', header=0)

def loadPreprocessedModeTsigOnehotenc():
    return pd.read_csv(data_csv_path_onehot_mode_tsig, sep=',', header=0)

allgenres = ['Alternative', 'Anime', 'Blues', 'Children’s Music', 'Classical', 'Comedy',
 'Country', 'Dance', 'Electronic', 'Folk', 'Hip-Hop', 'Indie', 'Jazz', 'Movie',
 'Opera', 'Pop', 'R&B', 'Rap', 'Reggae', 'Reggaeton', 'Rock', 'Ska', 'Soul',
 'Soundtrack', 'World']

regular_input_features = ["popularity", "acousticness", "danceability",
    "energy", "instrumentalness",  "liveness", "loudness", "speechiness",
    "tempo", "valence"
]

onehot_input_features = [
    "mode_major", "mode_minor",
    "time_signature_3-4", "time_signature_4-4", "time_signature_5-4"
]

def getSpotifyBinarizedXY(remove_cols=None):
    data_filtered = loadPreprocessed()
    if (remove_cols is not None):
        for col in remove_cols:
            data_filtered = data_filtered[data_filtered.genre != col]
    X_df = data_filtered[regular_input_features]
    lb = LabelBinarizer()
    X = np.array(X_df)
    y = lb.fit_transform(np.array(data_filtered['genre']))

    print("Binarized Label to the following classes")
    print(lb.classes_)
    print(lb.classes_.shape[0])

    return X, y

def getSpotifyIntLabeledXY():
    spotifyDBData = loadPreprocessed()
    X = np.array(spotifyDBData[regular_input_features])

    label_encoder = LabelEncoder() 
    y = label_encoder.fit_transform(spotifyDBData['genre']) 

    return X, y

def getSpotifyOneHotEncXY():
    spotifyDBData = loadPreprocessedModeTsigOnehotenc()
    input_features = regular_input_features + onehot_input_features
    X = np.array(spotifyDBData[input_features])

    y = LabelEncoder().fit_transform(spotifyDBData['genre']) 
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
    ClassificationReport(model, X_test, y_test)    
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
        #remove_cols = ['Alternative', 'Anime', 'Blues', 'Children’s Music', 'Classical', 'Comedy',
            # 'Country', 'Dance', 'Electronic', 'Folk', 'Hip-Hop', 'Indie', 'Jazz', 'Movie',
            # 'Opera', 'Pop', 'R&B', 'Reggae', 'Reggaeton']
        remove_cols = None
        X, y = getSpotifyBinarizedXY(remove_cols)
    elif mode=='spotify_intlabels':
        X, y = getSpotifyIntLabeledXY()
    elif mode=='spotify_onehotin_labelout':
        X, y = getSpotifyOneHotEncXY()
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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = LoadAndSetupData('spotify_binarized')

#%%
classifier = KNeighborsClassifier(n_jobs=-1, weights='uniform', algorithm='ball_tree', p=2, n_neighbors=50)

model = OneVsRestClassifier(classifier)
model.fit(X_train, y_train)

#%%
# predictions = model.predict_proba(X_test)

y_true, y_pred = y_test, model.predict_proba(X_test)

target_names = ['Rap', 'Rock', 'Ska', 'Soul', 'Soundtrack', 'World']
print(classification_report(y_true, y_pred, target_names))

# CV=5
# VERBOSE=0

# start = time()
# random_tuned = RandomizedSearchCV(
#     model, 
#     {"n_jobs": [None]}, 
#     random_state=42,
#     n_iter=1, 
#     cv=CV, 
#     scoring='f1_micro', 
#     verbose=10, 
#     n_jobs=-1, 
#     iid=True)
# random_tuned.fit(X_train, y_train)
# t = time()-start

# # Report result
# b0, m0= FullReport(random_tuned , X_test, y_test, t)
# print(b0)



#%% Regular KNeighborsClassifier

from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = LoadAndSetupData('spotify_onehotin_labelout')

model = KNeighborsClassifier(n_jobs=-1, weights='uniform', algorithm='ball_tree', p=2, n_neighbors=50)
model.fit(X_train, y_train)
#%%
y_true, y_pred = y_test, model.predict(X_test)
# target_names = ['Rap', 'Rock', 'Ska', 'Soul', 'Soundtrack', 'World']

print(classification_report(y_true, y_pred, target_names=allgenres))

predictions= model.predict(X_test)
predictions_prob = model.predict_proba(X_test)


print(classification_report(y_true, y_pred, target_names=allgenres))
#%%
from sklearn.neighbors import KNeighborsClassifier

# Setup data
# 'iris', 'moon', 'mnist', 'spotify_binarized', 'spotify_intlabels'
X_train, X_test, y_train, y_test = LoadAndSetupData('spotify_intlabels')



# Setup search parameters
# model = KNeighborsClassifier(n_jobs=-1, weights='uniform', algorithm='ball_tree', p=2)

tuning_parameters = {
    'n_neighbors': [10,20,50,100]
    # 'weights':('uniform'),
    # 'algorithm':('ball_tree'),
    # 'p':[2]
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
#%% Genre Clustering

# genres = list(filter(lambda name: "genre_" in name, onehotenc.columns))
# audiofeature_cols = ['popularity', 'acousticness', 'danceability', 'energy', 
#                  'instrumentalness', 'liveness', 'loudness', 'speechiness', 
#                  'tempo', 'valence']       


# tracks = {genre: df_scaled.loc[df_scaled[genre] == 1]
#                     .reset_index(drop=True)
#                     .filter(audiofeature_cols)
#             for genre in genres}
    
# audiofeatures = {genre.replace("genre_",""): tracks[genre].describe()
#                  for genre in genres}

# # Mean and meadian
# afeatures_mean = {genre: tracks[genre].mean().values.flatten().tolist()
#                   for genre in genres}
# afeatures_median = {genre: tracks[genre].median().values.flatten().tolist()
#                     for genre in genres}

#%%
# from sklearn.cluster import KMeans

# X_train, X_test, y_train, y_test = LoadAndSetupData('spotify_intlabels')
# kmeans = KMeans(n_clusters=5, random_state=0).fit(X_train)


# %%


# Results from KNeighborsClassifier:

# model = KNeighborsClassifier(n_jobs=-1)
# tuning_parameters = {
#     'n_neighbors':[3,4,5],
#     'weights':('uniform', 'distance'),
#     'algorithm':('ball_tree', 'kd_tree', 'brute'),
#     'p':[2,3,4],   
# }

# CTOR for best model: KNeighborsClassifier(algorithm='ball_tree', leaf_size=30, metric='minkowski',
#                     metric_params=None, n_jobs=-1, n_neighbors=5, p=2,
#                     weights='uniform')
# best: dat=spotify_intlabels, score=0.23061, 
# model=KNeighborsClassifier(algorithm='ball_tree',n_neighbors=5,p=2,weights='uniform')



# model = KNeighborsClassifier(n_jobs=-1, weights='uniform', algorithm='ball_tree', p=2)
# tuning_parameters = {
#     'n_neighbors':[5,10,20]
# }

# Grid scores ('f1_micro') on development set:
# 	[ 0]: 0.231 (+/-0.003) for {'n_neighbors': 5}
# 	[ 1]: 0.257 (+/-0.002) for {'n_neighbors': 10}
# 	[ 2]: 0.275 (+/-0.003) for {'n_neighbors': 20}
# CTOR for best model: KNeighborsClassifier(algorithm='ball_tree', leaf_size=30, metric='minkowski',
#                      metric_params=None, n_jobs=-1, n_neighbors=20, p=2,
#                      weights='uniform')
# best: dat=spotify_intlabels, score=0.27533, model=KNeighborsClassifier(n_neighbors=20)


# model = KNeighborsClassifier(n_jobs=-1, weights='uniform', algorithm='ball_tree', p=2)
# tuning_parameters = {
#     'n_neighbors':[10,20,50,100]
#     # 'weights':('uniform'),
#     # 'algorithm':('ball_tree'),
#     # 'p':[2]
# }

# Grid scores ('f1_micro') on development set:
# 	[ 0]: 0.257 (+/-0.002) for {'n_neighbors': 10}
# 	[ 1]: 0.275 (+/-0.003) for {'n_neighbors': 20}
# 	[ 2]: 0.289 (+/-0.004) for {'n_neighbors': 50}
# 	[ 3]: 0.292 (+/-0.004) for {'n_neighbors': 100}
# CTOR for best model: KNeighborsClassifier(algorithm='ball_tree', leaf_size=30, metric='minkowski',
#                      metric_params=None, n_jobs=-1, n_neighbors=100, p=2,
#                      weights='uniform')
# best: dat=spotify_intlabels, score=0.29227, model=KNeighborsClassifier(n_neighbors=100)