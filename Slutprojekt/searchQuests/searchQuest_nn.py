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

data_csv_path = os.path.dirname(__file__) + "\\..\\DatafordelingPlot\\spotifyDBData_preprocessed_minmaxscaled.csv"
data_csv_path_onehot = os.path.dirname(__file__) + "\\..\\DatafordelingPlot\\spotifyDBData_preprocessed_minmaxscaled_onehotenc-all.csv"
data_csv_path_onehot_mode_tsig = os.path.dirname(__file__) + "\\..\\DatafordelingPlot\\spotifyDBData_preprocessed_minmaxscaled_onehotenc-mode-tsig.csv"
data_csv_path_onehot_mode_tsig_stdscaled = os.path.dirname(__file__) + "\\..\\DatafordelingPlot\\spotifyDBData_preprocessed_stdscaled_onehotenc-mode-tsig.csv"

def loadPreprocessed():
    return pd.read_csv(data_csv_path, sep=',', header=0)

def loadPreprocessedOnehotenc():
    return pd.read_csv(data_csv_path_onehot, sep=',', header=0)

def loadPreprocessedModeTsigOnehotenc():
    return pd.read_csv(data_csv_path_onehot_mode_tsig, sep=',', header=0)

def loadPreprocessedModeTsigOnehotencStdScaled():
    return pd.read_csv(data_csv_path_onehot_mode_tsig_stdscaled, sep=',', header=0)

all_genres = ['Alternative', 'Anime', 'Blues', 'Children’s Music', 'Classical', 'Comedy',
 'Country', 'Dance', 'Electronic', 'Folk', 'Hip-Hop', 'Indie', 'Jazz', 'Movie',
 'Opera', 'Pop', 'R&B', 'Rap', 'Reggae', 'Reggaeton', 'Rock', 'Ska', 'Soul',
 'Soundtrack', 'World']

regular_input_features = [
    "popularity", "acousticness", "danceability",
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

def getSpotifyOneHotEncStdScaledXY():
    spotifyDBData = loadPreprocessedModeTsigOnehotencStdScaled()
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
    elif mode=='spotify_onehotin_labelout_std':
        X, y = getSpotifyOneHotEncStdScaledXY()
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


#%% -----------------------------------
#   Neural Network
# -------------------------------------

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn import datasets

import numpy as np
from time import time

np.random.seed(42)

X_train, X_test, y_train, y_test = LoadAndSetupData('spotify_onehotin_labelout')

y_train_binary = to_categorical(y_train)
y_test_binary  = to_categorical(y_test)

assert y_train_binary.ndim==2
assert y_test_binary.ndim ==2

#%%
from keras import backend as K

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])

# # fit the model
# history = model.fit(Xtrain, ytrain, validation_split=0.3, epochs=10, verbose=0)

# # evaluate the model
# loss, accuracy, f1_score, precision, recall = model.evaluate(Xtest, ytest, verbose=0)

#%%

# Build Keras model 
model = Sequential()
model.add(Dense(input_dim=15, units=20, activation="tanh", kernel_initializer="normal"))
model.add(Dense(units=25, activation="softmax"))

#optimizer = SGD(lr=0.1)
optimizer = Adam(lr=0.1)
model.compile(loss='categorical_crossentropy', 
              optimizer=optimizer, 
              metrics=['acc', f1_m, precision_m, recall_m])

# Train
VERBOSE     = 10
EPOCHS      = 35

start = time()
history = model.fit(X_train, y_train_binary, 
    validation_data=(X_test, y_test_binary), 
    epochs=EPOCHS, verbose=VERBOSE)
t = time()-start

print(f"OK, training time={t:0.1f}")


#%%
# score = model.evaluate(X_test, y_test_binary, verbose=0)

# print(f"Training time: {t:0.1f} sec")
# print({score[0]}) # loss is score 0 by definition?
# print({score[1]})
# print(f"All scores in history: {score}")

loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test_binary, verbose=10)
print("loss: " + str(loss))
print("accuracy: " + str(accuracy))
print("f1_score: " + str(f1_score))
print("precision: " + str(precision))
print("recall: " + str(recall))

# from sklearn.metrics import classification_report

# y_pred = model.predict(x_test, batch_size=64, verbose=1)
# y_pred_bool = np.argmax(y_pred, axis=1)

# print(classification_report(y_test, y_pred_bool))