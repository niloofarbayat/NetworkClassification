import numpy as np
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Model, Input, Sequential
from keras.layers import GRU, LSTM, Embedding, Dense, TimeDistributed, Bidirectional, Activation
from sklearn.model_selection import train_test_split
from keras.metrics import categorical_accuracy
from keras import backend as K
import tensorflow as tf
from keras import optimizers
import pandas as pd
from keras.utils.np_utils import to_categorical

def read_csv(file_path, has_header=True):
    with open(file_path) as f:
        if has_header: f.readline()
        data = []
        for line in f:
            line = line.strip().split(",")
            data.append([x for x in line])
    return data

def data_load_and_filter(datasetfile, min_connections):
    dataset = read_csv(datasetfile)
    X = np.array([z[1:] for z in dataset])
    y = np.array([z[0] for z in dataset])
    print("Shape of X =", np.shape(X))
    print("Shape of y =", np.shape(y))     
    
    print("Entering filtering section! ")
    snis, counts = np.unique(y, return_counts=True)
    above_min_conns = list()

    for i in range(len(counts)):
        if counts[i] > min_connections:
            above_min_conns.append(snis[i])

    print("Filtering done. SNI classes remaining: ", len(above_min_conns))
    indices = np.isin(y, above_min_conns)
    X = X[indices]
    y = y[indices]

    print("Filtered shape of X =", np.shape(X))
    print("Filtered shape of y =", np.shape(y))   

    return X, y

#########################################################
##### USE SAME MIN_CONN FILTER AS PAPER EXCEPT WE   #####
##### USE FIRST 100 SEQ AND WE DON'T DROP COLUMNS   #####
#########################################################

X, y = data_load_and_filter("training/GCDay1seq.csv", 100)

##### BASIC PARAMETERS ####
n_samples = np.shape(X)[0]
time_steps = np.shape(X)[1] # we have a time series of 100 payload sizes
n_features = 1 # 1 feature which is payload size
seq_len = 100
n_neurons = 50

##### CREATES MAPPING FROM SNI STRING TO INT #####
class_map = {sni:i for i, sni in enumerate(np.unique(y))}
rev_class_map = {val: key for key, val in class_map.items()}

n_labels = len(class_map)
print(n_labels)

##### CHANGE Y TO PD SO ITS EASIER TO MAP #####
y_pd = pd.DataFrame(y)
y_pd = y_pd[0].map(class_map)

##### DUPLICATE Y LABELS, WE WILL NEED THIS LATER #####
y = y_pd.values.reshape(n_samples,1).repeat(time_steps, axis=1)

##### CREATE A NEW SEQUENCE ARRAY OF 0s THAT ARE INTS #####
sequences = np.zeros((len(X), seq_len), dtype=int)

##### COPY X_TRAIN INTO THE SEQUENCES BUT THIS TIME IT'LL ALL BE INTS #####
for i, row in enumerate(X):
    line = np.array(row)
    sequences[i, -len(row):] = line[-seq_len:]

##### REPLACE X_TRAIN WITH THE NEW INT ARRAY #####
X = sequences

##### RESHAPE FOR LSTM #####
X = X.reshape(n_samples, time_steps, n_features)
y = y.reshape(n_samples, time_steps, n_features)

##### TRAIN TEST SPLIT #####
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1, random_state = 0)

##### BUILD LSTM MODEL #####
model = Sequential()


