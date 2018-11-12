import numpy as np
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Model, Input, Sequential
from keras.layers import GRU, LSTM, Embedding, Dense, TimeDistributed, Bidirectional, Activation, Dropout
from sklearn.model_selection import train_test_split
from keras.metrics import categorical_accuracy
from keras import backend as K
import tensorflow as tf
from keras import optimizers
import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import KFold
from keras.callbacks import History
import matplotlib.pyplot as plt

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

def DLClassification(datasetfile, min_connections):
    X, y = data_load_and_filter(datasetfile, min_connections)

    ##### BASIC PARAMETERS #####
    n_samples = np.shape(X)[0]
    time_steps = np.shape(X)[1] # we have a time series of 100 payload sizes
    n_features = 1 # 1 feature which is packet size
    seq_len = 100

    ##### CREATES MAPPING FROM SNI STRING TO INT #####
    class_map = {sni:i for i, sni in enumerate(np.unique(y))}
    rev_class_map = {val: key for key, val in class_map.items()}

    n_labels = len(class_map)

    ##### CHANGE Y TO PD SO ITS EASIER TO MAP #####
    y_pd = pd.DataFrame(y)
    y_pd = y_pd[0].map(class_map)

    print(y_pd.head)

    ##### DUPLICATE Y LABELS, WE WILL NEED THIS LATER #####
    y = y_pd.values.reshape(n_samples,)

    ##### CREATE A NEW SEQUENCE ARRAY OF 0s THAT ARE INTS #####
    sequences = np.zeros((len(X), seq_len), dtype=int)
    
    ##### COPY X_TRAIN INTO THE SEQUENCES BUT THIS TIME IT'LL ALL BE INTS #####
    for i, row in enumerate(X):
        line = np.array(row)
        sequences[i, -len(row):] = line[-seq_len:]

    ##### REPLACE X_TRAIN WITH THE NEW INT ARRAY #####
    X = sequences

    ##### RESHAPE FOR LSTM #####
    X = np.reshape(X, (n_samples, time_steps, n_features))

    print(y.shape)
    ##### TRAIN TEST SPLIT #####

    BATCH_SIZE = 32
    EPOCHS = 20
    FOLDS = 10

    # FOR NOW TRUNCATE SOME DATA SO ITS PROPERLY DIVISIBLE SO WE CAN USE STATEFULNESS, WILL IMPLEMENT A BETTER APPROACH LATER
    cutoff = BATCH_SIZE * FOLDS * int(len(X) / (BATCH_SIZE * FOLDS))
    X = X[:cutoff]
    y = y[:cutoff]

    kf = KFold(n_splits=FOLDS, shuffle=True)
    total = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = Sequential()
        history = History()

        model.add(LSTM(128, return_sequences=True, stateful=True, batch_input_shape=(BATCH_SIZE, time_steps, n_features)))
        model.add(LSTM(128, return_sequences=True, stateful=True))
        model.add(LSTM(128, stateful=True))
        model.add(Dense(n_labels, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['acc'])
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, shuffle=False, validation_data=(X_test, y_test), callbacks = [history])
        
        accuracy = history.history['val_acc'][-1]
        print("ACCURACY: ", accuracy)
        total+= accuracy

    print("AVG: ", 1. * total / FOLDS)
    return 1. * total / FOLDS


if __name__ == "__main__":
    # run once w/min connections = 100 as in paper
    DLClassification("training/GCDay1seq100.csv", 100)

    # try a variety of min conn settings for graph
    min_connections_to_try = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
    accuracies = []
    for min_connections in min_connections_to_try:
        accuracies.append(DLClassification("training/GCDay1seq100.csv", min_connections))

    plt.plot(min_connections_to_try, accuracies)
    plt.xlabel("Mininimum Connections")
    plt.ylabel("Accuracy")
    plt.show()


