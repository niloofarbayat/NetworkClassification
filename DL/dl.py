import numpy as np
from sklearn.metrics import accuracy_score
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
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import autosklearn.classification
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler

BATCH_SIZE = 64
EPOCHS = 100 # use early stopping
FOLDS = 10
SEQ_LEN = 25
NUM_ROWS = -1 # just use first day for now, set to -1 for all data
MIN_CONNECTIONS_LIST = [100]

def read_csv(file_path, has_header=True):
    with open(file_path) as f:
        if has_header: f.readline()
        data = []
        for line in f:
            line = line.strip().split(",")
            data.append([x for x in line])
    return data

####################################################
# Filter for SNIs meeting min connection threshold
####################################################
def data_load_and_filter(datasetfile, min_connections):
    dataset = read_csv(datasetfile)

    # Use first n rows
    dataset = dataset[:NUM_ROWS]

    # packet sizes
    X1 = np.array([z[1:SEQ_LEN + 1] for z in dataset])

    # payload sizes
    X2 = np.array([z[SEQ_LEN + 1:2*SEQ_LEN + 1] for z in dataset])

    # inter-arrival times
    X3 = np.array([z[2*SEQ_LEN + 1:3*SEQ_LEN + 1] for z in dataset])
    X3 = X3.astype(float)
    X3[np.where(X3 != 0 )] = np.log(X3[np.where(X3 != 0 )])
    
    # direction
    X4 = np.array([z[3*SEQ_LEN + 1:4*SEQ_LEN + 1] for z in dataset])

    y = np.array([z[0] for z in dataset])
    print("Shape of X1 =", np.shape(X1))
    print("Shape of X2 =", np.shape(X2))
    print("Shape of X3 =", np.shape(X3))
    print("Shape of X4 =", np.shape(X4))
    print("Shape of y =", np.shape(y))     
    
    print("Entering min connections filter section! ")
    snis, counts = np.unique(y, return_counts=True)
    above_min_conns = list()

    for i in range(len(counts)):
        if counts[i] > min_connections:
            above_min_conns.append(snis[i])

    print("Filtering done. SNI classes remaining: ", len(above_min_conns))
    indices = np.isin(y, above_min_conns)
    X1 = X1[indices]
    X2 = X2[indices]
    X3 = X3[indices]
    X4 = X4[indices]
    y = y[indices]

    print("Filtered shape of X1 =", np.shape(X1))
    print("Filtered shape of X2 =", np.shape(X2))
    print("Filtered shape of X3 =", np.shape(X3))
    print("Filtered shape of X4 =", np.shape(X4))
    print("Filtered shape of y =", np.shape(y))   

    ##### BASIC PARAMETERS #####
    n_samples = np.shape(X1)[0]
    time_steps = np.shape(X1)[1] # we have a time series of 100 payload sizes
    n_features = 1 # 1 feature which is packet size

    ##### CREATES MAPPING FROM SNI STRING TO INT #####
    class_map = {sni:i for i, sni in enumerate(np.unique(y))}
    rev_class_map = {val: key for key, val in class_map.items()}

    n_labels = len(class_map)

    ##### CHANGE Y TO PD SO ITS EASIER TO MAP #####
    y_pd = pd.DataFrame(y)
    y_pd = y_pd[0].map(class_map)

    ##### DUPLICATE Y LABELS, WE WILL NEED THIS LATER #####
    y = y_pd.values.reshape(n_samples,)

    return X1, X2, X3, X4, y, time_steps, n_features, n_labels, rev_class_map

#########################################################
###### USE RNN TO CLASSIFY PACKET SEQUENCES -> SNI ######
#########################################################
def DLClassification(X_train, X_test, y_train, y_test,time_steps, n_features, n_labels, dropout):
    X_train = np.stack([X_train], axis=2)
    X_test = np.stack([X_test], axis=2)

    # if you dont have newest keras version, you might have to remove restore_best_weights = True
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='min')
    model = Sequential()
    model.add(Conv1D(200, 3, activation='relu', input_shape=(time_steps, n_features)))
    model.add(BatchNormalization())
    model.add(Conv1D(400, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(GRU(200))
    model.add(Dropout(dropout))
    model.add(Dense(200, activation='sigmoid')) 
    model.add(Dropout(dropout))
    model.add(Dense(n_labels, activation='softmax')) 
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['acc'])
    model.summary()
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, shuffle=False, validation_data=(X_test, y_test), callbacks = [early_stopping])
    return model.predict(X_test)
  
#***********************************************************************************
# autosklearn classifier to find the best achievable accuracy
#***********************************************************************************
def auto_sklearn_classification(X_train, X_test, y_train, y_test):
  cls = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=300, per_run_time_limit=90, ml_memory_limit=50000)
  cls.fit(X_train, y_train)
  print(cls.sprint_statistics())
  print(cls.show_models())
  predictions = cls.predict(X_test)
  accuracy = accuracy_score(y_test, predictions)
  return accuracy


if __name__ == "__main__":
    datasetfile = "training/GCseq25.csv"

    kf = KFold(n_splits=FOLDS, shuffle=True)

    # try a variety of min conn settings for graph
    accuracies = []
    for min_connections in MIN_CONNECTIONS_LIST:
        X1, X2, X3, X4, y, time_steps, n_features, n_labels, rev_class_map = data_load_and_filter(datasetfile, min_connections)

        total_nn1, total_nn2, total_nn3, total_nn123, total_cls = 0, 0, 0, 0, 0
        for train_index, test_index in kf.split(X1):

            X1_train, X1_test = X1[train_index], X1[test_index] # Packet sizes
            X2_train, X2_test = X2[train_index], X2[test_index] # Payload sizes
            X3_train, X3_test = X3[train_index], X3[test_index] # Inter-Arrival Times
            
            # Directional features not used!
            # X4_train, X4_test = X4[train_index], X4[test_index]

            y_train, y_test = y[train_index], y[test_index]

            # CNN-RNN for Packet Size
            predictions1 = DLClassification(X1_train, X1_test, y_train, y_test, time_steps, n_features, n_labels, dropout=0.0)

            # CNN-RNN for Payload Size
            predictions2 = DLClassification(X2_train, X2_test, y_train, y_test, time_steps, n_features, n_labels, dropout=0.0)

            # CNN-RNN for Inter-Arrival times
            predictions3 = DLClassification(X3_train, X3_test, y_train, y_test, time_steps, n_features, n_labels, dropout=0.25)

            nn_acc1 = 1. * np.sum([np.argmax(x) for x in predictions1] == y_test) / len(y_test)
            print("CNN-RNN Packet ACCURACY: %s"%(nn_acc1))

            nn_acc2 = 1. * np.sum([np.argmax(x) for x in predictions2] == y_test) / len(y_test)
            print("CNN-RNN Payload ACCURACY: %s"%(nn_acc2))

            nn_acc3 = 1. * np.sum([np.argmax(x) for x in predictions3] == y_test) / len(y_test)
            print("CNN-RNN IAT ACCURACY: %s"%(nn_acc3))

            # Ensemble CNN-RNN
            predictions123 = (predictions1 * (1.0/3) + predictions2 * (1.0/3) + predictions3 * (1.0/3))
            nn_acc123 = 1. * np.sum([np.argmax(x) for x in predictions123] == y_test) / len(y_test)
            print("Ensemble CNN-RNN ACCURACY: %s"%(nn_acc123))

            total_nn1+= nn_acc1
            total_nn2+= nn_acc2
            total_nn3+= nn_acc3
            total_nn123+= nn_acc123
            
            # Uncomment for auto sklearn results on sequence features
            # cls_acc = auto_sklearn_classification(X_train, X_test, y_train, y_test)
            # print("Auto sklearn Accuracy: %s "%(cls_acc))
            # total_cls += cls_acc

            # Uncomment to run once
            # FOLDS = 1
            # break

        total_nn1 = 1. * total_nn1 / FOLDS
        total_nn2 = 1. * total_nn2 / FOLDS
        total_nn3 = 1. * total_nn3 / FOLDS
        total_nn123 = 1. * total_nn123 / FOLDS
        total_cls = 1. * total_cls / FOLDS

        print("AVG CNN-RNN Packet: %s\n AVG CNN-RNN Payload: %s\n AVG CNN-RNN IAT: %s\n AVG CNN-RNN Ensemble: %s\n AVG CLS: %s\n "%(total_nn1, total_nn2, total_nn3, total_nn123, total_cls))

        accuracies.append([total_nn1, total_nn2, total_nn3, total_nn123, total_cls])

    print(accuracies)


