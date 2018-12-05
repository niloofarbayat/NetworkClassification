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
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import csv
from collections import defaultdict
from utils import *

BATCH_SIZE = 64
EPOCHS = 100 # use early stopping
FOLDS = 10
SEQ_LEN = 25
NUM_ROWS = -1 # set to -1 for all data
MIN_CONNECTIONS_LIST = [100] # try a variety of min conn settings for model

#########################################################
# RANDOM FOREST FOR ML CLASSIFICATION
#########################################################
def MLClassification(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=250, n_jobs=10)
    rf.fit(X_train, y_train)
    return rf.predict_proba(X_test)

#########################################################
# BASELINE RNN FOR SEQUENCE CLASSIFICATION
#########################################################
def BaselineDLClassification(X_train, X_test, y_train, y_test,time_steps, n_features, n_labels): 
    # Create 3D input arrays (batch_size, time_steps, n_features = 1)
    X_train = np.stack([X_train], axis=2)
    X_test = np.stack([X_test], axis=2)

    # if you dont have newest keras version, you might have to remove restore_best_weights = True
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='min', restore_best_weights=True)
    model = Sequential()
    model.add(GRU(100, return_sequences=True, input_shape=(time_steps, n_features)))
    model.add(GRU(100, input_shape=(time_steps, n_features)))
    model.add(Dense(n_labels, activation='softmax')) 
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['acc'])
    model.summary()    
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, shuffle=False, validation_data=(X_test, y_test), callbacks = [early_stopping])
    return model.predict(X_test)

#########################################################
# BEST CNN-RNN FOR SEQUENCE CLASSIFICATION
#########################################################
def DLClassification(X_train, X_test, y_train, y_test, time_steps, n_features, n_labels, dropout):
    # Create 3D input arrays (batch_size, time_steps, n_features = 1)
    X_train = np.stack([X_train], axis=2)
    X_test = np.stack([X_test], axis=2)

    # if you dont have newest keras version, you might have to remove restore_best_weights = True
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='min', restore_best_weights=True)
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

if __name__ == "__main__":
    kf = KFold(n_splits=FOLDS, shuffle=True)

    statistics = [["model", "min connections", "accuracy", "precision", "recall", "f1_score"]]
    for min_connections in MIN_CONNECTIONS_LIST:
        datasetfile = "DL/training/GCseq25.csv"
        X, y = data_load_and_filter(datasetfile, min_connections, NUM_ROWS)
        X1, X2, X3, X4, y, time_steps, n_labels, n_features, rev_class_map = process_dl_features(X, y, SEQ_LEN)

        datasetfile = "ML/training/GCstats.csv"
        X, _ = data_load_and_filter(datasetfile, min_connections, NUM_ROWS)

        # Classifiers to test!
        stats = {}
        for model in ["Random Forest", "Baseline RNN", "Packet CNN-RNN", "Payload CNN-RNN", "IAT CNN-RNN", "Ensemble CNN-RNN", "Ensemble RF + CNN-RNN"]:
            stats[model] = [0,0,0,0]


        # Perform 10-Fold Cross Validation
        for train_index, test_index in kf.split(X1):
            
            # Statistical features
            X_train, X_test = X[train_index], X[test_index]

            # Packet features
            X1_train, X1_test = X1[train_index], X1[test_index]

            # Payload features
            X2_train, X2_test = X2[train_index], X2[test_index]

            # Inter-Arrival Time features
            X3_train, X3_test = X3[train_index], X3[test_index]

            # Directional features gave no improvement
            # X4_train, X4_test = X4[train_index], X4[test_index] 

            # Labels
            y_train, y_test = y[train_index], y[test_index]
            
            # Random Forest classifier
            predictions_rf = MLClassification(X_train, X_test, y_train, y_test)
            stats = update_stats(stats, "Random Forest", predictions_rf, y_test)
            print("Random Forest ACCURACY: %s"%(stats["Random Forest"][0]))

            # Baseline RNN classifier
            predictions_bl = BaselineDLClassification(X1_train, X1_test, y_train, y_test, time_steps, n_features, n_labels)
            stats = update_stats(stats, "Baseline RNN", predictions_bl, y_test)
            print("Baseline RNN Packet ACCURACY: %s"%(stats["Baseline RNN"][0]))

            # CNN-RNN classifier trained on packet sizes
            predictions1 = DLClassification(X1_train, X1_test, y_train, y_test, time_steps, n_features, n_labels, dropout=0.0)
            stats = update_stats(stats, "Packet CNN-RNN", predictions1, y_test)
            print("CNN-RNN Packet ACCURACY: %s"%(stats["Packet CNN-RNN"][0]))

            # CNN-RNN classifier trained on payload sizes
            predictions2 = DLClassification(X2_train, X2_test, y_train, y_test, time_steps, n_features, n_labels, dropout=0.0)
            stats = update_stats(stats, "Payload CNN-RNN", predictions2, y_test)
            print("CNN-RNN Payload ACCURACY: %s"%(stats["Payload CNN-RNN"][0]))

            # CNN-RNN classifier trained on inter-arrival times
            predictions3 = DLClassification(X3_train, X3_test, y_train, y_test, time_steps, n_features, n_labels, dropout=0.25)
            stats = update_stats(stats, "IAT CNN-RNN", predictions3, y_test)
            print("CNN-RNN IAT ACCURACY: %s"%(stats["IAT CNN-RNN"][0]))

            # Ensemble CNN-RNN (packet + payload + Inter-Arrival Time)
            predictions123 = (predictions1 * (1.0/3) + predictions2 * (1.0/3) + predictions3 * (1.0/3))
            stats = update_stats(stats, "Ensemble CNN-RNN", predictions123, y_test)
            print("Ensemble CNN-RNN ACCURACY: %s"%(stats["Ensemble CNN-RNN"][0]))

            # Ensemble (Random Forest + Ensemble CNN-RNN)
            predictions_best = (predictions_rf * 0.5 + predictions123 * 0.5)
            stats = update_stats(stats, "Ensemble RF + CNN-RNN", predictions_best, y_test)
            print("Ensemble RF + CNN-RNN ACCURACY: %s"%(stats["Ensemble RF + CNN-RNN"][0]))


            # Uncomment below to get per-SNI accuracy
            # output_class_accuracies(rev_class_map, predictions_rf, predictions_bl, predictions1, predictions2, predictions3, predictions123, predictions_best)

            # Uncomment to run once
            # FOLDS = 1
            # break

        for model, stats in stats.items():
            statistics.append([model, min_connections] + [1. * x / FOLDS for x in stats])
        
        with open('final_results.csv', 'a') as file:
            wr = csv.writer(file)
            for statistic in statistics:
                wr.writerow(statistic)
            statistics = []
        
