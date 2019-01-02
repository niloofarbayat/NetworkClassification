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
# BEST CNN-RNN FOR SEQUENCE CLASSIFICATION
#########################################################
def DLClassification(X_train, X_test, y_train, y_test, time_steps, n_features, n_labels):
    # if you dont have newest keras version, you might have to remove restore_best_weights = True
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='min', restore_best_weights=True)
    model = Sequential()
    model.add(Conv1D(200, 3, activation='relu', input_shape=(time_steps, n_features)))
    model.add(BatchNormalization())
    model.add(Conv1D(400, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(GRU(200))
    model.add(Dropout(0.1))
    model.add(Dense(200, activation='sigmoid')) 
    model.add(Dropout(0.1))
    model.add(Dense(n_labels, activation='softmax')) 
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['acc'])
    model.summary()
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, shuffle=True, validation_split=0.05, callbacks = [early_stopping])
    return model.predict(X_test)

if __name__ == "__main__":
    statistics = [["model", "min connections", "accuracy", "precision", "recall", "f1_score"]]
    for min_connections in MIN_CONNECTIONS_LIST:
        datasetfile = "DL/training/GCseq.csv"
        X, y = data_load_and_filter(datasetfile, min_connections, NUM_ROWS)
        X1, X2, X3, X4, y, time_steps, n_labels, rev_class_map = process_dl_features(X, y, SEQ_LEN)

        datasetfile = "ML/training/GCstats.csv"
        X, _ = data_load_and_filter(datasetfile, min_connections, NUM_ROWS)

        # Classifiers to test!
        stats = {}
        for model in ["Random Forest", "Packet", "Payload", "IAT", "Ensemble", "Ensemble + RF", "Ensemble + Domain", "Ensemble + RF + Domain"]:
            stats[model] = [0,0,0,0]

        FOLDS = 5
        kf = KFold(n_splits=FOLDS, shuffle=True)

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

            # Directional features
            X4_train, X4_test = X4[train_index], X4[test_index]

            # Labels
            y_train, y_test = y[train_index], y[test_index]

            # Random Forest classifier
            predictions_rf = MLClassification(X_train, X_test, y_train, y_test)
            stats = update_stats(stats, "Random Forest", predictions_rf, y_test)
            print("Random Forest ACCURACY: %s"%(stats["Random Forest"][0]))

            # Create 3D input arrays (batch_size, time_steps, n_features = 2)
            X_train = np.stack([X1_train, X4_train], axis=2)
            X_test = np.stack([X1_test, X4_test], axis=2)

            predictions_1 = DLClassification(X_train, X_test, y_train, y_test, time_steps, 2, n_labels)
            stats = update_stats(stats, "Packet", predictions_1, y_test)
            print("Packet ACCURACY: %s"%(stats["Packet"][0]))

            # Create 3D input arrays (batch_size, time_steps, n_features = 2)
            X_train = np.stack([X2_train, X4_train], axis=2)
            X_test = np.stack([X2_test, X4_test], axis=2)

            predictions_2 = DLClassification(X_train, X_test, y_train, y_test, time_steps, 2, n_labels)
            stats = update_stats(stats, "Payload", predictions_2, y_test)
            print("Payload ACCURACY: %s"%(stats["Payload"][0]))

            # Create 3D input arrays (batch_size, time_steps, n_features = 2)
            X_train = np.stack([X3_train, X4_train], axis=2)
            X_test = np.stack([X3_test, X4_test], axis=2)

            predictions_3 = DLClassification(X_train, X_test, y_train, y_test, time_steps, 2, n_labels)
            stats = update_stats(stats, "IAT", predictions_3, y_test)
            print("IAT ACCURACY: %s"%(stats["IAT"][0]))

            predictions_123 = (predictions_1 + predictions_2 + predictions_3) / 3.0
            stats = update_stats(stats, "Ensemble", predictions_123, y_test)
            print("Ensemble ACCURACY: %s"%(stats["Ensemble"][0]))

            # domain expertise
            _, freqs = np.unique(y_train, return_counts=True)
            predictions_domain = domain_expertise(predictions_123, freqs, y_test)
            stats = update_stats(stats, "Ensemble + Domain", predictions_domain, y_test)
            print("Ensemble + Domain ACCURACY: %s"%(stats["Ensemble + Domain"][0]))

            predictions_123_rf = (predictions_rf * 0.5 + predictions_123 * 0.5)
            stats = update_stats(stats, "Ensemble + RF", predictions_123_rf, y_test)
            print("Ensemble + RF ACCURACY: %s"%(stats["Ensemble + RF"][0]))

            # domain expertise
            _, freqs = np.unique(y_train, return_counts=True)
            predictions_domain = domain_expertise(predictions_123_rf, freqs, y_test)
            stats = update_stats(stats, "Ensemble + RF + Domain", predictions_domain, y_test)
            print("Ensemble + RF + Domain ACCURACY: %s"%(stats["Ensemble + RF + Domain"][0]))

            # Uncomment below to get per-SNI accuracy
            # output_class_accuracies(rev_class_map, predictions_rf, predictions_1, predictions_2, predictions_3, predictions_123, predictions_123_rf)

            # Uncomment to run once
            FOLDS = 1
            break

        for model, stats in stats.items():
            statistics.append([model, min_connections] + [1. * x / FOLDS for x in stats])
        
        with open('final_results.csv', 'a') as file:
            wr = csv.writer(file)
            for statistic in statistics:
                wr.writerow(statistic)
            statistics = []
        
