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
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import csv
from collections import defaultdict

BATCH_SIZE = 64
EPOCHS = 100 # use early stopping
FOLDS = 10
SEQ_LEN = 25
NUM_ROWS = -1 # just use first day for now, set to -1 for all data
MIN_CONNECTIONS_LIST = [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]

def read_csv(file_path, has_header=True):
    with open(file_path) as f:
        if has_header: f.readline()
        data = []
        for line in f:
            line = line.strip().split(",")
            data.append([x for x in line])
    return data

def ml_data_load_and_filter(datasetfile, min_connections):
    dataset = read_csv(datasetfile)
    
    dataset = dataset[:NUM_ROWS]

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
    
    #it's needed for auto_sklearn to work
    X = X.astype(np.float)
    return X, y

def dl_data_load_and_filter(datasetfile, min_connections):
    dataset = read_csv(datasetfile)

    # day one
    dataset = dataset[:NUM_ROWS]

    # packet sizes
    X1 = np.array([z[1:SEQ_LEN + 1] for z in dataset])

    # payload sizes
    X2 = np.array([z[SEQ_LEN + 1:2*SEQ_LEN + 1] for z in dataset])

    # inter-arrival times
    X3 = np.array([z[2*SEQ_LEN + 1:3*SEQ_LEN + 1] for z in dataset])
    X3 = X3.astype(float)
    X3[np.where(X3 != 0 )] = np.log(X3[np.where(X3 != 0 )])

    y = np.array([z[0] for z in dataset])
    print("Shape of X1 =", np.shape(X1))
    print("Shape of X2 =", np.shape(X2))
    print("Shape of X3 =", np.shape(X3))
    print("Shape of y =", np.shape(y))     
    
    print("Entering filtering section, min connections =", min_connections)
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
    y = y[indices]

    print("Filtered shape of X1 =", np.shape(X1))
    print("Filtered shape of X2 =", np.shape(X2))
    print("Filtered shape of X3 =", np.shape(X3))
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

    return X1, X2, X3, y, time_steps, n_features, n_labels, rev_class_map


#########################################################
###### USE RNN TO CLASSIFY PACKET SEQUENCES -> SNI ######
#########################################################

def DLClassification(X_train, X_test, y_train, y_test,time_steps, n_features, n_labels, dropout):
    # if you dont have newest keras version, you might have to remove restore_best_weights = True
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='min', restore_best_weights=True)
    model = Sequential()
    model.add(Conv1D(n_labels + 128, 3, activation='relu', input_shape=(time_steps, n_features)))
    model.add(BatchNormalization())
    model.add(Conv1D(n_labels + 256, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(GRU(n_labels + 100))
    model.add(Dropout(dropout))
    model.add(Dense(n_labels + 50, activation='sigmoid')) 
    model.add(Dropout(dropout))
    model.add(Dense(n_labels, activation='softmax')) 
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['acc'])
    model.summary()
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, shuffle=False, validation_data=(X_test, y_test), callbacks = [early_stopping])
    return model

def MLClassification(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=250, n_jobs=10)
    rf.fit(X_train, y_train)
    return rf

def BaselineDLClassification(X_train, X_test, y_train, y_test,time_steps, n_features, n_labels): 
    # if you dont have newest keras version, you might have to remove restore_best_weights = True
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='min', restore_best_weights=True)
    model = Sequential()
    model.add(GRU(n_labels + 100, return_sequences=True, input_shape=(time_steps, n_features)))
    model.add(GRU(n_labels + 100, input_shape=(time_steps, n_features)))
    model.add(Dense(n_labels, activation='softmax')) 
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['acc'])
    model.summary()    
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, shuffle=False, validation_data=(X_test, y_test), callbacks = [early_stopping])
    return model

def get_classification_accuracies_image(predictions_rf, predictions1, predictions2, predictions3, predictions123, predictions_all):
    classes = []
    accuracies1 = []
    accuracies2 = []
    accuracies3 = []
    accuracies123 = []
    accuracies_rf = []
    accuracies_all= []

    snis = np.unique(y_test)
    for sni in snis:
        indices = np.where(y_test == sni)
        correct1 = np.sum([np.argmax(x) for x in predictions1[indices]] == y_test[indices])
        correct2 = np.sum([np.argmax(x) for x in predictions2[indices]] == y_test[indices])
        correct3 = np.sum([np.argmax(x) for x in predictions3[indices]] == y_test[indices])
        correct123 = np.sum([np.argmax(x) for x in predictions123[indices]] == y_test[indices])
        correct_rf = np.sum([np.argmax(x) for x in predictions_rf[indices]] == y_test[indices])
        correct_all = np.sum([np.argmax(x) for x in predictions_all[indices]] == y_test[indices])

        classes.append(sni)
        accuracies1.append(1. * correct1 / len(indices[0]))
        accuracies2.append(1. * correct2 / len(indices[0]))
        accuracies3.append(1. * correct3 / len(indices[0]))
        accuracies123.append(1. * correct123 / len(indices[0]))
        accuracies_rf.append(1. * correct_rf / len(indices[0]))
        accuracies_all.append(1. * correct_all / len(indices[0]))

    indices = np.arange(len(snis))
    width = np.min(np.diff(indices)) / 10
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(indices-width-width,accuracies1,width,color='b')
    ax.bar(indices-width,accuracies2,width,color='r')
    ax.bar(indices,accuracies3,width,color='g')
    ax.bar(indices+width,accuracies123,width,color='y')
    ax.bar(indices+width+width,accuracies_rf,width,color='m')
    ax.bar(indices+width+width+width,accuracies_all,width,color='c')
    ax.set_xlabel('SNIs')
    ax.set_ylabel('Accuracy')
    plt.show()

#*********************************************************************************** 
# function to save sklear report on precision and recall into a dictionary, 
# considering the history of cross validation
#***********************************************************************************
def update_stats(stats, model, predictions, y_test):
    report = classification_report(y_test, [np.argmax(x) for x in predictions])
    
    report_list = []
    for row in report.split("\n"):
        parsed_row = [x for x in row.split("  ") if len(x) > 0]
        if len(parsed_row) > 0:
            report_list.append(parsed_row)

    stats[model][0] += float(1. * np.sum([np.argmax(x) for x in predictions] == y_test) / len(y_test))
    stats[model][1] += float(report_list[-1][1])
    stats[model][2] += float(report_list[-1][2])
    stats[model][3] += float(report_list[-1][3])

    return stats

if __name__ == "__main__":
    kf = KFold(n_splits=FOLDS, shuffle=True)

    # try a variety of min conn settings for model
    statistics = [["model", "min connections", "accuracy", "precision", "recall", "f1_score"]]
    for min_connections in MIN_CONNECTIONS_LIST:
        datasetfile = "DL/training/GCDay1seq25.csv"
        X1, X2, X3, y, time_steps, n_features, n_labels, rev_class_map = dl_data_load_and_filter(datasetfile, min_connections)

        datasetfile = "ML/training/GCDay1stats.csv"
        X, _ = ml_data_load_and_filter(datasetfile, min_connections)

        stats = {}
        for model in ["Random Forest", "Baseline RNN", "Packet RNN", "Payload RNN", "IAT RNN", "Ensemble RNN", "Ensemble All"]:
            stats[model] = [0,0,0,0]

        for train_index, test_index in kf.split(X1):
            
            X1_train, X1_test = X1[train_index], X1[test_index]
            X2_train, X2_test = X2[train_index], X2[test_index]
            X3_train, X3_test = X3[train_index], X3[test_index]
            X_train, X_test = X[train_index], X[test_index]

            X1_train = np.reshape(X1_train, (np.shape(X1_train)[0], np.shape(X1_train)[1], n_features))
            X1_test = np.reshape(X1_test, (np.shape(X1_test)[0], np.shape(X1_test)[1], n_features))
            X2_train = np.reshape(X2_train, (np.shape(X2_train)[0], np.shape(X2_train)[1], n_features))
            X2_test = np.reshape(X2_test, (np.shape(X2_test)[0], np.shape(X2_test)[1], n_features))
            X3_train = np.reshape(X3_train, (np.shape(X3_train)[0], np.shape(X3_train)[1], n_features))
            X3_test = np.reshape(X3_test, (np.shape(X3_test)[0], np.shape(X3_test)[1], n_features))

            y_train, y_test = y[train_index], y[test_index]
            
            rf = MLClassification(X_train, X_test, y_train, y_test)
            predictions_rf = rf.predict_proba(X_test)
            stats = update_stats(stats, "Random Forest", predictions_rf, y_test)
            print("Random Forest ACCURACY: %s"%(stats["Random Forest"][0]))

            model_bl = BaselineDLClassification(X1_train, X1_test, y_train, y_test, time_steps, n_features, n_labels)
            predictions_bl = model_bl.predict(X1_test)
            stats = update_stats(stats, "Baseline RNN", predictions_bl, y_test)
            print("Baseline Recurrent Neural Net Packet ACCURACY: %s"%(stats["Baseline RNN"][0]))

            model1 = DLClassification(X1_train, X1_test, y_train, y_test, time_steps, n_features, n_labels, 0.0)
            predictions1 = model1.predict(X1_test)
            stats = update_stats(stats, "Packet RNN", predictions1, y_test)
            print("Recurrent Neural Net Packet ACCURACY: %s"%(stats["Packet RNN"][0]))

            model2 = DLClassification(X2_train, X2_test, y_train, y_test, time_steps, n_features, n_labels, 0.0)
            predictions2 = model2.predict(X2_test)
            stats = update_stats(stats, "Payload RNN", predictions2, y_test)
            print("Recurrent Neural Net Payload ACCURACY: %s"%(stats["Payload RNN"][0]))

            model3 = DLClassification(X3_train, X3_test, y_train, y_test, time_steps, n_features, n_labels, 0.25)
            predictions3 = model3.predict(X3_test)
            stats = update_stats(stats, "IAT RNN", predictions3, y_test)
            print("Recurrent Neural Net IAT ACCURACY: %s"%(stats["IAT RNN"][0]))

            predictions123 = (predictions1 * (1.0/3) + predictions2 * (1.0/3) + predictions3 * (1.0/3))
            stats = update_stats(stats, "Ensemble RNN", predictions123, y_test)
            print("Recurrent Neural Net Ensemble ACCURACY: %s"%(stats["Ensemble RNN"][0]))

            predictions_all = (predictions_rf * 0.5 + predictions123 * 0.5)
            stats = update_stats(stats, "Ensemble All", predictions_all, y_test)
            print("Ensemble All ACCURACY: %s"%(stats["Ensemble All"][0]))

            # get_classification_accuracies_image(predictions_rf, predictions1, predictions2, predictions3, predictions123, predictions_all)

            #Uncomment to run once
            FOLDS = 1
            break

        for model, stats in stats.items():
            statistics.append([model, min_connections] + [1. * x / FOLDS for x in stats])

        print(statistics)

        with open('final_results.csv', 'a') as file:
            wr = csv.writer(file)
            for statistic in statistics:
                wr.writerow(statistic)
            statistics = []
