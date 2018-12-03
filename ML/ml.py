# Random Forest versus autosklearn classifier
#***********************************************************************************

import numpy as np
import sklearn.ensemble, sklearn.model_selection
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import autosklearn.classification
import pandas
from sklearn.metrics import classification_report
import csv   
from collections import defaultdict

MIN_CONNECTIONS_LIST = [100]
FOLDS = 10
NUM_ROWS = -1 # set to -1 for all data

def read_csv(file_path, has_header=True):
    with open(file_path) as f:
        if has_header: f.readline()
        data = []
        for line in f:
            line = line.strip().split(",")
            data.append([x for x in line])
    return data

#***********************************************************************************
# Filter for SNIs meeting min connection threshold
#***********************************************************************************
def data_load_and_filter(datasetfile, min_connections):
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

#***********************************************************************************
# SNI prediction using Random Forest Classifier
#***********************************************************************************
def MLClassification(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=250, n_jobs=10)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    accuracy = accuracy_score(predictions,y_test)

    report = []
    report_str = classification_report(y_test, predictions)
    for row in report_str.split("\n"):
        parsed_row = [x for x in row.split("  ") if len(x) > 0]
        if len(parsed_row) > 0:
            report.append(parsed_row)
    
    precision = float(report[-1][1])
    recall = float(report[-1][2])
    f1_score = float(report[-1][3])
    return accuracy, precision, recall, f1_score


#***********************************************************************************
# Autosklearn classifier to find the best achievable accuracy
#***********************************************************************************
def auto_sklearn_classification(X_train, X_test, y_train, y_test):
    cls = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=300, per_run_time_limit=90, ml_memory_limit=10000)
    cls.fit(X_train, y_train)
    predictions = cls.predict(X_test)
    accuracy = accuracy_score(predictions,y_test)

    report = []
    report_str = classification_report(y_test, predictions)
    for row in report_str.split("\n"):
        parsed_row = [x for x in row.split("  ") if len(x) > 0]
        if len(parsed_row) > 0:
            report.append(parsed_row)
    
    precision = float(report[-1][1])
    recall = float(report[-1][2])
    f1_score = float(report[-1][3])
    return accuracy, precision, recall, f1_score
  
if __name__ == "__main__":
    
    datasetfile = "training/GCstats.csv"

    kf = KFold(n_splits=FOLDS, shuffle=True)
    for min_connections in MIN_CONNECTIONS_LIST:
        X, y = data_load_and_filter(datasetfile, min_connections)
        total_rf, total_cls = [0,0,0,0], [0,0,0,0]

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            accuracy, precision, recall, f1_score = MLClassification(X_train, X_test, y_train, y_test)
            print("Random Forest ACCURACY: %s"%(accuracy))
            total_rf[0] += accuracy
            total_rf[1] += precision
            total_rf[2] += recall
            total_rf[3] += f1_score

            accuracy, precision, recall, f1_score = auto_sklearn_classification(X_train, X_test, y_train, y_test)
            print("Auto sklearn ACCURACY: %s "%(accuracy))
            
            total_cls[0] += accuracy
            total_cls[1] += precision
            total_cls[2] += recall
            total_cls[3] += f1_score
            
            # Uncomment to run once
            FOLDS = 1
            break

        print("AVG Random Forest: %s, AVG Auto-Sklearn: %s "%(1. * total_rf[0] / FOLDS, 1. * total_cls[0] / FOLDS))