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
    dataset = dataset[:50000]

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
# Flat Classification: SNi prediction using Random Forest Classifier
#
# This function includes some code from the following paper:
#
# Multi-Level identification Framework to Identify HTTPS Services
# Author by Wazen Shbair,
# University of Lorraine,
# France
# wazen.shbair@gmail.com
# January, 2017 
#***********************************************************************************
def MLClassification(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=250, n_jobs=10)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    accuracy = accuracy_score(predictions,y_test)

    """
    class_map = {sni:i for i, sni in enumerate(np.unique(y_test))}
    rev_class_map = {val: key for key, val in class_map.items()}
    snis = np.unique(y_test)
    
    classes = []
    accuracies = []
    for sni in snis:
        indices = np.where(y_test == sni)
        correct = np.sum(predictions[indices] == y_test[indices])
        classes.append(class_map[sni])
        accuracies.append(1. * correct / len(indices[0]))

    plt.bar(classes, accuracies)
    """

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
# autosklearn classifier to find the best achievable accuracy
#***********************************************************************************
def auto_sklearn_classification(X_train, X_test, y_train, y_test):
  cls = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=300, per_run_time_limit=90, ml_memory_limit=10000)
  cls.fit(X_train, y_train)
  predictions = cls.predict(X_test)
  accuracy = accuracy_score(predictions,y_test)
  return accuracy
  
if __name__ == "__main__":
    
    folds = 10
    datasetfile = "training/GCstats.csv"
    # run once
    #MLClassification("training/GCstats.csv", 100)

    # for graph
    min_connections_to_try = [100]
    
    kf = KFold(n_splits=folds, shuffle=True)
    for min_connections in min_connections_to_try:
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

            """
            accuracy, precision, recall, f1_score = auto_sklearn_classification(X_train, X_test, y_train, y_test)
            print("Auto sklearn Accuracy: %s "%(cls_acc))
            
            total_cls[0] += accuracy
            total_cls[1] += precision
            total_cls[2] += recall
            total_cls[3] += f1_score
            """
            
        print("AVG RF: %s, AVF CLS: %s "%(1. * total_rf[0] / folds, 1. * total_cls[0] / folds))
      
        with open('precision%s.csv'%(min_connections), 'a') as f:
            accuracy = total_rf[0] / folds
            precision = total_rf[1] / folds
            recall = total_rf[2] / folds  
            f1_score = total_rf[3] / folds 
            writer = csv.writer(f)
            writer.writerow([accuracy, precision, recall, f1_score])