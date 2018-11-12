import numpy as np
import sklearn.ensemble, sklearn.model_selection
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

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

#***********************************************************************************
# Flat Classification 
#
# This function borrowed from the following paper:
#
# Multi-Level identification Framework to Identify HTTPS Services
# Author by Wazen Shbair,
# University of Lorraine,
# France
# wazen.shbair@gmail.com
# January, 2017
#
# SNi prediction using Random Forest Classifier
#***********************************************************************************
def MLClassification(datasetfile, min_connections):
    X, y = data_load_and_filter(datasetfile, min_connections)

    rf = RandomForestClassifier(n_estimators=250, n_jobs=10)
    kf = KFold(n_splits=10, shuffle=True)

    total = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        rf.fit(X_train, y_train)
        l1 = rf.predict(X_test)
        accuracy = accuracy_score(l1,y_test)
        print("ACCURACY: ", accuracy)
        total+= accuracy

    print("AVG: ", 1. * total / 10)
    return 1. * total / 10


if __name__ == "__main__":

    # run once
    #MLClassification("training/GCDay1stats.csv", 100)

    # for graph
    min_connections_to_try = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
    accuracies = []
    for min_connections in min_connections_to_try:
        accuracies.append(MLClassification("training/GCDay1stats.csv", min_connections))

    plt.plot(min_connections_to_try, accuracies)
    plt.xlabel("Mininimum Connections")
    plt.ylabel("Accuracy")
    plt.show()





