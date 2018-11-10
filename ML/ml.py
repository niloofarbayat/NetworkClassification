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

####################################################
# Second Level models creator
####################################################
def Lv2modelsCreator(trainset, keyset, flimit):
    streamdict = defaultdict(list)
    models = defaultdict(list)
    dataset = zip(keyset, trainset)
    for row in dataset:
        streamdict[row[0]].append(row[1:44])
        randomforestmodel = RandomForestClassifier(n_estimators=10, n_jobs=10)

    for k in streamdict:
        perTLD = streamdict[k]
        #Full Feature set
        ktarget = np.array([x[0][flimit] for x in perTLD])
        ktrain = np.array([x[0][0:flimit] for x in perTLD])
        randomforestmodel.fit(ktrain, ktarget)
        s = pickle.dumps(randomforestmodel)
        models[k].append(s)
    return models

#############################################
# The Evaluation method
###############################################
def multiLevelEval(l2real, l2predict):
    total = 0
    partial = 0
    unknown = 0

    for i in range(0, len(l2real)):
        temp1 = tldextract.extract(l2real[i])
        temp2 = tldextract.extract(l2predict[i])
        realRD = temp1.domain + "." + temp1.suffix
        predictRD = temp2.domain + "." + temp2.suffix

        if l2real[i] == l2predict[i]:
            total=total+1.0

        elif realRD == predictRD:
            partial = partial + 1.0
    else:
        unknown = unknown + 1
    if realRD == "google.com" and predictRD == "gstatic.com":
        #partial=partial+1.0
        total = total + 1.0
    elif predictRD == "google.com" and realRD == "gstatic.com":
        #partial=partial+1.0
        total = total + 1.0

    print("[+] Partial : " + str(round(float(partial) / len(l2real),2)))
    print("[+] Full :" + str(round((float(total) / len(l2real)),2)))
    print("[+] Invalid :" + str(round((float(unknown) / len(l2real)),2)))
    return (float(total) / len(l2real))

#***********************************************************************************
# Multi-Level Classification 
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
# SNi modification for the sub-domain parts only
#***********************************************************************************

#TODO: Do we care about this?
def MultiLevelClassification(datasetfile, flimit):
    dataset=read_csv(datasetfile)
    result = []
    X = np.array([z[1:25] for z in dataset])
    y = np.array([z[0] for z in dataset])

    #TODO: need to filter out snis with few connections as with flat classification
    rf = RandomForestClassifier(n_estimators=250, n_jobs=10)
    kf = KFold(n_splits=10, shuffle=True)
    totalac = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        features = np.array([zp[0:flimit] for zp in X_train])
        features_test = np.array([zp[0:flimit] for zp in X_test])
        l2label = np.array([zp[flimit] for zp in X_test])
        Modles = Lv2modelsCreator(X_train, y_train, flimit)

        l2predict = []
        l2real = []

        streamdict = defaultdict(list)
        rf.fit(features, y_train)
        l1 = rf.predict(features_test)

        for i in range(0, len(l1)):
            streamdict[l1[i]].append(X_test[i])
        for k in streamdict:
            preTLD = streamdict[k]
            m = pickle.loads(Modles[k][0])
            feature = np.array([x[0:flimit] for x in preTLD])
            labels = np.array([x[flimit] for x in preTLD])
            l2 = m.predict(feature)
            l2predict = l2predict + l2.tolist()
            l2real = l2real + labels.tolist()
        totalac.append(multiLevelEval(l2real, l2predict))
        break

    return np.mean(totalac)
    
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
    #MultiLevelClassification("training/GCDay1stats.csv",42)

    # for graph
    min_connections_to_try = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
    accuracies = []
    for min_connections in min_connections_to_try:
        accuracies.append(MLClassification("training/GCDay1stats.csv", min_connections))

    plt.plot(min_connections_to_try, accuracies)
    plt.xlabel("Mininimum Connections")
    plt.ylabel("Accuracy")
    plt.show()





