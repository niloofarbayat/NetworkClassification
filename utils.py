import numpy as np
import pandas as pd
import csv
from sklearn.metrics import classification_report

#*********************************************************************************** 
# Utility function to write accuracies per class to a CSV, for a variety of classifiers
#***********************************************************************************
def output_class_accuracies(rev_class_map, predictions_rf, predictions_bl, predictions1, predictions2, predictions3, predictions123, predictions_best):
    classes = []
    accuracies_rf = []
    accuracies_bl = []
    accuracies1 = []
    accuracies2 = []
    accuracies3 = []
    accuracies123 = []
    accuracies_all= []

    snis = np.unique(y_test)
    for sni in snis:
        indices = np.where(y_test == sni)
        correct_rf = np.sum([np.argmax(x) for x in predictions_rf[indices]] == y_test[indices])
        correct_bl = np.sum([np.argmax(x) for x in predictions_bl[indices]] == y_test[indices])
        correct1 = np.sum([np.argmax(x) for x in predictions1[indices]] == y_test[indices])
        correct2 = np.sum([np.argmax(x) for x in predictions2[indices]] == y_test[indices])
        correct3 = np.sum([np.argmax(x) for x in predictions3[indices]] == y_test[indices])
        correct123 = np.sum([np.argmax(x) for x in predictions123[indices]] == y_test[indices])
        correct_all = np.sum([np.argmax(x) for x in predictions_best[indices]] == y_test[indices])

        classes.append(rev_class_map[sni])
        accuracies_rf.append(1. * correct_rf / len(indices[0]))
        accuracies_bl.append(1. * correct_bl / len(indices[0]))
        accuracies1.append(1. * correct1 / len(indices[0]))
        accuracies2.append(1. * correct2 / len(indices[0]))
        accuracies3.append(1. * correct3 / len(indices[0]))
        accuracies123.append(1. * correct123 / len(indices[0]))
        accuracies_all.append(1. * correct_all / len(indices[0]))

    with open('class_results.csv', 'w') as file:
        wr = csv.writer(file)
        wr.writerow([' '] + classes)
        wr.writerow(['Random Forest'] + accuracies_rf)
        wr.writerow(['Baseline RNN'] + accuracies_bl)
        wr.writerow(['Packet CNN-RNN'] + accuracies1)
        wr.writerow(['Payload CNN-RNN'] + accuracies2)
        wr.writerow(['IAT CNN-RNN'] + accuracies3)
        wr.writerow(['Ensemble CNN-RNN'] + accuracies123)
        wr.writerow(['Ensemble RF + CNN-RNN'] + accuracies_all)

def read_csv(file_path, has_header=True):
    with open(file_path) as f:
        if has_header: f.readline()
        data = []
        for line in f:
            line = line.strip().split(",")
            data.append([x for x in line])
    return data


#*********************************************************************************** 
# Filter the data set using the minimum connections filter
#***********************************************************************************
def data_load_and_filter(datasetfile, min_connections, NUM_ROWS=-1):
    dataset = read_csv(datasetfile)
    
    # Use first n rows if necessary
    dataset = dataset[:NUM_ROWS]

    X = np.array([z[1:] for z in dataset])
    y = np.array([z[0] for z in dataset])
    print("Shape of X =", np.shape(X))
    print("Shape of y =", np.shape(y))     
    
    print("Entering min connections filter section! ")
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
# Function separate out input data into packet, payload, IAT, direction sequences
#
# len(X[0]) = 100
# X_1...X_25 = Packet Sizes
# X_26...X_50 = Payload Sizes
# X_51...X_75 = Inter-Arrival Times
# X_76...X_100 = Directional Features
#***********************************************************************************
def process_dl_features(X, y, SEQ_LEN=25):
    # packet, payload, IAT, direction
    X1 = X[:,:SEQ_LEN]
    X2 = X[:,SEQ_LEN:2*SEQ_LEN]
    X3 = X[:,2*SEQ_LEN:3*SEQ_LEN]
    X4 = X[:,3*SEQ_LEN:4*SEQ_LEN]

    X3[np.where(X3 != 0 )] = np.log(X3[np.where(X3 != 0 )])

    print("Filtered shape of X1 =", np.shape(X1))
    print("Filtered shape of X2 =", np.shape(X2))
    print("Filtered shape of X3 =", np.shape(X3))
    print("Filtered shape of X4 =", np.shape(X4))
    print("Filtered shape of y =", np.shape(y))   

    ##### BASIC PARAMETERS #####
    n_samples = np.shape(X1)[0]
    time_steps = np.shape(X1)[1] # we have a time series of 100 payload sizes

    ##### CREATES MAPPING FROM SNI STRING TO INT #####
    class_map = {sni:i for i, sni in enumerate(np.unique(y))}
    rev_class_map = {val: key for key, val in class_map.items()}

    n_labels = len(class_map)

    ##### CHANGE Y TO PD SO ITS EASIER TO MAP #####
    y_pd = pd.DataFrame(y)
    y_pd = y_pd[0].map(class_map)

    ##### DUPLICATE Y LABELS, WE WILL NEED THIS LATER #####
    y = y_pd.values.reshape(n_samples,)

    return X1, X2, X3, X4, y, time_steps, n_labels, rev_class_map


#*********************************************************************************** 
# Function to save sklearn report on precision, recall, F1-Score into a dictionary
#***********************************************************************************
def update_stats(stats, model, predictions, y_test):
    report = classification_report(y_test, [np.argmax(x) for x in predictions])
    
    report_list = []
    for row in report.split("\n"):
        parsed_row = [x for x in row.split("  ") if len(x) > 0]
        if len(parsed_row) > 0:
            report_list.append(parsed_row)

    # save accuracy, precision, recall, F1-Score to dictionary
    stats[model][0] += float(1. * np.sum([np.argmax(x) for x in predictions] == y_test) / len(y_test)) 
    stats[model][1] += float(report_list[-1][1])
    stats[model][2] += float(report_list[-1][2])
    stats[model][3] += float(report_list[-1][3])

    return stats