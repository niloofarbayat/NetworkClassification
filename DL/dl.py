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
#import autosklearn.classification
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler

BATCH_SIZE = 64
EPOCHS = 100 # use early stopping
FOLDS = 10
SEQ_LEN = 25
NUM_ROWS = 51554 # just use first day for now, set to -1 for all data

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
    
    print("Entering filtering section! ")
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

    ##### RESHAPE FOR LSTM #####
    #X = np.reshape(X, (n_samples, time_steps, n_features))
    return X1, X2, X3, y, time_steps, n_features, n_labels, rev_class_map
  
def create_model(time_steps, n_features, n_labels):
    model = Sequential()
    model.add(Conv1D(n_labels + 128, 3, activation='relu', input_shape=(time_steps, n_features)))
    model.add(BatchNormalization())
    model.add(Conv1D(n_labels + 256, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(GRU(n_labels + 100))
    model.add(Dense(n_labels + 50, activation='sigmoid')) 
    model.add(Dense(n_labels, activation='softmax')) 
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['acc'])
    model.summary()

    return model  

#########################################################
###### USE RNN TO CLASSIFY PACKET SEQUENCES -> SNI ######
#########################################################

def DLClassification(X_train, X_test, y_train, y_test,time_steps, n_features, n_labels):
    # if you dont have newest keras version, you might have to remove restore_best_weights = True
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='min', restore_best_weights=True)
    model = create_model(time_steps, n_features, n_labels)
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, shuffle=False, validation_data=(X_test, y_test), callbacks = [early_stopping])
    return model
  
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
    datasetfile = "training/GCDay1seq25.csv"

    # run w/min connections
    min_connections_to_try = [100]

    kf = KFold(n_splits=FOLDS, shuffle=True)

    # try a variety of min conn settings for graph
    accuracies = []
    for min_connections in min_connections_to_try:
        X1, X2, X3, y, time_steps, n_features, n_labels, rev_class_map = data_load_and_filter(datasetfile, min_connections)

        """
        scaler = Normalizer()
        scaler.fit(X1)
        X1 = scaler.transform(X1)
        scaler.fit(X2)
        X2 = scaler.transform(X2)
        """
        total_nn1, total_nn2, total_nn3, total_nn123, total_cls = 0, 0, 0, 0, 0
        for train_index, test_index in kf.split(X1):
            
            # Uncomment to just run once
            """
            if total_nn1 > 0:
                FOLDS = 1
                continue
            """

            X1_train, X1_test = X1[train_index], X1[test_index]
            X2_train, X2_test = X2[train_index], X2[test_index]
            X3_train, X3_test = X3[train_index], X3[test_index]

            X1_train = np.reshape(X1_train, (np.shape(X1_train)[0], np.shape(X1_train)[1], n_features))
            X1_test = np.reshape(X1_test, (np.shape(X1_test)[0], np.shape(X1_test)[1], n_features))
            X2_train = np.reshape(X2_train, (np.shape(X2_train)[0], np.shape(X2_train)[1], n_features))
            X2_test = np.reshape(X2_test, (np.shape(X2_test)[0], np.shape(X2_test)[1], n_features))
            X3_train = np.reshape(X3_train, (np.shape(X3_train)[0], np.shape(X3_train)[1], n_features))
            X3_test = np.reshape(X3_test, (np.shape(X3_test)[0], np.shape(X3_test)[1], n_features))

            y_train, y_test = y[train_index], y[test_index]
            
            model1 = DLClassification(X1_train, X1_test, y_train, y_test, time_steps, n_features, n_labels)

            model2 = DLClassification(X2_train, X2_test, y_train, y_test, time_steps, n_features, n_labels)

            model3 = DLClassification(X3_train, X3_test, y_train, y_test, time_steps, n_features, n_labels)

            predictions1 = model1.predict(X1_test)
            nn_acc1 = 1. * np.sum([np.argmax(x) for x in predictions1] == y_test) / len(y_test)
            print("Recurrent Neural Net Packet ACCURACY: %s"%(nn_acc1))

            predictions2 = model2.predict(X2_test)
            nn_acc2 = 1. * np.sum([np.argmax(x) for x in predictions2] == y_test) / len(y_test)
            print("Recurrent Neural Net Payload ACCURACY: %s"%(nn_acc2))

            predictions3 = model3.predict(X3_test)
            nn_acc3 = 1. * np.sum([np.argmax(x) for x in predictions3] == y_test) / len(y_test)
            print("Recurrent Neural Net IAT ACCURACY: %s"%(nn_acc3))

            predictions123 = (predictions1 * (1.0/3) + predictions2 * (1.0/3) + predictions3 * (1.0/3))
            nn_acc123 = 1. * np.sum([np.argmax(x) for x in predictions123] == y_test) / len(y_test)
            print("Recurrent Neural Net Ensemble ACCURACY: %s"%(nn_acc123))

            classes = []
            accuracies1 = []
            accuracies2 = []
            accuracies3 = []
            accuracies123 = []

            snis = np.unique(y_test)
            for sni in snis:
                indices = np.where(y_test == sni)
                correct1 = np.sum([np.argmax(x) for x in predictions1[indices]] == y_test[indices])
                correct2 = np.sum([np.argmax(x) for x in predictions2[indices]] == y_test[indices])
                correct3 = np.sum([np.argmax(x) for x in predictions3[indices]] == y_test[indices])
                correct123 = np.sum([np.argmax(x) for x in predictions123[indices]] == y_test[indices])
                classes.append(sni)
                accuracies1.append(1. * correct1 / len(indices[0]))
                accuracies2.append(1. * correct2 / len(indices[0]))
                accuracies3.append(1. * correct3 / len(indices[0]))
                accuracies123.append(1. * correct123 / len(indices[0]))

            indices = np.arange(len(snis))
            width = np.min(np.diff(indices)) / 6
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.bar(indices-width-width,accuracies1,width,color='b')
            ax.bar(indices-width,accuracies2,width,color='r')
            ax.bar(indices,accuracies3,width,color='g')
            ax.bar(indices+width,accuracies123,width,color='y')
            ax.set_xlabel('SNIs')
            ax.set_ylabel('Accuracy')

            # Uncomment this for SNI classification accuracies image
            plt.show()  


            
            total_nn1+= nn_acc1
            total_nn2+= nn_acc2
            total_nn3+= nn_acc3
            total_nn123+= nn_acc123
            
            # Uncomment for auto sklearn results on sequence features
            """
            cls_acc = auto_sklearn_classification(X_train, X_test, y_train, y_test)
            print("Auto sklearn Accuracy: %s "%(cls_acc))
            total_cls += cls_acc
            """

        total_nn1 = 1. * total_nn1 / FOLDS
        total_nn2 = 1. * total_nn2 / FOLDS
        total_nn3 = 1. * total_nn3 / FOLDS
        total_nn123 = 1. * total_nn123 / FOLDS
        total_cls = 1. * total_cls / FOLDS

        print("AVG RNN Packet: %s\n AVG RNN Payload: %s\n AVG RNN IAT: %s\n AVG RNN Ensemble: %s\n AVG CLS: %s\n "%(total_nn1, total_nn2, total_nn3, total_nn123, total_cls))
  
        accuracies.append([total_nn1, total_nn2, total_nn3, total_nn123, total_cls])

    """
    plt.plot(min_connections_to_try, accuracies)
    plt.xlabel("Mininimum Connections")
    plt.ylabel("Accuracy")
    plt.show()
    """

    print(accuracies)


