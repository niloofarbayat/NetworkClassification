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
from keras.callbacks import History
import matplotlib.pyplot as plt
#from attention_decoder import AttentionDecoder
import autosklearn.classification


BATCH_SIZE = 32
EPOCHS = 10
FOLDS = 10   


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

    ##### BASIC PARAMETERS #####
    n_samples = np.shape(X)[0]
    time_steps = np.shape(X)[1] # we have a time series of 100 payload sizes
    n_features = 1 # 1 feature which is packet size
    seq_len = 100

    ##### CREATES MAPPING FROM SNI STRING TO INT #####
    class_map = {sni:i for i, sni in enumerate(np.unique(y))}
    rev_class_map = {val: key for key, val in class_map.items()}

    n_labels = len(class_map)

    ##### CHANGE Y TO PD SO ITS EASIER TO MAP #####
    y_pd = pd.DataFrame(y)
    y_pd = y_pd[0].map(class_map)

    ##### DUPLICATE Y LABELS, WE WILL NEED THIS LATER #####
    y = y_pd.values.reshape(n_samples,)

    ##### CREATE A NEW SEQUENCE ARRAY OF 0s THAT ARE INTS #####
    sequences = np.zeros((len(X), seq_len), dtype=int)
    
    ##### COPY X_TRAIN INTO THE SEQUENCES BUT THIS TIME IT'LL ALL BE INTS #####
    for i, row in enumerate(X):
        line = np.array(row)
        sequences[i, -len(row):] = line[-seq_len:]

    ##### REPLACE X_TRAIN WITH THE NEW INT ARRAY #####
    X = sequences

    ##### RESHAPE FOR LSTM #####
    #X = np.reshape(X, (n_samples, time_steps, n_features))
    return X, y, time_steps, n_features, n_labels

  
  
def create_model(time_steps, n_features, n_labels):
    model = Sequential()

    model.add(GRU(100+n_labels, return_sequences=True, input_shape=(time_steps, n_features)))
    model.add(GRU(100+n_labels))
    model.add(Dense(n_labels, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['acc'])
    model.summary()

    return model  

#########################################################
###### USE RNN TO CLASSIFY PACKET SEQUENCES -> SNI ######
#########################################################

def DLClassification(X_train, X_test, y_train, y_test,time_steps, n_features, n_labels):
    X_train = np.reshape(X_train, (np.shape(X_train)[0], np.shape(X_train)[1], n_features))
    X_test = np.reshape(X_test, (np.shape(X_test)[0], np.shape(X_test)[1], n_features))

    history = History()

    model = create_model(time_steps, n_features, n_labels)
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, shuffle=False, validation_data=(X_test, y_test), callbacks = [history])

    accuracy = history.history['val_acc'][-1]
    return accuracy
  
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
    # run once w/min connections = 100 as in paper
    datasetfile = "training/GCDay1seq100.csv"
    min_connections_to_try = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]

    kf = KFold(n_splits=FOLDS, shuffle=True)

    # try a variety of min conn settings for graph
    accuracies = []
    for min_connections in min_connections_to_try:
        X, y, time_steps, n_features, n_labels = data_load_and_filter(datasetfile, min_connections)
        total_nn, total_cls = 0, 0
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            nn_acc = DLClassification(X_train, X_test, y_train, y_test,time_steps, n_features, n_labels)
            print("Neural Net ACCURACY: %s"%(nn_acc))
            
            cls_acc = auto_sklearn_classification(X_train, X_test, y_train, y_test)
            print("Auto sklearn Accuracy: %s "%(cls_acc))
            
            total_nn+= nn_acc
            total_cls += cls_acc


        total_nn, total_cls = 1. * total_nn / FOLDS, 1. * total_cls / FOLDS 
        print("AVG Neural Net: %s, AVG CLS: %s "%(total_nn, total_cls))
  
        accuracies.append([total_nn, total_cls])

    '''
    plt.plot(min_connections_to_try, accuracies)
    plt.xlabel("Mininimum Connections")
    plt.ylabel("Accuracy")
    plt.show()
    '''
    
    print(accuracies)


