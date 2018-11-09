# NetworkClassification
### Overview

### Requirements
- tldextract
- matplotlib
- numpy
- sklearn

### Running
1. Edit create_pcap_stat.py
- path to pcap file
- path for output csv file with statistical features 
- path for the output csv file with sequence features 
2. python3 create_pcap_stat.py
3. Edit ml.py
- Specify path to csv file with statistical features
- Set desired algorithm
4. python3 ml.py


### DL Folder

DL Folder contains main.py which is responsible for predicting SNI using packet
sequences. High level summary can be broken down below:

1. Read CSV of the packet sequence captured.
2. It filters by a minimum number of connections that are consistent with thosed
used in the ML script.
3. Train_test_split is run to split the data into training and validation set.
4. It builds a Sequence Model (GRU or LSTM at the moment) and predicts the results.
