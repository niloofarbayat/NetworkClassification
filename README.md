# NetworkClassification
### Overview

### Requirements
- tldextract
- matplotlib
- numpy
- sklearn
- tensorflow
- keras

### Preprocessing

##### create_pcap_stat.py
1. Specifications:
- Path to pcap file
- Path for output csv file with statistical features 
- Path for output csv file with sequence features 
2. Run with python3 create_pcap_stat.py
- Loads pcap files into memory by iterating through the HTTPs flow and grouping them on connection (see pytcpdump).
3. Calculate statistical features include the following:
- Packet size: num, 25th, 50th, 75th, max, avg, var (remote->local, local->remote, combined)
- Inter-arrival time: 25th, 50th, 75th (remote->local, local->remote, combined)
- Payload size: 25th, 50th, 75th, max, avg, var (remote->local, local->remote)
4. Calculate packet sequence features (specify first n packets to use)

##### pytcpdump.py
1. Loads one or more pcap files into memory by iterating through the HTTPs flow and grouping packets on connection (TCP only). 
2. Stores the following attributes for each connection in a cache: 
- SNI
- Accumulated bytes
- Arrival times
- Packet sizes
- Payload sizes. 

##### pytcpdump_utils.py
Utility functions for pytcpdump. Includes functions for parsing connection id, ip position, etc.

### ML
ML Folder contains ml.py, which runs Random Forest classification on statistical features from
the TCP handshake. High level summary can be broken down below:

1. Read CSV of the packet/payload/inter-arrival time features.
2. Filter for SNIs with a minimum number of connections.
3. Create a Random Forest Classifier and run 10-Fold Cross Validation for accuracy

### DL
DL Folder contains main.py which is responsible for predicting SNI using packet
sequences. High level summary can be broken down below:

1. Read CSV of the packet sequence captured.
2. It filters by a minimum number of connections that are consistent with thosed
used in the ML script.
3. Train_test_split is run to split the data into training and validation set.
4. It builds a Sequence Model (GRU or LSTM at the moment) and predicts the results.
