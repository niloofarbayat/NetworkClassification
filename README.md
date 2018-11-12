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
- Specifications:
1. Path to pcap file
2. Path for output csv file with statistical features 
3. Path for output csv file with sequence features 
- Run with python3 create_pcap_stat.py
- Program loads pcap files into memory by iterating through the HTTPs flow and grouping them on connection (see pytcpdump).
- Calculate statistical features that include the following:
1. Packet size: num, 25th, 50th, 75th, max, avg, var (remote->local, local->remote, combined)
2. Inter-arrival time: 25th, 50th, 75th (remote->local, local->remote, combined)
3. Payload size: 25th, 50th, 75th, max, avg, var (remote->local, local->remote)
- Calculate packet sequence features (specify first n packets to use)

##### pytcpdump.py
- Loads one or more pcap files into memory by iterating through the HTTPs flow and grouping packets on connection (TCP only). 
- Stores the following attributes for each connection in a cache: 
1. SNI
2. Accumulated bytes
3. Arrival times
4. Packet sizes
5. Payload sizes. 

##### pytcpdump_utils.py
- Utility functions for pytcpdump. Includes functions for parsing connection id, ip position, etc.

### ML
- ML Folder contains ml.py, which runs Random Forest classification on statistical features from
the TCP handshake. High level summary can be broken down below:

1. Read CSV of the packet/payload/inter-arrival time features.
2. Filter for SNIs with a minimum number of connections.
3. Create a Random Forest Classifier and run 10-Fold Cross Validation for accuracy

### DL
- DL Folder contains main.py which is responsible for predicting SNI using packet
sequences. High level summary can be broken down below:

1. Read CSV of the packet sequence captured.
2. It filters by a minimum number of connections that are consistent with thosed
used in the ML script.
3. Train_test_split is run to split the data into training and validation set.
4. It builds a Sequence Model (GRU or LSTM at the moment) and predicts the results.
