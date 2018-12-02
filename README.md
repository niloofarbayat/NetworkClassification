# NetworkClassification
### Overview

### Requirements
- Specified in `requirements.txt`
- Python 3.6

### Preprocessing

##### `create_pcap_stat.py`

Specifications:

1. Path to pcap file(s)
2. Path for output csv file with statistical features 
3. Path for output csv file with sequence features 

Run with python3.6 `create_pcap_stat.py`

Program loads pcap files into memory by iterating through the HTTPs flow and grouping them on connection (see pytcpdump).
Calculate statistical features that include the following:

1. Packet size: num, 25th, 50th, 75th, max, avg, var (remote->local, local->remote, combined)
2. Inter-arrival time: 25th, 50th, 75th (remote->local, local->remote, combined)
3. Payload size: 25th, 50th, 75th, max, avg, var (remote->local, local->remote)

Get sequence features with padding (specify first n packets to use):

1. Packet sizes
2. Payload sizes
3. Inter-arrival times
4. Directionality

##### `pytcpdump.py`

Loads one or more pcap files into memory by iterating through the HTTPs flow and grouping packets on connection (TCP only). 
Stores the following attributes for each connection in a cache: 

1. SNI
2. Accumulated bytes
3. Arrival times
4. Packet sizes
5. Payload sizes. 

##### `pytcpdump_utils.py`
Utility functions for pytcpdump. Includes functions for parsing connection id, ip position, etc.

### ML
ML Folder contains ml.py, which runs Random Forest classification on statistical features from
the TCP handshake. High level summary can be broken down below:

1. Read CSV (`training/GCstats.csv`) of the packet/payload/inter-arrival time statistical features.
2. Filter for SNIs meeting a minimum number of connections.
3. Create a Random Forest Classifier and run 10-Fold Cross Validation for accuracy.
4. (Optional) Create Auto-Sklearn classifier and run 10-Fold Cross Validation for accuracy.

### DL
DL Folder contains `dl.py` which is responsible for predicting SNI using sequence data. High level 
summary broken down below:

1. Read CSV (`training/GCseq25.csv`) of the packet/payload/inter-arrival time sequence features.
2. Filter for SNIs meeting a minimum number of connections.
3. Create three CNN-RNNs (one for each feature sequence) and run 10-Fold Cross Validation.
4. Get accuracy results for each CNN-RNNs, as well as an ensemble classifier
5. (Optional) Create Auto-Sklearn classifier and run 10-Fold Cross Validation for accuracy.

### `main.py`
Reads in CSV files (`ML/training/GCstats.csv`, `DL/training/GCseq25.csv`). Filters SNIs meeting a minimum
number of connections. Creates the following classifiers: 

1. Random Forest
2. Baseline RNN trained on packet size sequences
3. CNN-RNN trained on packet size sequences
4. CNN-RNN trained on payload size sequences
5. CNN-RNN trained on inter-arrival time sequences
6. Ensemble CNN-RNN
7. Ensemble CNN-RNN + Random Forest

Writes accuracy results to `final_results.csv`. (Optional) Can also write per-SNI class results to `class_results.csv`.

