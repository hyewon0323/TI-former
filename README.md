# TI-former
TI-former: A Time-Interval Prediction Transformer for Timestamped Sequences, SERA 2023

Abstract—The Transformer is a widely used neural network architecture for natural language processing. Recently, it has been applied to time series prediction tasks. However, the vanilla transformer has a critical limitation in that it cannot predict the time intervals between elements. To overcome this limitation, we propose a new model architecture called TI-former (Time Interval Transformers) that predicts both the sequence elements and the time intervals between them. To incorporate the elements’ sequential order and temporal interval information, first we propose a new positional encoding method. Second, we modify the output layer to predict both the next sequence element and the time interval simultaneously. Lastly, we suggest a new loss function for timestamped sequences, namely Time soft-DTW, which measures similarity between sequences considering timestamps. We present experimental results based on synthetic sequence data. The experimental results show that our proposed model outperforms than vanilla transformer model in various sequence lengths, sequence numbers, and element occurrence time ranges.


<img src = https://github.com/hyewon0323/TI-former/assets/85382585/8f7f1efc-59f8-4a2a-a732-971011b826e5 width="40%" height="60%">
