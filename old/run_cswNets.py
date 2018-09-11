import numpy as np
import tensorflow as tf
from cswNets import *


# X = [[1,0],[0,1],[1,1],[0,0]]; X = np.array(X)
# Y = [[2],[2],[3],[3]]; Y = np.array(Y)


# read ptb dataset
import ptbExamples.ptb.reader as ptbreader
train_data, valid_data, test_data, vocabulary = ptbreader.ptb_raw_data('ptbExamples/data')
num_time_steps = 2
Xtrain,Ytrain,vocab_size = slice_and_stride(train_data,num_time_steps)

## setup architecture
arch = {'num_time_steps':num_time_steps,
        'celldim': 5,
        'vocab_size':vocab_size,
        'embed_size': 5}

lstm = LSTM(arch,saving=True)

## training 
train_info = {'batch_size': 1000,
              'num_epochs': 100,
              'loss_op': tf.losses.mean_squared_error}

eval_data = lstm.train(Xtrain,Ytrain,train_info)
