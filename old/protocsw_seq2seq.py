import numpy as np
import tensorflow as tf
from cswNets import *
# print('\nmulti GPU version\n')
# from cswNets_gpu import *


## PROTO CSW

# transition graph


graph_dict = {
  0:[1,2],
  1:[3,4],
  2:[4,3],
  3:[5,6],
  4:[6,5],
  5:[7,7],
  6:[7,7]
}


# each path makes a single training example

def gen_path(graph_dict,pr):
  """ graph is a dict{A:[B,C]}
  A is current state B is transitioned to with pr"""
  path = []
  st = 0
  while st < 7:
    path.append(st)
    if np.random.binomial(1,pr):
      st1 = graph_dict[st][0]
    else:
      st1 = graph_dict[st][1]
    st = st1
  return path

def gen_xy(path,filler_id):
  """ given a path through the graph e.g. [A,C] 
  make set of training examples ([A,x],[C,x])
  where A, is state and x is filler_id """
  X = []
  Y = []
  for i in range(len(path)-1):
    x,y = path[i:i+2]  
    X.extend([x,filler_id])
    Y.extend([y,filler_id])
  return np.array(X),np.array(Y)

def gen_m_samples(m,graph_dict,transition_pr,filler_id):
  X = []
  Y = []
  for itr in range(m):
    path = gen_path(graph_dict,transition_pr)
    x,y = gen_xy(path,filler_id)
    X.append(x)
    Y.append(y)
  X = np.array(X)
  Y = np.array(Y)
  return X,Y


### DEFINE TASK


## data
RAND = False
COND = False

print('\n','RAND',RAND,'COND',COND,'\n')


if RAND:
  fix_vocab = 8
  rand_vocab = 2
else:
  fix_vocab = 10
  rand_vocab = 0

if COND:
  X1,Y1 = gen_m_samples(m=50,graph_dict=graph_dict,transition_pr=.8,filler_id=8)
  X2,Y2 = gen_m_samples(m=50,graph_dict=graph_dict,transition_pr=.2,filler_id=9)
  X = np.vstack([X1,X2])
  Y = np.vstack([Y1,Y2])
else:
  X,Y = gen_m_samples(m=100,graph_dict=graph_dict,transition_pr=.8,filler_id=8)


# setup
vocab_size = fix_vocab+rand_vocab
arch = {'input_seq_len':2,
       'output_seq_len':2,
       'num_sequences': 3,
       'fix_vocab_size': fix_vocab,
       'rand_vocab_size': rand_vocab,
       'netdim': 2*vocab_size}

# train

train_info = {'batch_size': 10,
              'num_epochs': 15000,
              'loss_op': tf.losses.mean_squared_error}

train_data = {'X_train':X,'Y_train':Y,
              'X_test':X,'Y_test':Y}

lstm = LSTMseq2(arch,saving=True)
rnn = RNNseq2(arch,saving=True)

lstm_exp_data = run_k_experiments(lstm,train_data,train_info,1)
rnn_exp_data = run_k_experiments(rnn,train_data,train_info,1)

