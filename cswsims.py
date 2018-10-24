import numpy as np
import tensorflow as tf
import RNNs
from CSW import CSWTask 

""" GOAL

WORKFLOW: train & eval, save and restore data, analyze. 

separating RNNs allow me to modularize different tasks
separating trainers allows me to not have to rebuild graph every time 

NB depth == unroll_depth
"""


NUM_CLASSES = 12
IN_LEN = 1
OUT_LEN = 1
DEPTH = 5 # must be less than number of samples in path 

class NetGraph():

  def __init__(self,rnn_size):
    """
    """
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph)
    # dimensions
    self.rnn_size = rnn_size
    self.embed_dim = rnn_size
    self.depth = DEPTH 
    self.in_len = IN_LEN
    self.out_len = OUT_LEN
    self.num_classes = NUM_CLASSES
    # build
    self.RNN = RNNs.basicRNN
    self.build()

  def build(self):
    with self.graph.as_default():
      # place holders
      self.setup_placeholders()
      # cell and task
      self.cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.rnn_size,dropout_keep_prob=self.dropout_keep_prob)
      
      # pipeline
      self.xbatch_id,self.ybatch_id = self.data_pipeline() # x(batches,bptt,in_tstep), y(batch,bptt,out_tstep)
      ## inference
      self.embed_mat = tf.get_variable('embedding_matrix',[self.num_classes,self.embed_dim])
      self.xbatch = tf.nn.embedding_lookup(self.embed_mat,self.xbatch_id,name='xembed') # batch,bptt,in_len,in_dim
      self.unscaled_logits = self.RNN(self,self.depth,self.in_len,self.out_len) # batch,bptt*out_len,num_classes
      # awkward syntax allows me to modularize RNNs
      ## loss
      self.ybatch_onehot = tf.one_hot(indices=self.ybatch_id,depth=self.num_classes) # batch,bptt,out_len,num_classes
      self.train_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                          labels=self.ybatch_onehot,logits=self.unscaled_logits)
      self.minimizer_op = tf.train.GradientDescentOptimizer(0.005).minimize(self.train_loss)
      ## accuracy
      self.yhat_sm = tf.nn.softmax(self.unscaled_logits)
      self.yhat_id = tf.argmax(self.yhat_sm,-1)
      # self.acc_op = setup_acc_op(self.yhat_id,self.ybatch_id)
      ## extra
      self.sess.run(tf.global_variables_initializer())
      self.saver_op = tf.train.Saver()

  def setup_placeholders(self):
    self.xph = xph = tf.placeholder(tf.int32,
                  shape=[None,self.depth,self.in_len],
                  name="xdata_placeholder")
    self.yph = yph = tf.placeholder(tf.int32,
                  shape=[None,self.depth,self.out_len],
                  name="ydata_placeholder")
    self.batch_size_ph = tf.placeholder(tf.int64,
                  shape=[],
                  name="batchsize_placeholder")
    self.dropout_keep_prob = tf.placeholder(tf.float32,
                                shape=[],
                                name="dropout_ph")
    self.embedding_switch = tf.placeholder(tf.int64,
                                shape=[],
                                name="embedding_switch")
    return None

  def data_pipeline(self):
    """
    setup data iterator pipeline
    creates self.itr_initop and self.dataset
    returns x,y = get_next
    """
    dataset = tf.data.Dataset.from_tensor_slices((self.xph,self.yph))
    # dataset = dataset.apply(
    #             tf.contrib.data.sliding_window_batch(
    #               window_size=DEPTH,stride=1)
    #             )
    dataset = dataset.batch(self.batch_size_ph)
    dataset = self.dataset = dataset.shuffle(100000)

    iterator = tf.data.Iterator.from_structure(
                dataset.output_types, dataset.output_shapes)
    xbatch,ybatch = iterator.get_next() 

    self.itr_initop = iterator.make_initializer(dataset)
    return xbatch,ybatch

  def reinitialize(self):
    print('**reinitializing weights**')
    with self.graph.as_default():
      self.sess.run(tf.global_variables_initializer())
    return None



class Trainer():

  def __init__(self,csw_net,graphpr):
    self.net = csw_net
    self.graphpr = graphpr
    self.embed_switch = 0
    return None

  # steps: single pass through dataset

  def train_step(self,Xtrain,Ytrain):
    """ updates model parameters using Xtrain,Ytrain
    """
    TRAIN_BATCH_SIZE = 1
    # initialize iterator with train data
    train_feed_dict = {
      self.net.xph: Xtrain,
      self.net.yph: Ytrain,
      self.net.batch_size_ph: TRAIN_BATCH_SIZE,
      self.net.dropout_keep_prob: .9,
      self.net.embedding_switch: self.embed_switch
      }
    self.net.sess.run([self.net.itr_initop],train_feed_dict)
    # train loop
    while True:
      try:
        _ = self.net.sess.run([self.net.minimizer_op],feed_dict=train_feed_dict)
        # print(loss)
      except tf.errors.OutOfRangeError:
        break
    return None

  def predict_step(self,Xpred,Ypred):
    """ makes predictions on full dataset
    currently predictions are made on both contexts using same embedding
    ideally i could make predictions on each context independently 
    so that could make predictions with the appropriate embedding
    e.g. include an argument in predict which says which Xdata to pass
      then pass XfullA with embed_switch 0, XfullB with embed_switch 1
    """
    pred_batch_size = len(Xpred)

    # initialize data datastructures
    pred_array_dtype = [('xbatch','int32',(pred_batch_size,self.net.depth,self.net.in_len)),
                        ('yhat','float32',(pred_batch_size,self.net.depth,self.net.out_len,self.net.num_classes)),
    ]
    pred_data_arr = np.zeros((),dtype=pred_array_dtype)

    # feed dict
    pred_feed_dict = {
      self.net.xph:Xpred,
      self.net.yph:Ypred,
      self.net.batch_size_ph: pred_batch_size,
      self.net.dropout_keep_prob: 1.0,
      self.net.embedding_switch: self.embed_switch
    }
    # initialize iterator with eval data
    self.net.sess.run(self.net.itr_initop,pred_feed_dict)
    # eval loop
    while True:
      try:
        xbatch,yhat = self.net.sess.run([self.net.xbatch_id,self.net.yhat_sm],
                                       feed_dict=pred_feed_dict)
        pred_data_arr['xbatch'] = xbatch
        pred_data_arr['yhat'] = yhat
      except tf.errors.OutOfRangeError:
        break 
    return pred_data_arr

  # blocks and sessions

  def train_block(self,num_epochs,csw_graph):
    """ 
    single block of training. e.g. training within single context.
    """
    num_evals = np.maximum(1,epochs_per_block/10)
    block_pred_data = np.zeros(num_evals)
    eval_idx = -1
    ## train loop
    for epoch_num in range(epochs_per_block): 
      # generate train data, update params
      Xtrain,Ytrain = self.csw.gen_train_data()
      self.train_step(Xtrain,Ytrain)
      ## eval 
      if epoch_num%(epochs_per_block//num_evals)==0: 
        eval_idx += 1
        block_pred_data[block_idx] = self.predict_step(self.Xfull,self.Xfull)
    return block_pred_data

  def train_sess(self,blocking):
    """ 
    main train wrapper
    multiple training sessions, in each of which 
      the number of epochs per block is constant
    blocking: [(num_blocks,num_epochs)]
    """
    pred_data_L = []
    for sess_num,(num_blocks,num_epochs) in enumerate(blocking):
      for block_num in range(num_blocks):
        pred_data = self.train_block(num_epochs,csw_graph)
        pred_data_L.append(pred_data)
    return pred_data_L

  def main_loop(self,curricula):
    """ 
    this loop trains on multiple curricula: each curriculum 
      specifies number of blocks and epochs_per_block
    there are two contexts, in each block the network is trained 
      on single context but always makes predictions on both. 

    return: 
      pred_data['yhat'], shape: (epochs,path,depth,len,num_classes)
    """
    # initialize task: two graphs two filler ids
    task = CSWTask()
    graphs = [task.get_graph(self.graphpr),task.get_graph(1-self.graphpr)]
    graphids = [10,11]
    # prediction dataset
    Xfull = task.Xfull_onestory_det()
    # array for recording data
    total_num_evals = 0
    for nb,epb in curricula: total_num_evals += nb*epb
    pred_array_dtype = [('xbatch','int32',(len(Xfull),self.net.depth,self.net.in_len)),
                        ('yhat','float32',(len(Xfull),self.net.depth,self.net.out_len,
                                            self.net.num_classes))]
    pred_data = np.zeros((total_num_evals), dtype=pred_array_dtype)
    # main loop
    eval_idx = -1
    for nblocks,epb in curricula:
      print('curriculum',(nblocks,epb))
      for block in range(nblocks):
        blockid = block%2
        graph = graphs[blockid]
        graphid = graphids[blockid]
        # print('block',block)
        for ep in range(epb):
          # train step
          path = task.gen_single_path(graph)
          Xtrain,Ytrain = task.dataset_onestory(path,depth=DEPTH)
          self.train_step(Xtrain,Ytrain)
          # predict and eval
          eval_idx += 1
          predstep_data = self.predict_step(Xfull,Xfull)
          pred_data[eval_idx] = predstep_data
    return pred_data




