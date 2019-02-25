import numpy as np
import tensorflow as tf
from customtf import LSTMBaseline,LSTMFgatePE


TRAIN_BATCH_SIZE = 1

"""
also save cell state trajectory during eval
change train & save file to save (1) model (2) train data (3) eval data at end of training

"""

## NB depth should be 6*nstories

class MetaLearner():

  def __init__(self,stsize,feedPE,random_seed=1):
    """
    """
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph)
    self.random_seed = random_seed
    # dimensions
    self.stsize = stsize
    self.embed_dim = stsize
    self.nstories = 6
    self.depth = 6*6 
    self.num_classes = 12
    self.feedPE = feedPE
    # build
    self.build()

  def build(self):
    with self.graph.as_default():
      tf.set_random_seed(self.random_seed)
      print('initializing sub%.2i'%self.random_seed)
      # place holders
      self.setup_placeholders()
      # pipeline
      self.xbatch_id,self.ybatch_id = self.data_pipeline() # x(batches,bptt), y(batch,bptt)
      ## embedding 
      self.embed_mat = tf.get_variable('embedding_matrix',[self.num_classes,self.embed_dim])
      self.xbatch = tf.nn.embedding_lookup(self.embed_mat,self.xbatch_id,name='xembed') # batch,bptt,stsize
      self.ybatch_onehot_full = tf.one_hot(indices=self.ybatch_id,depth=self.num_classes) # batch,bptt,nclasses
      ## inference
      self.unscaled_logits_full,self.final_cell_state_op,self.states = self.RNN() # batch,bptt,nclasses
      ## eval loss
      self.eval_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                          labels=self.ybatch_onehot_full,logits=self.unscaled_logits_full)
      ## train_loss
      self.unscaled_logits_bptt = self.unscaled_logits_full[:,-3*6:,:]
      self.ybatch_onehot_bptt = self.ybatch_onehot_full[:,-3*6:,:]
      self.train_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                          labels=self.ybatch_onehot_bptt,logits=self.unscaled_logits_bptt)
      print("ADAM01")
      self.minimizer_op = tf.train.AdamOptimizer(0.01).minimize(self.train_loss)
      ## accuracy
      self.yhat_sm = tf.nn.softmax(self.unscaled_logits_full)
      self.yhat_id = tf.argmax(self.yhat_sm,-1)
      ## extra
      self.sess.run(tf.global_variables_initializer())
      self.saver_op = tf.train.Saver()
    return None

  def setup_placeholders(self):
    self.xph = xph = tf.placeholder(tf.int32,
                  shape=[None,self.depth],
                  name="xdata_placeholder")
    self.yph = yph = tf.placeholder(tf.int32,
                  shape=[None,self.depth],
                  name="ydata_placeholder")
    self.batch_size_ph = tf.placeholder(tf.int64,
                  shape=[],
                  name="batchsize_placeholder")
    self.dropout_keep_prob = tf.placeholder(tf.float32,
                                shape=[],
                                name="dropout_ph")
    self.cellstate_ph = tf.placeholder(tf.float32,
                  shape=[None,self.stsize],
                  name = "initialstate_ph")
    return None

  def data_pipeline(self):
    """
    setup data iterator pipeline
    creates self.itr_initop and self.dataset
    returns x,y = get_next
    """
    # dataset
    dataset = tf.data.Dataset.from_tensor_slices((self.xph,self.yph))
    dataset = dataset.batch(self.batch_size_ph)
    # dataset = self.dataset = dataset.shuffle(100000)
    # iterator
    iterator = tf.data.Iterator.from_structure(
                dataset.output_types, dataset.output_shapes)
    xbatch,ybatch = iterator.get_next() 
    self.itr_initop = iterator.make_initializer(dataset)
    return xbatch,ybatch

  def reinitialize(self):
    print('**reinitializing weights** - incremental seeds')
    with self.graph.as_default():
      self.random_seed = self.random_seed + 1
      print('reinitializing sub%.2i'%self.random_seed)
      tf.set_random_seed(self.random_seed)
      self.sess.run(tf.global_variables_initializer())
    return None

  def RNN(self):
    """ 
    general RNN structure that allows specifying 
      - depth: number of (input_seq,output_seq) that are unrolled
      - in_len: length of each input sequence
      - out_len: length of each output sequence
      
    RNN structure:
      takes in state and a filler
      returns prediction for next state and a filler
    returns unscaled logits
    """
    ## CELLL
    if (self.feedPE=='baseline') or (self.feedPE=='input'):
      cell = self.cell = LSTMBaseline(
            self.stsize,dropout_keep_prob=self.dropout_keep_prob)
    elif (self.feedPE == 'fgate'):
      cell = self.cell = LSTMFgatePE(
            self.stsize,dropout_keep_prob=self.dropout_keep_prob)
    # unroll RNN
    with tf.variable_scope('RNN_SCOPE') as cellscope:
      # initialize state
      initialPE = PE_t = tf.zeros_like(self.ybatch_onehot_full[:,0,:],tf.float32)
      initstate = state = tf.nn.rnn_cell.LSTMStateTuple(self.cellstate_ph,self.cellstate_ph)
      # unroll
      outputL,stateL,fgateL,PEL = [],[],[],[]
      for tstep in range(self.depth):
        # input projection
        if (self.feedPE=='baseline'):
          input_t = tf.layers.dense(
                      self.xbatch[:,tstep,:],
                      self.stsize,tf.nn.relu,name='inproj')
        elif (self.feedPE=='input') or (self.feedPE=='fgate'):
          input_t = tf.layers.dense(
                      tf.concat([self.xbatch[:,tstep,:],PE_t],1),
                      self.stsize,tf.nn.relu,name='inproj')
        # cell update
        output,state = cell(input_t, state)
        # outproj 
        out_proj = tf.layers.dense(output,self.num_classes,tf.nn.relu,name='outproj_unscaled_logits')
        # update PE
        PE_t = tf.math.squared_difference(tf.nn.softmax(out_proj),self.ybatch_onehot_full[:,tstep,:])
        # collect
        PEL.append(PE_t)
        stateL.append(state[0])
        fgateL.append(cell.forget_act)
        outputL.append(out_proj)
        cellscope.reuse_variables()
    # formatting
    outputs = tf.stack(outputL,axis=1)
    states = tf.stack(stateL,axis=1)
    self.fgate = tf.stack(fgateL,axis=1)
    self.PE = tf.stack(PEL,axis=1)
    return outputs,state,states

""" 

"""

class Trainer():

  def __init__(self,net,shift_pr,graph_pr):
    self.net = net
    self.graph_pr = graph_pr
    self.shift_pr = shift_pr
    self.task = CSWMLTask(self.graph_pr)
    return None

  # steps: single pass through dataset

  def train_step(self,Xtrain,Ytrain,cell_state):
    """ updates model parameters using Xtrain,Ytrain
    """
    # initialize iterator with train data
    train_feed_dict = {
      self.net.xph: Xtrain,
      self.net.yph: Ytrain,
      self.net.batch_size_ph: TRAIN_BATCH_SIZE,
      self.net.dropout_keep_prob: .9,
      self.net.cellstate_ph: cell_state
      }
    self.net.sess.run([self.net.itr_initop],train_feed_dict)
    ## train step
    _,new_cell_state = self.net.sess.run([
                        self.net.minimizer_op,
                        self.net.final_cell_state_op],
                         feed_dict=train_feed_dict)
    return new_cell_state

  def eval_step(self,Xeval,Yeval,cell_state='rand'):
    """ makes predictions on full dataset
    currently predictions are made on both contexts using same embedding
    ideally i could make predictions on each context independently 
    so that could make predictions with the appropriate embedding
    """
    if cell_state == 'rand':
      cell_state = np.random.random([1,self.net.stsize])
    # repeat cell state for each prediction
    # cell_state = np.repeat(cell_state,axis=0)
    # initialize data datastructure for collecting data
    eval_array_dtype = [('xbatch','int32',(self.net.depth)),
                        ('yhat','float32',(self.net.depth,self.net.num_classes)),
                        ('states','float32',(self.net.depth,self.net.stsize)),
                        ('fgate','float32',(self.net.depth,self.net.stsize)),
                        ('PE','float32',(self.net.depth,self.net.num_classes))
                        ]
    eval_data_arr = np.zeros((),dtype=eval_array_dtype)
    # feed dict
    pred_feed_dict = {
      self.net.xph:Xeval,
      self.net.yph:Yeval,
      self.net.batch_size_ph: 1,
      self.net.dropout_keep_prob: 1.0,
      self.net.cellstate_ph: cell_state
    }
    # initialize iterator with eval data
    self.net.sess.run(self.net.itr_initop,pred_feed_dict)
    # eval loop
    while True:
      try:
        xbatch,yhat,states,fgate,PE = self.net.sess.run(
          [self.net.xbatch_id,self.net.yhat_sm,self.net.states,self.net.fgate,self.net.PE],feed_dict=pred_feed_dict)
        eval_data_arr['xbatch'] = xbatch.squeeze()
        eval_data_arr['yhat'] = yhat.squeeze()
        eval_data_arr['states'] = states.squeeze()
        eval_data_arr['fgate'] = fgate.squeeze()
        eval_data_arr['PE'] = PE.squeeze()
      except tf.errors.OutOfRangeError:
        break 
    return eval_data_arr

  def eval_loop(self,context_strL):
    """ 
    evaluates on sequences generated from context_strL
    """
    eval_arr_dtype = [(context_str, [
                        ('xbatch','int32',(self.net.depth)),
                        ('yhat','float32',(self.net.depth,self.net.num_classes)),
                        ('states','float32',(self.net.depth,self.net.stsize)),
                        ('fgate','float32',(self.net.depth,self.net.stsize))
                        ]) for context_str in context_strL]
    eval_arr = np.zeros([],dtype=eval_arr_dtype)
    for context_str in context_strL:
      Xeval,Yeval = self.task.get_Xeval(context_str)
      evalstep_data = self.eval_step(Xeval,Yeval,cell_state='rand')
      eval_arr[context_str] = evalstep_data
    return eval_arr

  def train_loop(self,nepochs):
    """ 
    """
    nevals = nepochs
    Xeval,Yeval = self.task.get_Xeval('A1B1A1A1B1B1')
    print('training eval is on ABAABB')
    ## setup eval data array
    eval_array_dtype = [('xbatch','int32',(self.net.depth)),
                        ('yhat','float32',(self.net.depth,self.net.num_classes)),
                        ('states','float32',(self.net.depth,self.net.stsize)),
                        ('fgate','float32',(self.net.depth,self.net.stsize)),
                        ('PE','float32',(self.net.depth,self.net.num_classes))
                        ]
    train_data = np.zeros((nevals), dtype=eval_array_dtype)
    ## init cell state
    rand_cell_state = cell_state = np.random.random([1,self.net.stsize])
    # train loop
    for ep in range(nepochs):
      # generate data
      pathL = self.task.gen_pathL(self.net.nstories,self.shift_pr)
      Xtrain,Ytrain = self.task.dataset_kstories(pathL)
      # eval network on data
      trainstep_data = self.eval_step(Xeval,Yeval,cell_state)
      train_data[ep] = trainstep_data
      # update params
      cell_state = self.train_step(Xtrain,Ytrain,cell_state)
      cell_state = cell_state[0] # only c-state
      # eval
      if ep%(nepochs/20)==0:
        print(100*ep/nepochs)
    return train_data.squeeze()


""" CSW METALEARNING TASK
adjacent stories are from different graphs w.p. pr_shift
"""

class CSWMLTask():

  def __init__(self,graph_pr):
    self.graphs = [self.get_graph(graph_pr),self.get_graph(1-graph_pr)]
    self.end_node = 9
    # randomly initialize context
    self.graph_idx = np.random.binomial(1,.5) # 0 or 1

  def get_graph(self,graph_pr):
    """ returns a dict which encodes graph
    {frnode:{tonode:pr,},}
    """
    r = np.round(graph_pr,2)
    s = np.round(1-graph_pr,2)
    graph = {
      0:{1:.5,2:.5},
      1:{3:r,4:s},
      2:{3:s,4:r},
      3:{5:r,6:s},
      4:{5:s,6:r},
      5:{7:r,8:s},
      6:{7:s,8:r},
      7:{9:1.0},
      8:{9:1.0}
    }
    return graph

  def gen_single_path(self,graph):
    """
    begins at 0
    transitions to 1 or 2 w.p. .5
    subsequent transitions controlled by PR
    ends when reach END_NODE
    """
    path = []
    frnode = 0
    while frnode!= self.end_node:
      path.append(frnode)
      distr = graph[frnode]
      tonodes = list(distr.keys())
      pr = list(distr.values())
      frnode = np.random.choice(tonodes,p=pr)
    path.append(self.end_node)
    return np.array(path)

  def gen_pathL(self,k,pr_shift):
    """ 
    generates k paths fully interleaving graphA and graphB
    returns a list, each item being a path
    """
    pathL = []
    for i in range(k):
      # w/pr pr_shift, change contexts
      if np.random.binomial(1,pr_shift):
        self.graph_idx = (self.graph_idx+1)%2
      graph = self.graphs[self.graph_idx]
      path = self.gen_single_path(graph)
      pathL.append(path)
    return pathL

  def dataset_kstories(self,pathL):
    """
    given a pathL `list of arr` 
    returns:
      X = [[[begin,st(t),st(t+1)],],]
      shape: (samples,depth)
    """
    num_paths = len(pathL)
    kpaths = np.concatenate([pathL[i] for i in range(num_paths)])
    kpaths = np.expand_dims(kpaths,0)
    X = kpaths
    Y = np.roll(X,-1)
    return X,Y

  def get_Xeval(self,context_str):
    """ given a context_str (e.g. 'ABA') generates an eval dataset 
    """
    D = {'A1':[0, 1, 3, 5, 7, 9],
    		 'A2':[0, 2, 4, 6, 8, 9],
         'B1':[0, 1, 4, 5, 8, 9],
         'B2':[0, 2, 3, 6, 7, 9]
         }
    pathL = [D[context_str[2*cidx:2*cidx+2]] for cidx in np.arange(int(len(context_str)/2))]
    Xeval,Yeval = self.dataset_kstories(list(pathL))
    return Xeval,Yeval
