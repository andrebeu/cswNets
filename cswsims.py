import numpy as np
import tensorflow as tf
from customtf import LayerNormBasicLSTMCell as CustomLSTM


NUM_STORIES = 2
TRAIN_BATCH_SIZE = 1


## NB depth should be 7*nstories

class MetaLearner():

  def __init__(self,stsize,depth,random_seed=1):
    """
    """
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph)
    self.random_seed = random_seed
    # dimensions
    self.stsize = stsize
    self.embed_dim = stsize
    self.depth = depth 
    self.num_classes = 12
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
      ## inference
      self.unscaled_logits,self.final_cell_state_op,self.states = self.RNN() # batch,bptt,nclasses
      ## loss
      self.ybatch_onehot = tf.one_hot(indices=self.ybatch_id,depth=self.num_classes) # batch,bptt,nclasses
      self.train_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                          labels=self.ybatch_onehot,logits=self.unscaled_logits)
      print("SGD01")
      self.minimizer_op = tf.train.GradientDescentOptimizer(0.01).minimize(self.train_loss)
      ## accuracy
      self.yhat_sm = tf.nn.softmax(self.unscaled_logits)
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
    consumes a sentence at a time
      
    RNN structure:
      takes in state and a filler
      returns prediction for next state and a filler
    returns unscaled logits
    """
    xbatch = self.xbatch
    cell = self.cell = CustomLSTM(
            self.stsize,dropout_keep_prob=self.dropout_keep_prob)
    xbatch = tf.layers.dense(xbatch,self.stsize,tf.nn.relu,name='inproj')
    # unroll RNN
    with tf.variable_scope('RNN_SCOPE') as cellscope:
      # initialize state
      # initial_state = state = cell.zero_state(tf.cast(self.batch_size_ph,tf.int32),tf.float32)
      initstate = state = tf.nn.rnn_cell.LSTMStateTuple(self.cellstate_ph,self.cellstate_ph)
      # unroll
      outputL,stateL,fgateL = [],[],[]
      zero_input = tf.zeros_like(xbatch[:,0,:])
      for tstep in range(self.depth):
        # input
        __,state = cell(xbatch[:,tstep,:], state)
        stateL.append(state[0])
        fgateL.append(cell.forget_act)
        cellscope.reuse_variables()
        # output: inputs are zeroed out
        output,state = cell(zero_input, state) 
        stateL.append(state[0])
        fgateL.append(cell.forget_act)
        outputL.append(output)
    # format for y_hat
    outputs = tf.stack(outputL,axis=1)
    states = tf.stack(stateL,axis=1)
    self.fgate = tf.stack(fgateL,axis=1)
    # project to unscaled logits (to that outdim = num_classes)
    outputs = tf.layers.dense(outputs,self.num_classes,tf.nn.relu,name='outproj_unscaled_logits')
    return outputs,state,states



class Trainer():

  def __init__(self,net):
    self.net = net
    self.graphpr = 0.8
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
    # train loop
    while True:
      try:
        train_step_data,new_cell_state = self.net.sess.run(
          [self.net.minimizer_op,self.net.final_cell_state_op],feed_dict=train_feed_dict)
        # print(loss)
      except tf.errors.OutOfRangeError:
        break
    return train_step_data,new_cell_state

  def predict_step(self,Xpred,Ypred,cell_state):
    """ makes predictions on full dataset
    currently predictions are made on both contexts using same embedding
    ideally i could make predictions on each context independently 
    so that could make predictions with the appropriate embedding
    """
    batch_size = len(Xpred)
    # repeat cell state for each prediction
    cell_state = np.repeat(cell_state,batch_size,axis=0)
    # initialize data datastructure for collecting data
    pred_array_dtype = [('xbatch','int32',(batch_size,self.net.depth,self.net.in_len)),
                        ('yhat','float32',(batch_size,self.net.depth,self.net.out_len,self.net.num_classes)),
                        ('states','float32',(batch_size,self.net.depth*2,self.net.stsize)),
                        ('fgate','float32',(batch_size,self.net.depth*2,self.net.stsize))
                        ]
    pred_data_arr = np.zeros((),dtype=pred_array_dtype)
    # feed dict
    pred_feed_dict = {
      self.net.xph:Xpred,
      self.net.yph:Ypred,
      self.net.batch_size_ph: batch_size,
      self.net.dropout_keep_prob: 1.0,
      self.net.cellstate_ph: cell_state
    }
    # initialize iterator with eval data
    self.net.sess.run(self.net.itr_initop,pred_feed_dict)
    # eval loop
    while True:
      try:
        xbatch,yhat,states,fgate = self.net.sess.run(
          [self.net.xbatch_id,self.net.yhat_sm,self.net.states,self.net.fgate],feed_dict=pred_feed_dict)
        pred_data_arr['xbatch'] = xbatch
        pred_data_arr['yhat'] = yhat
        pred_data_arr['states'] = states
        pred_data_arr['fgate'] = fgate
      except tf.errors.OutOfRangeError:
        break 
    return pred_data_arr

  def train_multi_unroll(self,num_epochs,XevalL,num_stories=NUM_STORIES,curr='interleave'):
    """ 
    currently just unrolling two stories
    blocked is AA,BB,AA
    interleave is AB,AB,AB
    """
    # setup task vars
    task = CSWTask()
    graph0 = task.get_graph(self.graphpr)
    graph1 = task.get_graph(1-self.graphpr)
    graphids = [10,11]
    graphidL = [graphids[i%2] for i in range(num_stories)] 
    Xeval = task.format_Xeval(XevalL)
    # eval data array
    pred_array_dtype = [('xbatch','int32',(len(Xeval),self.net.depth,self.net.in_len)),
                        ('yhat','float32',(len(Xeval),self.net.depth,self.net.out_len,self.net.num_classes)),
                        ('states','float32',(len(Xeval),self.net.depth*2,self.net.stsize)),
                        ('fgate','float32',(len(Xeval),self.net.depth*2,self.net.stsize))
                        ]
    pred_data = np.zeros((num_epochs), dtype=pred_array_dtype)
    # init cell state
    zero_cell_state = cell_state = np.zeros(shape=[TRAIN_BATCH_SIZE,self.net.stsize])
    # train loop
    for ep in range(num_epochs):
      if curr == 'interleave':
        graphA = graph0
        graphB = graph1
      elif curr == 'block':
        if ep%2 == 0:
          graphA = graphB = graph0
        else:
          graphA = graphB = graph1
      # train step
      pathL = task.gen_pathL(num_stories,graphA,graphB)
      Xtrain,Ytrain = task.dataset_kstories_pomdp(pathL,graphidL)
      train_step_data,cell_state = self.train_step(Xtrain,Ytrain,cell_state)
      cell_state = cell_state[0]
      # eval
      pred_step_data = self.predict_step(Xeval,Xeval,cell_state)
      pred_data[ep] = pred_step_data
    return pred_data


""" CSWML TASK
read 10 stories, adjacent stories are from different graphs w.p. pr_shift
"""

class CSWMLTask():

  def __init__(self,graph_pr):
    self.graphA = self.get_graph(graph_pr)
    self.graphB = self.get_graph(1-graph_pr)
    self.end_node = 9

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
    graphs = [self.graphA,self.graphB]
    graphids = [10,11]
    pathL = []
    graphidL = []
    idx = np.random.binomial(1,.5) 
    for i in range(k):
      if np.random.binomial(1,pr_shift):
        print('SHIFTING')
        idx = (idx+1)%2
      graph = graphs[idx]
      path = self.gen_single_path(graph)
      graphidL.append(graphids[idx])
      pathL.append(path)
    return pathL,graphidL

  def dataset_kstories(self,pathL,graphidL):
    """
    given a pathL `list of arr` and graphidL `list of int`
    returns:
      X = [[[begin,id,st(t),st(t+1)],],]
      Y = [[[id,st(t+1),f1(t+1)],],]
      shape: (samples,depth)
    """
    num_paths = len(pathL)
    kpaths = np.concatenate([np.insert(pathL[i],0,graphidL[i]) for i in range(num_paths)])
    kpaths = np.expand_dims(kpaths,0)
    X = kpaths
    Y = np.roll(X,-1)
    return X,Y
