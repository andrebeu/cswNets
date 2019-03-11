import numpy as np
import tensorflow as tf
from customtf import LayerNormBasicLSTMCell as CustomLSTM

"""
"""

class CSWNet():

  def __init__(self,stsize,random_seed=1):
    """
    """
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph)
    self.random_seed = random_seed
    # dimensions
    self.stsize = stsize
    self.embed_dim = stsize
    self.nstories = 6
    self.story_depth = 6
    self.depth = self.story_depth*self.nstories 
    self.num_nodes = 10
    self.num_contexts = 2
    # indexes for context and node time steps
    self.tstep_idx_context = np.arange(0,self.depth,self.story_depth)
    self.tstep_idx_node = np.delete(np.arange(self.depth),self.tstep_idx_context)
    # build
    self.build()

  def build(self):
    with self.graph.as_default():
      tf.set_random_seed(self.random_seed)
      print('initializing sub%.2i'%self.random_seed)
      # place holders
      self.setup_placeholders()
      # pipeline
      self.xbatch_id,self.ybatch_node_id,self.ybatch_context_id = self.data_pipeline() # x(batches,bptt), y(batch,bptt)
      ## embedding 
      self.node_emat = tf.get_variable('node_emat',[self.num_nodes,self.embed_dim])
      self.xbatch = tf.nn.embedding_lookup(self.node_emat,self.xbatch_id,name='xembed') # batch,bptt,stsize
      ## inference
      self.ylogits_node_full,self.ylogits_context_full,self.final_state = self.RNN(self.xbatch) # batch,bptt,nclasses
      ## train_loss
      # take separate training timesteps for each readout
      self.ylogits_node_train = tf.gather(self.ylogits_node_full,self.tstep_idx_node,axis=1)
      self.ylogits_context_train = tf.gather(self.ylogits_context_full,self.tstep_idx_context,axis=1)
      # one hot labels
      self.ybatch_onehot_node = tf.one_hot(indices=self.ybatch_node_id,depth=self.num_nodes) 
      self.ybatch_onehot_context = tf.one_hot(indices=self.ybatch_context_id,depth=self.num_contexts) 
      # loss
      self.train_loss_node = tf.nn.softmax_cross_entropy_with_logits_v2(
                                  labels=self.ybatch_onehot_node,
                                  logits=self.ylogits_node_train)
      self.train_loss_context = tf.nn.softmax_cross_entropy_with_logits_v2(
                                  labels=self.ybatch_onehot_context,
                                  logits=self.ylogits_context_train)
      self.train_loss = tf.concat([self.train_loss_node,self.train_loss_context],axis=1)
      ## optimizer
      self.minimize_node = tf.train.AdamOptimizer(0.0001).minimize(self.train_loss_node)
      self.minimize_context = tf.train.AdamOptimizer(0.0001).minimize(self.train_loss_context)
      self.minimizer_op = tf.group([self.minimize_node,self.minimize_context])
      # self.minimizer_op = self.minimize_context
      ## softmax normalization and argmax
      self.ynode_sm = tf.nn.softmax(self.ylogits_node_train) 
      self.ynode_id = tf.argmax(self.ynode_sm,-1)
      self.ycontext_sm = tf.nn.softmax(self.ylogits_context_full) 
      self.ycontext_id = tf.argmax(self.ycontext_sm,-1)
      ## extra
      self.sess.run(tf.global_variables_initializer())
      self.saver_op = tf.train.Saver()
    return None

  def setup_placeholders(self):
    self.xph = tf.placeholder(tf.int32,
                  shape=[1,self.depth],
                  name="xdata_ph")
    self.yph = tf.placeholder(tf.int32,
                  shape=[1,self.depth],
                  name="ydata_node_ph")
    self.cell_state = tf.placeholder(tf.float32,
                  shape=[1,self.stsize],
                  name="cell_state_ph")
    self.dropout_keep_pr = tf.placeholder(tf.float32,
                  shape=[],
                  name="dropout_ph")
    return None

  def data_pipeline(self):
    """
    setup data iterator pipeline
    creates self.itr_initop and self.dataset
    returns x,y = get_next
    """
    yph_node = tf.gather(self.yph,self.tstep_idx_node,axis=1)
    yph_context = tf.gather(self.yph,self.tstep_idx_context,axis=1)
    # dataset
    dataset = tf.data.Dataset.from_tensor_slices((self.xph,yph_node,yph_context))
    # iterator
    iterator = tf.data.Iterator.from_structure(
                dataset.output_types, dataset.output_shapes)
    xbatch,ybatch_node,ybatch_context = iterator.get_next() 
    self.itr_initop = iterator.make_initializer(dataset)
    # include batch dimension
    xbatch = tf.expand_dims(xbatch,0)
    ybatch_node = tf.expand_dims(ybatch_node,0)
    ybatch_context = tf.expand_dims(ybatch_context,0)
    return xbatch,ybatch_node,ybatch_context

  def reinitialize(self):
    print('**reinitializing weights** - incremental seeds')
    with self.graph.as_default():
      self.random_seed = self.random_seed + 1
      print('reinitializing sub%.2i'%self.random_seed)
      tf.set_random_seed(self.random_seed)
      self.sess.run(tf.global_variables_initializer())
    return None

  def RNN(self,xbatch):
    """ 
    """
    cell = self.cell = CustomLSTM(
            self.stsize,dropout_keep_prob=self.dropout_keep_pr)
    xbatch = tf.layers.dense(xbatch,self.stsize,tf.nn.relu,name='inproj')
    # unroll RNN
    with tf.variable_scope('RNN_SCOPE') as cellscope:
      # initialize state
      state = tf.nn.rnn_cell.LSTMStateTuple(self.cell_state,self.cell_state)
      # unroll
      outL,stateL,fgateL = [],[],[]
      for tstep in range(self.depth):
        output,state = cell(xbatch[:,tstep,:], state)
        outL.append(output)
        stateL.append(state[0])
        fgateL.append(cell.forget_act)
        cellscope.reuse_variables()
    # states and gates for inspection
    self.states = tf.stack(stateL,axis=1)
    self.fgate = tf.stack(fgateL,axis=1)
    ### readout layer
    lstm_outputs = tf.stack(outL,axis=1)
    # layers
    ylogits_node = tf.layers.dense(lstm_outputs,
      self.num_nodes,tf.nn.relu,name='logits_nodes')
    ylogits_context = tf.layers.dense(lstm_outputs,
      self.num_contexts,tf.nn.relu,name='logits_context')
    return ylogits_node,ylogits_context,state[0]

  def RNN_keras(self,xbatch):
    """
    NB unlike before no input projection
    """
    inlayer = tf.keras.layers.Dense(self.stsize,activation='relu')
    xbatch = inlayer(xbatch)
    lstm_cell = tf.keras.layers.LSTMCell(self.stsize,dropout=0)
    init_state = lstm_cell.get_initial_state(
                    tf.get_variable('initial_state',
                      trainable=True, 
                      shape=[1,self.stsize]))
    lstm_layer = tf.keras.layers.RNN(lstm_cell,
                    stateful=False,
                    return_sequences=True,
                    return_state=True)
    lstm_outputs,final_output,final_state = lstm_layer(xbatch,
          initial_state=[self.cell_state,self.cell_state])
    
    ##  readout layers
    # separate context from node prediction timesteps
    lstm_outputs_node = tf.gather(lstm_outputs,self.tstep_idx_node,axis=1)
    lstm_outputs_context = tf.gather(lstm_outputs,self.tstep_idx_context,axis=1)
    # output layer
    ylogits_node = tf.keras.layers.Dropout(0)(
                    tf.keras.layers.Dense(self.num_nodes,activation=None)(
                      lstm_outputs_node))
    ylogits_context = tf.keras.layers.Dropout(0)(
                    tf.keras.layers.Dense(self.num_contexts,activation=None)(
                      lstm_outputs_context))
    return ylogits_node,ylogits_context,final_state


""" 
"""

class Trainer():

  def __init__(self,net,shift_pr,graph_pr):
    self.net = net
    self.graph_pr = graph_pr
    self.shift_pr = shift_pr
    self.task = CSWTask(graph_pr)
    return None

  # steps: single pass through dataset

  def train_step(self,Xtrain,Ytrain,cell_state):
    """ updates model parameters using Xtrain,Ytrain
    """
    # tf.keras.backend.set_learning_phase(1)
    # initialize iterator with train data
    train_feed_dict = {
      self.net.xph:Xtrain,
      self.net.yph:Ytrain,
      self.net.cell_state:cell_state,
      self.net.dropout_keep_pr:0.9
      }
    self.net.sess.run([self.net.itr_initop],train_feed_dict)
    _,final_cell_state = self.net.sess.run([
      self.net.minimizer_op,self.net.final_state
      ],train_feed_dict)
    return final_cell_state

  def eval_step(self,Xeval,Yeval,cell_state):
    """ makes predictions on full dataset
    currently predictions are made on both contexts using same embedding
    ideally i could make predictions on each context independently 
    so that could make predictions with the appropriate embedding
    """
    # tf.keras.backend.set_learning_phase(0)
    # initialize data datastructure for collecting data
    eval_array_dtype = [('xbatch','int32',(self.net.depth)),
                        ('ynode_sm','float32',(self.net.depth-self.net.nstories,self.net.num_nodes)),
                        ('ycontext_sm','float32',(self.net.depth,self.net.num_contexts)),
                        # ('states','float32',(self.net.depth,self.net.stsize)),
                        # ('fgate','float32',(self.net.depth,self.net.stsize))
                        ]
    evalstep_data = np.zeros((),dtype=eval_array_dtype)
    # feed dict
    pred_feed_dict = {
      self.net.xph:Xeval,
      self.net.yph:Yeval,
      self.net.cell_state:cell_state,
      self.net.dropout_keep_pr:1.0
    }
    # initialize iterator with eval data
    self.net.sess.run(self.net.itr_initop,pred_feed_dict)
    # eval loop
    while True:
      try:
        xbatch,ynode_sm,ycontext_sm = self.net.sess.run([
                                      self.net.xbatch_id,
                                      self.net.ynode_sm,
                                      self.net.ycontext_sm,
                                      # self.net.states,
                                      # self.net.fgate
                                      ],feed_dict=pred_feed_dict)
        evalstep_data['xbatch'] = xbatch.squeeze()
        evalstep_data['ynode_sm'] = ynode_sm.squeeze()
        evalstep_data['ycontext_sm'] = ycontext_sm.squeeze()
        # evalstep_data['states'] = states.squeeze()
        # evalstep_data['fgate'] = fgate.squeeze()
      except tf.errors.OutOfRangeError:
        break 
    return evalstep_data

  def eval_loop(self,context_strL):
    """ 
    evaluates on sequences generated from context_strL
    """
    eval_arr_dtype = [(context_str, [
                        ('xbatch','int32',(self.net.depth)),
                        ('yhat','float32',(self.net.depth,self.net.num_classes)),
                        # ('states','float32',(self.net.depth,self.net.stsize)),
                        # ('fgate','float32',(self.net.depth,self.net.stsize))
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
    Xeval,Yeval = self.task.get_Xeval('A1A1A1B1B1B1')
    ## setup eval data array
    eval_array_dtype = [('xbatch','int32',(self.net.depth)),
                        ('ynode_sm','float32',(self.net.depth-self.net.nstories,self.net.num_nodes)),
                        ('ycontext_sm','float32',(self.net.depth,self.net.num_contexts)),
                        # ('states','float32',(self.net.depth,self.net.stsize)),
                        # ('fgate','float32',(self.net.depth,self.net.stsize))
                        ]
    train_data = np.zeros((nepochs), dtype=eval_array_dtype)
    cell_state = np.zeros([1,self.net.stsize])
    # train loop
    for ep in range(nepochs):
      # generate data
      pathL,graphidL = self.task.gen_pathL(self.net.nstories,self.shift_pr)
      Xtrain,Ytrain = self.task.dataset_kstories(pathL,graphidL)
      # update params
      cell_state = self.train_step(Xtrain,Ytrain,cell_state)
      # eval
      train_data[ep] = self.eval_step(Xeval,Yeval,cell_state)
      if ep%(nepochs/20)==0:
        print(100*ep/nepochs)
    return train_data.squeeze()


""" CSW METALEARNING TASK
adjacent stories are from different graphs w.p. pr_shift
"""

class CSWTask():

  def __init__(self,graph_pr):
    self.graphs = [self.get_graph(graph_pr),self.get_graph(1-graph_pr)]
    self.end_node = 9
    # randomly initialize context
    self.graph_idx = np.random.binomial(1,.5) # 0 or 1
    self.graphids = [0,1]

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
      tonode = list(distr.keys())
      pr = list(distr.values())
      frnode = np.random.choice(tonode,p=pr)
    path.append(self.end_node)
    return np.array(path)

  def gen_pathL(self,k,pr_shift):
    """ 
    generates k paths fully interleaving graphA and graphB
    returns a list, each item being a path
    """
    pathL = []
    graphidL = []
    for i in range(k):
      # w/pr pr_shift, change contexts
      if np.random.binomial(1,pr_shift):
        self.graph_idx = (self.graph_idx+1)%2
      graph = self.graphs[self.graph_idx]
      path = self.gen_single_path(graph)
      graphidL.append(self.graphids[self.graph_idx])
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
    X = np.concatenate(pathL)
    Y = np.roll(X,-1)
    Y[np.arange(0,len(X),6)] = graphidL
    X = np.expand_dims(X,0)
    Y = np.expand_dims(Y,0)
    return X,Y

  def get_Xeval(self,context_str):
    """ given a context_str (e.g. 'ABA') generates an eval dataset 
    """
    D = {'A1':[[0, 1, 3, 5, 7, 9],10],
    		 'A2':[[0, 2, 4, 6, 8, 9],10],
         'B1':[[0, 1, 4, 5, 8, 9],11],
         'B2':[[0, 2, 3, 6, 7, 9],11]
         }
    pathL = [D[context_str[2*cidx:2*cidx+2]][0] for cidx in np.arange(int(len(context_str)/2))]
    graphidL = [D[context_str[2*cidx:2*cidx+2]][1] for cidx in np.arange(int(len(context_str)/2))]
    Xeval,Yeval = self.dataset_kstories(list(pathL),list(graphidL))
    return Xeval,Yeval


