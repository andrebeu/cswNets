import os
import tensorflow as tf
import numpy as np
from datetime import datetime as dt
import matplotlib.pyplot as plt
import json


"""
fillers task and transition task requires different embedding mechanisms
fillers task needs filler embeddings to be changing every epoch
this is wastefully implemented by having two embedding matrices, one of which
  gets reinitialized at the begining of each epoch
when calculating the accuracy, i need to pass an embedding matrix, 
  when doing the fillers task i pass the random embedding matrix,
  when doing the transition task i pass the fixed embedding matrix

need to work out a way of supporting both tasks at the same time
can i form a tensor out of two tensors and then reinitialize only half
  of that tensor? yup!

"""

"""
curriculum learning: global vs local var init
"""

NUM_OF_EVALS = 500
VERB = True


"""
in the process of changing output of networks to allow for varibale length
need to work on accuracy evaluating function. currently there are squeezes and expand dims
which i don't quite understand. 
"""


""" FROM CATHY """


def get_01_accuracy(y_hat_ids, y_batch_ids, embed_mat):
  # print(1)
  y_hat_ids = tf.cast(y_hat_ids,tf.int32)
  y_batch_ids = tf.cast(y_batch_ids,tf.int32)
  eq = tf.equal(y_hat_ids,y_batch_ids)
  acc_state = tf.cast(
                tf.stack(
                  [eq[:,0],eq[:,2],eq[:,4]], axis=1),
                tf.float32)
  acc_filler = tf.cast(
                tf.stack(
                  [eq[:,1],eq[:,3],eq[:,5]], axis=1),
                tf.float32)
  return tf.reduce_mean(acc_state), tf.reduce_mean(acc_filler)




"""
helper funs
"""

def get_dt():
  return str(dt.now().strftime('%m-%d-%H.%M.%S.%f'))


def restore_saved_model(save_path):
  """ given a path, reads arch and chkpoint to assemble 
      the model, and returns a saved model
  """
  with open(save_path + '_arch') as arch_f:
    arch = json.load(arch_f)
  model = LSTMseq2(arch)
  model.saver_op.restore(model.sess,save_path)
  return model


def run_k_experiments(net,train_info,RAND,COND,k=1):
  acc_L = []
  for it in range(k):
    time1 = dt.now()
    net.reinitialize()
    eval_data = net.train(RAND,COND,train_info)
    acc_L.append(eval_data['accuracy'])
    print(it+1,dt.now()-time1)
  exp_data = np.array(acc_L)
  return exp_data



## NETWORK OBJECTS

class BaseRNN():

  def __init__(self,arch,saving):
    """ 
    """
    self.arch = arch
    self.dt_str = get_dt() 
    self.dtype_ = tf.float32
    self.graph = tf.Graph()
    # self.model_dir = 'savedmodels/LSTM_%s' % self.dt_str

    # self.sess = tf.InteractiveSession(graph=self.graph)
    self.sess = tf.Session(graph=self.graph)
    self.saving = saving

    self.input_seq_len = input_seq_len = arch['input_seq_len']
    self.output_seq_len = output_seq_len = arch['output_seq_len']
    self.celldim = celldim = arch['netdim']
    self.outdim = outdim = arch['netdim']
    self.embed_size = embed_size = arch['netdim']
    self.fix_vocab_size = fix_vocab_size = arch['fix_vocab_size']
    self.rand_vocab_size = rand_vocab_size = arch['rand_vocab_size']
    
    ## setup graph
    with self.graph.as_default():
      ## placeholders for input and output id
      (self.xph,self.yph,self.batch_size_ph) = (xph,yph,batch_size_ph
        ) = self.setup_placeholders(xph_dim=input_seq_len,
                                    yph_dim=output_seq_len)
      ## dataset and iterator
      (self.iterator,self.dataset) = (iterator,dataset,
        ) = self.setup_iterator(xph,yph,batch_size_ph)
      (self.x_batch_ids,self.y_batch_ids) = (x_batch_ids,y_batch_ids
        ) = iterator.get_next()
      self.itr_initop = itr_initop = iterator.make_initializer(dataset)
      ## embedding mechanism: given batch of ids, embed ids into batch of vectors
      (self.embed_mat, self.randomize_embed_mat
        ) = (embed_mat, randomize_embed_mat
        ) = self.get_embed_mat()
      ## inference
      self.y_batch = y_batch = self.get_embed_vec(y_batch_ids,embed_mat)
      self.x_batch = x_batch = self.get_embed_vec(x_batch_ids,embed_mat)
      self.y_hat = y_hat = self.setup_inference(x_batch)
      ## accuracy
      self.y_hat_ids = y_hat_ids = self.get_closest_embed_id(y_hat, embed_mat)
      self.acc_op = get_01_accuracy(y_hat_ids, y_batch_ids, embed_mat) # from cathy
      ## loss op
      self.loss_op = self.get_loss_op(tf.losses.mean_squared_error)
      ## extra
      self.saver_op = tf.train.Saver()
      self.sess.run(tf.global_variables_initializer())
      self.save_model('model-initial-%s'%self.dt_str)    
    return None

  ## 

  def reinitialize(self):
    with self.graph.as_default():
      self.sess.run(tf.global_variables_initializer())
    return None

  ## network settup

  def setup_placeholders(self,xph_dim,yph_dim):
    xph = tf.placeholder(tf.int32,
                  shape=[None,xph_dim],
                  name="xdata_placeholder")
    yph = tf.placeholder(tf.int32,
                  shape=[None,yph_dim],
                  name="ydata_placeholder")
    batch_size_ph = tf.placeholder(tf.int64,
                  shape=[],
                  name="batchsize_placeholder")
    return xph,yph,batch_size_ph

  def setup_iterator(self,xph,yph,batch_size_ph):
    """ returns an iterator which can be initialized 
        in a session, requires feed_dict = {xph,yph,batch_size_ph}
    """ 
    ## dataset
    dataset = tf.data.Dataset.from_tensor_slices((xph,yph))
    dataset = dataset.shuffle(10000)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size_ph))

    ## iterator
    iterator = tf.data.Iterator.from_structure(
                dataset.output_types, dataset.output_shapes)
    return iterator,dataset

  def get_optimizer(self):
    """ returns gradient_minimizer and loss_op
        if no loss_op is passed, constructs MSE
    """
    with self.graph.as_default():
      optimizer_op = tf.train.GradientDescentOptimizer(0.005)
      grads_and_vars = optimizer_op.compute_gradients(self.loss_op)
      updateparams_op = optimizer_op.apply_gradients(grads_and_vars)
    return updateparams_op

  ## train and eval

  def train(self,RAND,COND,train_info):
    """ 
    might want to break this function down soon

    how can I change this to save the data from training as a tensorflow
    variable that i can then restore 
    """

    train_t1 = get_dt()
    # input: training parameters
    batch_size = train_info['batch_size']

    ## assembles MSE loss_op in optimizer
    updateparams_op = self.get_optimizer()

    ## train loop
    num_epochs = train_info['num_epochs']
    # output: store data from trainin
    eval_data = {'loss':[],'accuracy':[]}
    for epoch_num in range(num_epochs):
      epoch_t1 = dt.now()
      # proto CSW data generate
      train_data_dict = gen_data_dict(batch_size,RAND,COND)
      train_feed_dict = {self.xph:train_data_dict['X_data'],
                         self.yph:train_data_dict['Y_data'],
                          self.batch_size_ph:batch_size}
      # initialize iterator with train data
      # randomize embedding matrix
      self.sess.run([self.itr_initop,self.randomize_embed_mat],train_feed_dict)
      # train loop
      while True:
        try: 
          _ = self.sess.run(updateparams_op,train_feed_dict)
        except tf.errors.OutOfRangeError:
          break
      # eval
      if epoch_num%(num_epochs//NUM_OF_EVALS)==0: 
        eval_acc = self.eval(RAND,COND)
        eval_data['accuracy'].append(eval_acc)
        if VERB and (epoch_num)%(num_epochs//(NUM_OF_EVALS/5))==0: 
          print('epoch',np.round(epoch_num/num_epochs,2),
                'elapsed time',dt.now()-epoch_t1)

    # save
    self.save_model('model-trained-%s' % train_t1) # model params
    self.save_data(eval_data,'data_from_training_on-%s'%train_t1) # data
    return eval_data

  def eval(self,RAND,COND,sess=None):
    """ makes predictions and computes loss
    """
    if not sess: sess = self.sess
    # feed dictionary
    eval_data_dict = gen_data_dict(100,RAND,COND)
    eval_feed_dict = {self.xph:eval_data_dict['X_data'], 
                      self.yph:eval_data_dict['Y_data'], 
                      self.batch_size_ph:1}
    # initialize iterator with eval data
    sess.run(self.itr_initop,eval_feed_dict)
    # compute loss and predictions
    filler_acc_L = []
    state_acc_L = []
    while True:
      try:
        state_acc,filler_acc = sess.run(self.acc_op,eval_feed_dict)
        # print(state_acc,filler_acc)
        state_acc_L.append(state_acc)
        filler_acc_L.append(filler_acc)
      except tf.errors.OutOfRangeError:
        break
    acc_arr = np.vstack([state_acc_L,filler_acc_L]).transpose()
    mean_acc = np.mean(acc_arr,0)
    return mean_acc

  ## save and restore 

  def save_model(self,save_name):
    """ saves both chkpoint and stores architecture required
    for later reconstructing the model using restore_saved_model
    """
    from copy import deepcopy
    if self.saving == False: return None
    # save model
    save_path = self.saver_op.save(self.sess,self.model_dir+'/'+save_name)
    arch_to_save = deepcopy(self.arch)
    # arch_to_save['cell'] = str(arch_to_save['cell'])
    # save architecture info
    with open(self.model_dir+'/'+save_name+"_arch", 'w') as arch_f:
      arch_f.write(json.dumps(arch_to_save))
    print('saving to', save_path)
    return save_path

  def save_data(self,data_dict,save_name):
    """ takes dict of data 
        saves them as a json dumps in model_dir
    """
    save_fpath = self.model_dir+'/'+save_name
    np.save(save_fpath,data_dict)
    return None
    # ensure json compatible 
    data_dict = {k:[np.float64(i) for i in v] for k,v in data_dict.items()}
    if self.saving == False: return None
    save_fpath = self.model_dir+'/'+save_name
    with open(save_fpath,'w') as train_data_f:
      train_data_f.write(json.dumps(data_dict))
    return None

  # embedding 

  def get_embed_mat(self):
    """ embedding matrix made of two parts a fixed part and a random part
    """
    embed_mat_fixed = tf.get_variable(name="embed_mat_fixed",
                        shape=[self.fix_vocab_size,self.embed_size],
                        dtype=self.dtype_)
    embed_mat_random = tf.get_variable(name="embed_mat_random",
                        shape=[self.rand_vocab_size,self.embed_size],
                        dtype=self.dtype_)
    embed_mat = tf.concat([embed_mat_fixed,embed_mat_random],axis=0)
    randomize_embed_mat = tf.variables_initializer([embed_mat_random])
    return embed_mat, randomize_embed_mat

  def get_embed_vec(self,embed_ids,embed_mat):
    """ 
    takes in embed_ids: batch of embed_id sequences [[1,2],[3,2],[5,1]]
    returns batch of embed_vec sequences [[v1,v2],[v3,v2],[v5,v1]]
    """

    # def embed_rand(x): return tf.nn.embedding_lookup(self.embed_mat_random,x)
    # def embed_fix(x): return tf.nn.embedding_lookup(self.embed_mat_fixed,x)
    # embed_vec = lambda embed_id: tf.cond(pred=embed_id<tf.constant(FILLER_K),
    #                               true_fn=lambda:embed_rand(embed_id),
    #                               false_fn=lambda:embed_fix(embed_id))
    # embed_vec_seq = lambda id_seq: tf.map_fn(embed_vec,id_seq,dtype=tf.float32)
    # embed_batch = tf.map_fn(embed_vec_seq,embed_ids,dtype=tf.float32)
    # return embed_batch
    embed_ids = tf.cast(embed_ids,tf.int32)
    embed_vec = tf.nn.embedding_lookup(embed_mat, embed_ids)
    # embed_vec = tf.nn.embedding_lookup(self.embed_mat, tf.cast(embed_ids,tf.int32))
    return embed_vec

  # inference and loss

  def get_loss_op(self,loss_op_):
    """ y_batch is a batch of embedding vector seqeunces returned by embedding_lookup 
        y_hat is the tensor returned by inference, given x_batch
    """
    with self.graph.as_default():
      with tf.variable_scope('loss_op'):
        # y_batch = tf.squeeze(self.y_batch,[1])
        y_batch = self.y_batch
        y_hat = self.y_hat
        loss_op = loss_op_(y_batch, y_hat)
    return loss_op 

  def get_closest_embed_id(self,embed_vecs,embed_mat):
    """ y_hat and y_batch are [batch_size,num_time_steps,embed_dim]
    embed_mat is []
    """
    # if len(embed_vecs.shape) < 2: embed_vecs = tf.expand_dims(batch_array, axis=0) 
    ## embed_mat matrix
    normed_embed_mat = tf.cast(tf.nn.l2_normalize(embed_mat, axis=1), tf.float32)
    normed_embed_mat = tf.transpose(normed_embed_mat, [1, 0])
    normed_embed_mat = tf.expand_dims(normed_embed_mat,axis=0)
    ## embed_vecs
    normed_embed_vecs = tf.cast(tf.nn.l2_normalize(embed_vecs, axis=-1), tf.float32) # assume last dimension is embedding dimension
    ## cosine sim
    cosine_similarity = tf.matmul(normed_embed_vecs, normed_embed_mat, name='closest_embed_id')
    max_similarity = tf.squeeze(tf.argmax(cosine_similarity, 2))
    return max_similarity






class RNN(BaseRNN):

  def __init__(self,arch,saving=False):
    ## unpack info from architecture
    super().__init__(arch,saving)

  def setup_inference(self,x_batch):
    """ given x_batch [samples,input_seq_len,embdim], and architecture specifications
    construct rnn inference.
    """
    # setup RNN cell
    self.cell = cell = tf.contrib.rnn.BasicRNNCell(self.celldim)
    # initialize state
    self.initial_state = cell_state = cell.zero_state(
      tf.cast(self.batch_size_ph,tf.int32),self.dtype_)
    # unroll RNN
    with tf.variable_scope('CELL_SCOPE') as cellscope:
      for time_step in range(self.input_seq_len):
        if time_step > 0: cellscope.reuse_variables()
        cell_output, cell_state = cell(x_batch[:,time_step,:], cell_state)
        cell_state = tf.contrib.layers.layer_norm(cell_state, scope=cellscope)
    y_hat = cell_output
    y_hat = tf.expand_dims(cell_output,1)
    return y_hat


class RNNseq(BaseRNN):

  def __init__(self,arch,saving=False):
    ## unpack info from architecture
    super().__init__(arch,saving)

  def setup_inference(self,x_batch):
    """ given x_batch [samples,input_seq_len,embdim], and architecture specifications
    construct rnn inference.
    """
    # setup RNN cell
    self.cell = cell = tf.contrib.rnn.BasicRNNCell(self.celldim)
    # initialize state
    self.initial_state = cell_state = cell.zero_state(
      tf.cast(self.batch_size_ph,tf.int32),self.dtype_)
    # unroll RNN
    with tf.variable_scope('CELL_SCOPE') as cellscope:
      # unroll input
      for time_step in range(self.input_seq_len):
        if time_step > 0: cellscope.reuse_variables()
        cell_output, cell_state = cell(x_batch[:,time_step,:], cell_state)
        cell_state = tf.contrib.layers.layer_norm(cell_state, scope=cellscope)
      # unroll output
      output_L = []
      zero_input = tf.zeros_like(x_batch)
      for time_step in range(self.output_seq_len):
        cell_output, cell_state = cell(zero_input[:,time_step,:], cell_state)
        cell_state = tf.contrib.layers.layer_norm(cell_state, scope=cellscope)
        output_L.append(cell_output)
    y_hat = tf.stack(output_L,axis=1)
    return y_hat


class LSTM(BaseRNN):

  def __init__(self,arch,saving=False):
    ## unpack info from architecture
    super().__init__(arch,saving)

  def setup_inference(self,x_batch):
    """ given x_batch [samples,input_seq_len,embdim], and architecture specifications
    construct rnn inference.
    """
    # setup RNN cell
    self.cell = cell = tf.contrib.rnn.LSTMCell(self.celldim)
    # initialize state
    self.initial_state = cell_state_and_output = cell.zero_state(
      tf.cast(self.batch_size_ph,tf.int32),self.dtype_)
    # unroll RNN
    with tf.variable_scope('CELL_SCOPE') as cellscope:
      for time_step in range(self.input_seq_len):
        if time_step > 0: cellscope.reuse_variables()
        _, (cell_state, cell_output) = cell(x_batch[:,time_step,:], cell_state_and_output)
        cell_state = tf.contrib.layers.layer_norm(cell_state, scope=cellscope)
        cell_state_and_output = tf.contrib.rnn.LSTMStateTuple(cell_state, cell_output)
    y_hat = tf.expand_dims(cell_output,1)
    return y_hat


class LSTMseq(BaseRNN):

  def __init__(self,arch,saving=False):
    ## unpack info from architecture
    super().__init__(arch,saving)

  def setup_inference(self,x_batch):
    """ given x_batch [samples,input_seq_len,embdim], and architecture specifications
    construct rnn inference.
    """
    # setup RNN cell
    self.cell = cell = tf.contrib.rnn.LSTMCell(self.celldim)
    # initialize state
    self.initial_state = cell_state_and_output = cell.zero_state(
      tf.cast(self.batch_size_ph,tf.int32),self.dtype_)
    # input unroll
    with tf.variable_scope('CELL_SCOPE') as cellscope:
      # input unroll
      for time_step in range(self.input_seq_len):
        if time_step > 0: cellscope.reuse_variables()
        _, (cell_state, cell_output) = cell(x_batch[:,time_step,:], cell_state_and_output)
        cell_state = tf.contrib.layers.layer_norm(cell_state, scope=cellscope)
        cell_state_and_output = tf.contrib.rnn.LSTMStateTuple(cell_state, cell_output)
      # output unroll
      output_L = []
      zero_input = tf.zeros_like(x_batch)
      for time_step in range(self.output_seq_len):
        _, (cell_state, cell_output) = cell(zero_input[:,time_step,:], cell_state_and_output)
        cell_state = tf.contrib.layers.layer_norm(cell_state, scope=cellscope)
        cell_state_and_output = tf.contrib.rnn.LSTMStateTuple(cell_state, cell_output)
        output_L.append(cell_output)
    y_hat = tf.stack(output_L,axis=1)
    return y_hat


class RNNseq2(BaseRNN):

  def __init__(self,arch,saving=False):
    ## unpack info from architecture
    self.num_sequences = arch['num_sequences']
    self.model_dir = 'savedmodels/RNNseq2_%s' % get_dt()
    super().__init__(arch,saving)
    


  def setup_placeholders(self,xph_dim,yph_dim):
    xph = tf.placeholder(tf.int32,
                  shape=[None,self.num_sequences*xph_dim],
                  name="xdata_placeholder")
    yph = tf.placeholder(tf.int32,
                  shape=[None,self.num_sequences*yph_dim],
                  name="ydata_placeholder")
    batch_size_ph = tf.placeholder(tf.int64,
                  shape=[],
                  name="batchsize_placeholder")
    return xph,yph,batch_size_ph

  def setup_inference(self,x_batch):
    """ 
    """
    # setup RNN cell
    self.cell = cell = tf.contrib.rnn.BasicRNNCell(self.celldim)
    # initialize state
    self.initial_state = cell_state = cell.zero_state(tf.cast(self.batch_size_ph,tf.int32),self.dtype_)
    zero_input = tf.zeros_like(x_batch) # feed during output unroll
    output_L = [] # collect output
    # unroll RNN
    with tf.variable_scope('CELL_SCOPE') as cellscope:
      for seq_num in range(self.num_sequences):
        # unroll input
        for time_step in range(self.input_seq_len):
          if time_step > 0: cellscope.reuse_variables()
          _, cell_state = cell(x_batch[:,time_step,:], cell_state)
          cell_state = tf.contrib.layers.layer_norm(cell_state, scope=cellscope)
        # unroll output
        for time_step in range(self.output_seq_len):
          cell_output, cell_state = cell(zero_input[:,time_step,:], cell_state)
          cell_state = tf.contrib.layers.layer_norm(cell_state, scope=cellscope)
          output_L.append(cell_output)
    y_hat = tf.stack(output_L,axis=1)
    return y_hat


class LSTMseq2(BaseRNN):

  def __init__(self,arch,saving=False):
    ## unpack info from architecture
    self.num_sequences = arch['num_sequences']
    self.model_dir = 'savedmodels/LSTMseq2_%s' % get_dt()
    super().__init__(arch,saving)
    


  def setup_placeholders(self,xph_dim,yph_dim):
    xph = tf.placeholder(tf.int32,
                  shape=[None,self.num_sequences*xph_dim],
                  name="xdata_placeholder")
    yph = tf.placeholder(tf.int32,
                  shape=[None,self.num_sequences*yph_dim],
                  name="ydata_placeholder")
    batch_size_ph = tf.placeholder(tf.int64,
                  shape=[],
                  name="batchsize_placeholder")
    return xph,yph,batch_size_ph

  def setup_inference(self,x_batch):
    """ 
    """
    # setup cell
    self.cell = cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.celldim)
    # initialize state
    self.initial_state = cell_state_and_output = cell.zero_state(
      tf.cast(self.batch_size_ph,tf.int32),self.dtype_)
    # for output unroll
    output_L = []
    zero_input = tf.zeros_like(x_batch)
    # unroll RNN
    with tf.variable_scope('CELL_SCOPE') as cellscope:
      for seq_num in range(self.num_sequences):
        # input unroll
        for time_step in range(self.input_seq_len):
          if time_step > 0: cellscope.reuse_variables()
          _, cell_state_and_output = cell(x_batch[:,time_step,:], cell_state_and_output)
        # unroll output
        for time_step in range(self.output_seq_len):
          cell_output, cell_state_and_output = cell(zero_input[:,time_step,:], cell_state_and_output)
          output_L.append(cell_output)
    y_hat = tf.stack(output_L,axis=1)
    return y_hat



def gen_data_dict(story_size=100,RAND=False,COND=False):

  graph_dict = {
    0:[1,2],
    1:[3,4],
    2:[4,3],
    3:[5,6],
    4:[6,5],
    5:[7,7],
    6:[7,7]
  }


  def gen_path(graph_dict,pr):
    """ graph is a dict{A:[B,C]}
    A is current state B is transitioned to with pr
    each path makes a single training example"""
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

  if RAND:
    filler_id = 15
  else:
    filler_id = 8

  if COND:
    X1,Y1 = gen_m_samples(m=int(story_size/2),graph_dict=graph_dict,transition_pr=.8,filler_id=filler_id)
    X2,Y2 = gen_m_samples(m=int(story_size/2),graph_dict=graph_dict,transition_pr=.2,filler_id=filler_id+1)
    X = np.vstack([X1,X2])
    Y = np.vstack([Y1,Y2])
  else:
    X,Y = gen_m_samples(m=story_size,graph_dict=graph_dict,transition_pr=.8,filler_id=filler_id)

  train_data_dict = {'X_data':X,'Y_data':Y}
  return train_data_dict


