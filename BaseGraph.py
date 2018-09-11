import os
import tensorflow as tf
import numpy as np
from datetime import datetime as dt
import json
from cswNetEngine import CSW
  

"""
Cell defines the computation
RNN defines the task structure
Graph constructs the tf graph (embedding matrix, saving, accuracy ops...)


LR exp:
LR given by hoppfield energy between network output and training sample.
prediction: when a network settles on a strategy that does well for graph A, 
  the learning rate will remain low. if we suddenly we shift to graph B, 
  the learning rate will go up. 
but the question remains: will this changing LR enable/improve learning
  over naive fixed LR? 

eval function currently evaluates on graph presented for training
what i really want is to eval on the 'entire dataset' i.e. every datapoint presented
  to the network up to that point.

consider a softmax at the output layer instead. instead of computing accuracy
  by doing a nearest neighbor search, just do max over the softmax layer.
"""

""" REVERTED: MODIFICATIONS - 21 AUG 

moved softmax from basegraph into Cell
Cell now gets x_batch_ids (data) instead of x_batch_embedded (embedded data)
  this has implications for the dimension of the inputs to the cell. 
    - instead of changing what dim is assumed in Cell, I expand_dim when not embedding
    this means the in/out dim assumed by cell is [batch,time,dim]
moving embedding matrix formation into RNN
  because structuring embedding matrix is related to the task

considering changing call profile:
  BaseGraph(RNN(config),Cell(dims))
currently using Class implementation
  some functions might have different versions deppending on the takes
  this might be better implemented as different functions (instead of class methods)

-- QHPCACTX

hpca representation is not representing 'front end' information
rather hpca holds indices/keys that can be used to trigger filling
  ctx queries hpca, hpca feeds declarative bits of info into cortex situational representation. 

when/what encode/retrieve?
  hpca records parameter values (role:filler) as conjunctive codes 
  these conjunctive codes can be inputted into cortical representation]

dynamics: obs info, retireve or not, predict next state
every transition is conditioned on the filling value of a role, hence require filler info to predict
cortex buffer holds some kv params, but sometimes you dont have full info.
  minmatch controls how much match between current cortex buffer and saved info in hpca required for recollection

i think there is a softmax read-out from cortex buffer ()

large chunks good, but nick pointed out small chunnks afford better generalization

"""


"""
instead of changing csw, i am going to implement metalearn in a notebook of its own
embed: csw embeds ids, metalearn takes function values (no embedding)
output: csw outputs softmax predictions, metalearn is regression problem

"""

VERB = False

"""
cell
  in/out/cell size
rnn
  vocab size 
"""

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




class BaseGraph():
  """ 
  
  tensorflow backbone: setup graph, sess, datafeed, placeholders
  """

  def __init__(self,saving=None):


    self.saving = saving
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph)
    self.dt_str = get_dt() 

    # task/architecture info
    # inherited: rnn: (_vocab_size,num_unrolls), cell:( _embed_size) 
    self.input_seq_len = input_seq_len = self.num_timesteps * self.num_unrolls 
    self.output_seq_len = output_seq_len = self.num_timesteps * self.num_unrolls
    
    # build tensorflow graph
    self.setup_graph(input_seq_len,output_seq_len)
    return None


  def setup_graph(self,input_seq_len,output_seq_len):
    """ operations within graph
    this should make explicit what is getting passed where
    some self. variables are stored but not passed if they are used elsewhere
    """
    with self.graph.as_default():
      self.cell.build()
      with tf.name_scope('dataset'):
        ## placeholders: xph,yph,batch_size_ph,dropout_keep_prob,embed_switch
        xph,yph = self.setup_placeholders(input_seq_len,output_seq_len)
        ## dataset and iterator: iterator, itr_initop, x_batch_ids, y_batch_ids
        x_batch_ids, y_batch_ids = self.setup_iterator(xph,yph)
      #@# MOVE EMBEDDING MATRIX BUILDING INTO RNN, EMBEDDING INTO CELL
      ## inference
      with tf.name_scope('inference'):
        self.setup_embed_mat(self._vocab_size,self.cell._embed_size)
        x_batch_embedded = self.embed(x_batch_ids)
        y_hat = self.y_hat = self.compute_output(x_batch_embedded) # outprojection
      ## optimization op
      with tf.name_scope('optimization'):
        loss_op = self.setup_loss_op(predict=y_hat,labels=y_batch_ids)
        self.setup_optimizer(loss_op)
      ## accuracy
      with tf.name_scope('accuracy'):
        # self.y_hat_ids = y_hat_ids = self.get_closest_embed_id(y_hat)
        y_hat_ids = self.y_hat_ids = tf.argmax(y_hat,axis=2)
        self.setup_acc_op(y_hat_ids,y_batch_ids) 
        # self.setup_mean_acc()
      ## summary
      with tf.name_scope('extra'):
        self.sess.run(tf.global_variables_initializer())
        # self.write_summary_op = tf.summary.merge_all()
        # self.saver_op = tf.train.Saver()
        # self.save_model('model-initial-%s'%self.dt_str)    
    return None

  def reinitialize(self):
    print('reinitializing rnn',dt.now())
    with self.graph.as_default():
      self.sess.run(tf.global_variables_initializer())
    return None

  ## dataset

  def setup_placeholders(self,xph_dim,yph_dim):
    self.xph = xph = tf.placeholder(tf.int32,
                  shape=[None,xph_dim],
                  name="xdata_placeholder")
    self.yph = yph = tf.placeholder(tf.int32,
                  shape=[None,yph_dim],
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
    return xph,yph

  def setup_iterator(self,xph,yph):
    """ returns an iterator which can be initialized 
        in a session, requires feed_dict = {xph,yph,batch_size_ph}
    """ 
    ## dataset
    dataset = tf.data.Dataset.from_tensor_slices((xph,yph))
    dataset = dataset.shuffle(10000)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size_ph))

    ## iterator
    self.iterator = iterator = tf.data.Iterator.from_structure(
                dataset.output_types, dataset.output_shapes)
    # initializer
    self.itr_initop = iterator.make_initializer(dataset)
    # get next batch of ids
    x_batch_ids,y_batch_ids = self.x_batch_ids,self.y_batch_ids = iterator.get_next()
    return x_batch_ids,y_batch_ids

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

  ## loss and optimization

  def setup_loss_op(self,predict,labels):
    """ labels [?,T,dim] is a batch of embedding vector seqeunces returned by embedding_lookup 
        predict [?,T,dim] is the tensor returned by inference, given x_batch

    """
    labels_onehot = tf.one_hot(indices=labels,depth=self._vocab_size)
    loss_op = self.loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_onehot,logits=predict)
    tf.summary.scalar('loss',loss_op)
    return loss_op 

  def setup_optimizer(self,loss_op_):
    """ returns gradient_minimizer and loss_op
        if no loss_op is passed, constructs MSE
    """
    loss_op = loss_op_
    optimizer_op = tf.train.GradientDescentOptimizer(0.005)
    grads_and_vars = optimizer_op.compute_gradients(loss_op)
    self.updateparams_op = optimizer_op.apply_gradients(grads_and_vars)
    return None

  ## evaluation

  def setup_acc_op(self,y_hat_ids,y_batch_ids):
    """ given predicted and actual ids,
    """
    y_hat_ids = tf.cast(y_hat_ids,tf.int32)
    y_batch_ids = tf.cast(y_batch_ids,tf.int32)
    eq = tf.equal(y_hat_ids,y_batch_ids)
    self.acc_op = tf.cast(eq, tf.float32)
    return None

  def get_closest_embed_id(self,embed_vecs):
    """ 
    embed_vecs [batch_size,time_steps,embed_dim] 
    embed_mat [vocab,embed_dim]
    closest_embed_id [batch_size,time_steps]
    """
    embed_mat = self.embed_mat
    ## normalize: assume last axis is embed_dim
    normed_emat = tf.nn.l2_normalize(embed_mat,axis=-1,name='normed_embed_mat')
    normed_evecs = tf.nn.l2_normalize(embed_vecs,axis=-1,name='normed_embed_vecs') 
    # compute cosine similarity: map matmul over batch axis
    lambda_matmul = lambda vec_arr: tf.matmul(normed_emat,vec_arr,transpose_b=True,name='cos_sim_fun')
    # [batch,vocab,time]
    cos_batch = tf.map_fn(lambda_matmul,normed_evecs,name='cos_sim_batch') 
    # take max similar over vocab axis
    closest_embed_id = tf.argmax(cos_batch,axis=1)
    return closest_embed_id

  ## recording data

  def setup_data_summary(self):
    merge_op = tf.summary.merge_all()
    data_summary = self.sess.run(merge_op)
    return data_summary

  # embedding

  def setup_embed_mat(self,vocab_size,embed_dim):
    """ embedding matrix made of two parts a fixed part and a random part
    """
    self.embed_mat1  = tf.get_variable(
                          name="embed_mat1",
                          shape=[vocab_size,embed_dim],
                          dtype=tf.float32,
                          trainable=True)
    self.embed_mat2  = tf.get_variable(
                          name="embed_mat2",
                          shape=[vocab_size,embed_dim],
                          dtype=tf.float32,
                          trainable=True)
    return None 

  def embed(self,embed_ids):
    """ 
    takes in embed_ids: batch of embed_id sequences [[1,2],[3,2],[5,1]]
    returns batch of embed_vec sequences [[v1,v2],[v3,v2],[v5,v1]]
    """
    embed_ids = tf.cast(embed_ids,tf.int32)
    embed_vec = tf.cond(
      tf.equal(self.embedding_switch,1),
      true_fn = lambda: tf.nn.embedding_lookup(self.embed_mat1, embed_ids),
      false_fn = lambda: tf.nn.embedding_lookup(self.embed_mat2, embed_ids)
    )
    # embed_vec = tf.nn.embedding_lookup(self.embed_mat, embed_ids)    
    return embed_vec

