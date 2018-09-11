import tensorflow as tf
import numpy as np
from cswNetEngine import CSW
  
VERB = False
TWO_SCHEMAS = True


## train and eval

class Trainer():

  def __init__(self,rnn):
    # data generating
    # still unsure if unconditioned is working
    self.csw = csw = CSW(conditioned=True)
    self.Xfull = self.get_Xfull() 
    self.rnn = rnn
    self.dropout_pr = 0.8

  # embeddings

  def get_embed_id(self,block_num,twosch=TWO_SCHEMAS):
    """ currently schema_id can either be 1 or 0
    """
    if twosch:
      return block_num%2
    else:
      return 1

  # training modes

  def metalearn_blocktrain(self,num_blocks,num_epochs_per_block):
    """ 
    use num_blocks=1 for simpletrain
    """
    num_evals_per_block = np.maximum(1,num_epochs_per_block/10)
    Xdata,Ydata = gen_poly_dataset(_len)
    # recording network performance
    sess_eval_L,sess_pred_L = [],[]
    # training block
    for block_num in range(num_blocks):
      block_eval_L,block_pred_L = [],[]
      ## train loop
      for epoch_num in range(num_epochs_per_block): 
        # generate train data, update params
        Xtrain,Ytrain = self.gen_train_data(block_cond)
        self.train_step(Xtrain,Ytrain,embed_switch=block_embed_id)
        ## eval 
        if epoch_num%(num_epochs_per_block//num_evals_per_block)==0: 
          print(block_num,epoch_num)
          # evaluate (loss,acc) on train data, predict (yhat) on full dataset, 
          eval_data = self.eval(Xtrain,Ytrain,embed_switch=block_embed_id)
          pred_data = self.predict(self.Xfull,self.Xfull,embed_switch=0)
          # collect data
          eval_data['cond'] = block_cond
          block_eval_L.append(eval_data)
          block_pred_L.append(pred_data)
      # data from block
      sess_eval_L.append(block_eval_L)
      sess_pred_L.append(block_pred_L)
    return np.array(sess_eval_L),np.array(sess_pred_L)

  ## CSW

  def blocktrain(self,num_blocks,num_epochs_per_block):
    """ 
    use num_blocks=1 for simpletrain
    """
    num_evals_per_block = np.maximum(1,num_epochs_per_block/10)

    # recording network performance
    sess_eval_L,sess_pred_L = [],[]
    # training block
    for block_num in range(num_blocks):
      block_eval_L,block_pred_L = [],[]
      ## block variables
      if block_num%2 == 0: 
        block_cond = 'location.latent.false' # 15
      else: 
        block_cond = 'location.latent.true' # 14
      block_embed_id = self.get_embed_id(block_num)
      ## train loop
      for epoch_num in range(num_epochs_per_block): 
        # generate train data, update params
        Xtrain,Ytrain = self.gen_train_data(block_cond)
        self.train_step(Xtrain,Ytrain,embed_switch=block_embed_id)
        ## eval 
        if epoch_num%(num_epochs_per_block//num_evals_per_block)==0: 
          print(block_num,epoch_num)
          # evaluate (loss,acc) on train data, predict (yhat) on full dataset, 
          eval_data = self.eval(Xtrain,Ytrain,embed_switch=block_embed_id)
          pred_data = self.predict(self.Xfull,self.Xfull,embed_switch=0)
          # collect data
          eval_data['cond'] = block_cond
          block_eval_L.append(eval_data)
          block_pred_L.append(pred_data)
      # data from block
      sess_eval_L.append(block_eval_L)
      sess_pred_L.append(block_pred_L)
    return np.array(sess_eval_L),np.array(sess_pred_L)

  def sesstrain(self,blocking):
    """ blocking is list of tuples [(num_blocks,num_epochs)]
        each tuple give num_blocks and num_evals for session
    """
    eval_data,pred_data = [],[]
    for sess_num,(num_blocks,num_epochs) in enumerate(blocking):
      print('sess',sess_num)
      eval_arr,pred_arr = self.blocktrain(num_blocks,num_epochs)
      eval_data.append(eval_arr)
      pred_data.append(pred_arr)
    return np.array(eval_data),np.array(pred_data)

  # training (update params), evaluation (loss, acc) and prediction (yhat)

  def train_step(self,Xdata,Ydata,embed_switch):
    """ does a full pass over dataset """
    # initialize iterator with train data
    rnn = self.rnn
    batch_size = 1
    train_feed_dict = {
      rnn.xph:Xdata,
      rnn.yph:Ydata,
      rnn.batch_size_ph:batch_size,
      rnn.dropout_keep_prob: self.dropout_pr,
      rnn.embedding_switch: embed_switch}
    
    rnn.sess.run([rnn.itr_initop],train_feed_dict)
    # train loop
    while True:
      try:
        _ = rnn.sess.run([rnn.updateparams_op],feed_dict=train_feed_dict)
        # print(loss)
      except tf.errors.OutOfRangeError:
        break
    return None

  def eval(self,Xeval,Yeval,embed_switch):
    """ makes predictions and computes loss
    """

    rnn = self.rnn
    sess = self.rnn.sess

    # initialize data datastructures
    eval_batch_size = len(Xeval)
    eval_array_dtype = [('loss','float32'),
                        ('acc','float32',(2)),
                        ('cond','25str')
    ]
    eval_data_arr = np.zeros((),dtype=eval_array_dtype)

    eval_feed_dict = {
      rnn.xph:Xeval,
      rnn.yph:Yeval,
      rnn.batch_size_ph:eval_batch_size,
      rnn.dropout_keep_prob: 1.0,
      rnn.embedding_switch: embed_switch
    }

    # initialize iterator with eval data
    sess.run(rnn.itr_initop,eval_feed_dict)
    # eval loop
    while True:
      try:
        # acc,summary = sess.run([rnn.acc_op,rnn.write_summary_op],self.eval_feed_dict)
        acc,loss = sess.run([rnn.acc_op,rnn.loss_op],eval_feed_dict)
        eval_data_arr['acc'] = np.mean(acc,0)
        eval_data_arr['loss'] = np.mean(loss)
      except tf.errors.OutOfRangeError:
        break
    return eval_data_arr

  def predict(self,Xpred,Ypred,embed_switch):
    """ makes predictions on full dataset
    currently predictions are made on both contexts using same embedding
    ideally i could make predictions on each context independently 
    so that could make predictions with the appropriate embedding
    e.g. include an argument in predict which says which Xdata to pass
      then pass XfullA with embed_switch 0, XfullB with embed_switch 1
    """

    rnn = self.rnn
    sess = self.rnn.sess

    pred_batch_size = len(Xpred)

    # initialize data datastructures
    pred_array_dtype = [('xbatch','int32',(pred_batch_size,2)),
                        ('yhat','int32',(pred_batch_size,2)),
    ]
    pred_data_arr = np.zeros((),dtype=pred_array_dtype)

    pred_feed_dict = {
      rnn.xph:Xpred,
      rnn.yph:Ypred,
      rnn.batch_size_ph: pred_batch_size,
      rnn.dropout_keep_prob: 1.0,
      rnn.embedding_switch: embed_switch
    }
    # initialize iterator with eval data
    sess.run(rnn.itr_initop,pred_feed_dict)
    # eval loop
    while True:
      try:
        xbatch,yhat = sess.run([rnn.x_batch_ids,rnn.y_hat_ids],
                                       feed_dict=pred_feed_dict)
        pred_data_arr['xbatch'] = xbatch
        pred_data_arr['yhat'] = yhat
      except tf.errors.OutOfRangeError:
        break
    return pred_data_arr

  # CSW generating datasets

  def gen_train_data(self,cond):
    """ 
    used to generate training data 
    Xtrain and Ytrain are tuples [node,cond_id]
    using self.csw 
    """
    if self.csw.conditioned:
      assert cond, 'specify cond to gen sample from conditioned csw'
    # generate path
    if self.csw.conditioned: transition_condition = cond
    else: transition_condition = None
    path = self.csw.gen_path(transition_condition)
    # initialize data array
    node_id = path.pop(0) # first node in X
    cond_id = self.csw.cond_dict[cond]
    Xdata = [[node_id,cond_id]]
    Ydata = []
    # walk and collect
    while len(path):
      node_id = path.pop(0)
      node_cond_tup = [node_id,cond_id]
      Ydata.append(node_cond_tup)
      Xdata.append(node_cond_tup)
    # last node in path: only in Y
    Xdata.pop(-1) 
    return np.array(Xdata),np.array(Ydata)

  def get_Xfull(self):
    """ used for evaluating 
    make dataset with [node_id,cond_id] datapoints for every frnode
    transitions contained in edge_dict
    """
    Xdata,Ydata = [],[]
    cond_id_L = self.csw.cond_dict.values()
    frnode_L = [i['nodeid'] for i in self.csw.graph.values()]
    for cond_id in cond_id_L:
      for frnode_id in frnode_L:
        X_node_cond_tup = (frnode_id,cond_id)
        Xdata.append(X_node_cond_tup)
    Xfull = np.array(sorted([i for i in set(Xdata)]))
    return Xfull


# metalearn generating datasets

def scale(_input, _min, _max):

  """ scales array to lie between [min,max]
  """
  _input += -(np.min(_input))
  _input /= np.max(_input) / (_max - _min)
  _input += _min
  return _input

def quad_fun(coefs,_len):
  """ 
  coefs is 5-tuple 
  domain [-1,1] range [0.2,0.8]
  """
  a,b,c,d,e,f = coefs
  fx = lambda x1,x2: a*(x1**2) + b*(x2**2) + (c*x1*x2) + (c*x1) + (d*x2) + e + f
  X = np.array([(i,j) for i,j in itertools.permutations(np.linspace(-1,1,_len),2)])
  y = [fx(x[0],x[1]) for x in X]
  y = scale(y,0.2,0.8)
  return X,y

def gen_poly_dataset(_len):
  """ 
  returns Xdata: 3-tuples arr [x1[t],x2[t],y[t-1]]
  and corresponding Ydata arr [y[t]]
  NB: _len will get amplified
  """
  coefs = np.random.uniform(0,1,6)
  X,Y = quad_fun(coefs,_len)
  Xdata,Ydata = [],[]
  for t in range(1,len(X)):
    x1,x2 = X[t]
    yt,yt1 = Y[t],Y[t-1]
    Xdata.append([x1,x2,yt1])
    Ydata.append([yt])
  Xdata = np.array(Xdata)
  Ydata = np.array(Ydata)
  return Xdata,Ydata