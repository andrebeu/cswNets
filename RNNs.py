import tensorflow as tf
from BaseGraph import BaseGraph,get_dt
import Cells

VOCAB_SIZE = 20

""" 
each rnn: 
- initialized given a cell
- defines task structure (i/o mapping)
- inherets from basegraph

"""


class RNN_fullstory(BaseGraph):
  """ 
  """
  def __init__(self,cell):
    """ 
    num_timesteps used to setup placeholder 
    """
    self.cell = cell
    self.in_len = self.out_len = self.num_timesteps = 2
    self.num_unrolls = 1
    self._vocab_size = VOCAB_SIZE
    super().__init__() # BaseGraph

  def compute_output(self,x_batch):
    """ 
    x_batch [batch,time] 
      embedding done in cell, so cell_out [batch,time,dim]
    """

    # unroll RNN
    with tf.variable_scope('RNN_SCOPE') as cellscope:
      # setup RNN cell      
      cell = self.cell

      # initialize state
      self.initial_state = state = cell.zero_state(tf.cast(self.batch_size_ph,tf.int32))
      cell = tf.contrib.rnn.DropoutWrapper(cell,
              state_keep_prob=self.dropout_keep_prob,
              variational_recurrent=True,
              input_size=self.cell._input_size,
              dtype=tf.float32)
      
      # unroll
      logit_L = []
      for unroll in range(self.num_unrolls):
        # input
        for in_tstep in range(self.in_len):
          if in_tstep > 0: cellscope.reuse_variables()
          #@# WILL NEED TO CHANGE DIMENSION - PROBLEM WITH STATE CONCAT?
          __,state = cell(x_batch[:,in_tstep,:], state)
        # output: inputs are zeroed out
        for out_tstep in range(self.out_len):
          zero_input = tf.zeros_like(x_batch)
          cell_output, state = cell(zero_input[:,out_tstep,:], state) 
          # project to logit
          logit = self.compute_logit(cell_output,cellscope)
          logit_L.append(logit) 
    # format for y_hat
    logits = tf.stack(logit_L,axis=1)
    y_hat = tf.nn.softmax(logits)
    return y_hat

  def compute_logit(self,inpt,scope):
  	with tf.variable_scope('SM_SCOPE',reuse=tf.AUTO_REUSE):
	  	_kernel_out2logit = self.cell._kernel_out2logit = tf.get_variable(
	  		'kernel_out2logit', shape=[self.cell._output_size, self._vocab_size])
	  	logits = tf.matmul(inpt,_kernel_out2logit)
  	return logits
  

