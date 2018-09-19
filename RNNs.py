import tensorflow as tf

"""
RNNs define task structure
"""

### RNNs

def basicRNN(self,depth,in_len,out_len):
  """ 
  self: a tensorflow graph 
  RNN structure:
    takes in state and a filler
    returns prediction for next state and a filler
  returns unscaled logits
  """
  xbatch = self.xbatch
  cell = self.cell

  xbatch = tf.layers.dense(xbatch,self.rnn_size,tf.nn.relu,name='inproj')
  # unroll RNN
  with tf.variable_scope('RNN_SCOPE') as cellscope:
    # initialize state
    initial_state = state = cell.zero_state(tf.cast(self.batch_size_ph,tf.int32),tf.float32)
    # unroll
    outputL = []
    for unroll_step in range(depth):
      xroll = xbatch[:,unroll_step,:,:]
      # input
      for in_tstep in range(in_len):
        __,state = cell(xroll[:,in_tstep,:], state)
        cellscope.reuse_variables()
      # output: inputs are zeroed out
      outputs_rs = []
      for out_tstep in range(out_len):
        zero_input = tf.zeros_like(xroll)
        cell_output, state = cell(zero_input[:,out_tstep,:], state) 
        outputs_rs.append(cell_output)
      outputs_rollstep = tf.stack(outputs_rs,axis=1)
      outputL.append(outputs_rollstep)
  # format for y_hat
  outputs = tf.stack(outputL,axis=1)
  # project to unscaled logits (to that outdim = num_classes)
  outputs = tf.layers.dense(outputs,self.num_classes,tf.nn.relu,name='outproj_unscaled_logits')
  return outputs



def RNN_onesent(self,depth,in_len,out_len):
  """ 
  self: a tensorflow graph 
  RNN structure:
    takes in state and a filler
    returns prediction for next state and a filler
  returns unscaled logits
  """
  xbatch = self.xbatch
  cell = self.cell
  xbatch = tf.layers.dense(xbatch,self.rnn_size,tf.nn.relu,name='inproj')
  # unroll RNN
  with tf.variable_scope('RNN_SCOPE') as cellscope:
    # initialize state
    initial_state = state = cell.zero_state(tf.cast(self.batch_size_ph,tf.int32),tf.float32)
    # unroll
    outputL = []
    for unroll_step in range(depth):
      xroll = xbatch[:,unroll_step,:,:]
      # input
      for in_tstep in range(in_len):
        __,state = cell(xroll[:,in_tstep,:], state)
        cellscope.reuse_variables()
      # output: inputs are zeroed out
      outputs_rs = []
      for out_tstep in range(out_len):
        zero_input = tf.zeros_like(xroll)
        cell_output, state = cell(zero_input[:,out_tstep,:], state) 
        outputs_rs.append(cell_output)
      outputs_rollstep = tf.stack(outputs_rs,axis=1)
      outputL.append(outputs_rollstep)
  # format for y_hat
  outputs = tf.stack(outputL,axis=1)
  # project to unscaled logits (to that outdim = num_classes)
  outputs = tf.layers.dense(outputs,self.num_classes,tf.nn.relu,name='outproj_unscaled_logits')
  return outputs