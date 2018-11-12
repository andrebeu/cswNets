import tensorflow as tf

"""
RNNs define task structure
"""

### RNNs

def basicRNN(tfGraph,depth,in_len,out_len):
  """ 
  - tfGraph: 
      this awkward syntax allows modularizing RNNs

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
  xbatch = tfGraph.xbatch
  cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
          tfGraph.rnn_size,dropout_keep_prob=tfGraph.dropout_keep_prob)

  xbatch = tf.layers.dense(xbatch,tfGraph.rnn_size,tf.nn.relu,name='inproj')
  # unroll RNN
  with tf.variable_scope('RNN_SCOPE') as cellscope:
    # initialize state
    initial_state = state = cell.zero_state(tf.cast(tfGraph.batch_size_ph,tf.int32),tf.float32)
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
  outputs = tf.layers.dense(outputs,tfGraph.num_classes,tf.nn.relu,name='outproj_unscaled_logits')
  return outputs



def RNN_onesent(self,depth,in_len,out_len):
  """ 
  self: a tensorflow graph 
  consumes a story at a time
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