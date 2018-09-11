import tensorflow as tf
from tensorflow.python.ops import nn_ops
import collections


"""
NB inputs_shape is required by parent LayerRNNCell.build but unused here

defines an input/output map: i.e. the computation implemented. 
"""


class CustomLSTM():

  def __init__(self,input_size_,state_size_,output_size_):
    self._input_size = input_size_
    self._state_size =  state_size_
    self._output_size = output_size_
    # layer norm
    self._layer_norm = False
    self._norm_gain = 1.0
    self._norm_shift = 0.0
    self.built = False

  @property
  def input_size(self):
    """
    """
    return self._input_size
  @property
  def state_size(self):
    return self._state_size
  @property 
  def output_size(self):
    return self._output_size


  def zero_state(self,batch_size_):
    """
    """
    return tf.zeros(shape=[batch_size_,self._state_size])

  def _norm(self, inp, scope, dtype=tf.float32):
    """ copied from tf LayerNormBasicLSTM"""
    
    from tensorflow.contrib.layers.python.layers import layers
    from tensorflow.python.ops import variable_scope as vs
    from tensorflow.python.ops import init_ops

    shape = inp.get_shape()[-1:]
    gamma_init = init_ops.constant_initializer(self._norm_gain)
    beta_init = init_ops.constant_initializer(self._norm_shift)
    with vs.variable_scope(scope):
      # Initialize beta and gamma for use by layer_norm.
      vs.get_variable("gamma", shape=shape, initializer=gamma_init, dtype=dtype)
      vs.get_variable("beta", shape=shape, initializer=beta_init, dtype=dtype)
    normalized = layers.layer_norm(inp, reuse=True, scope=scope)
    return normalized

  def build(self):
    """ stored tensors corresponding to each variable for later inspection
    but concatenated for matmul
    """
    # input gate, 
    self._kernel_in2igate = tf.get_variable('kernel_in2igate',
                              shape=[self._input_size, self._state_size])
    self._kernel_st2igate = tf.get_variable('kernel_st2igate',
                              shape=[self._state_size, self._state_size])
    # forget gate
    self._kernel_in2fgate = tf.get_variable('kernel_in2fgate',
                              shape=[self._input_size, self._state_size])
    self._kernel_st2fgate = tf.get_variable('kernel_st2fgate',
                              shape=[self._state_size, self._state_size])
    # output gate
    self._kernel_in2ogate = tf.get_variable('kernel_in2ogate',
                              shape=[self._input_size, self._state_size])
    self._kernel_st2ogate = tf.get_variable('kernel_st2ogate',
                              shape=[self._state_size, self._state_size])
    # cell state
    self._kernel_in2state = tf.get_variable('kernel_in2state',
                              shape=[self._input_size, self._state_size])
    self._kernel_st2state = tf.get_variable('kernel_st2state',
                              shape=[self._state_size, self._state_size])
    # output projection
    self._kernel_st2out = tf.get_variable('kernel_st2out',
                              shape=[self._state_size, self._output_size])

    # concat [input,state] along input dimension
    # for performance i do T*[x,h] (concatenate x and h)
    # the above is left for later inspection
    self._kernel_igate = i = tf.concat([self._kernel_in2igate,
                                    self._kernel_st2igate],
                                    axis=0)
    self._kernel_fgate = f = tf.concat([self._kernel_in2fgate,
                                    self._kernel_st2fgate],
                                    axis=0)
    self._kernel_ogate = o = tf.concat([self._kernel_in2ogate,
                                    self._kernel_st2ogate],
                                    axis=0)
    self._kernel_state = j = tf.concat([self._kernel_in2state,
                                    self._kernel_st2state],
                                    axis=0)
    # concat [input_gate, forget_gate, output_gate, state_gate] along output dimension
    self._kernel = tf.concat([self._kernel_igate,self._kernel_fgate,
                              self._kernel_ogate,self._kernel_state],
                              axis=1)
    # bias
    self._bias = tf.get_variable("BIAS",
                      shape=[4*self._state_size],
                      initializer=tf.zeros_initializer)
    self.built = True
    return None

  def __call__(self,input_,old_state,scope=None):
    """  
    input_ dim is kept to [batch,time,dim]
      if input_ is not vectors, dim=1
    """

    if not self.built:
      self.build()

    gate_act_fun = tf.sigmoid
    state_act_fun = tf.tanh

    with tf.name_scope('BASIC_LSTM_CELL'):
      # concat input and state
      input_and_state = tf.concat([input_,old_state],axis=1)
      # compute preactivation 
      preact = tf.nn.bias_add(tf.matmul(input_and_state,self._kernel),self._bias)
      # simplit preactivations
      (igate_preact,fgate_preact,ogate_preact,state_preact
        ) = tf.split(preact,num_or_size_splits=4,axis=1)

      ## LAYER NORM
      if self._layer_norm:
        igate_preact = self._norm(igate_preact, "input")
        state_preact = self._norm(state_preact, "transform")
        fgate_preact = self._norm(fgate_preact, "forget")
        ogate_preact = self._norm(ogate_preact, "output")

      # state activation
      state_act = state_act_fun(state_preact)
      # dropout
      # state_act = nn_ops.dropout(state_act,self.dropout_keep_prob)

      # compute new cell state
      new_cell_state =  gate_act_fun(igate_preact) * state_act \
                      - gate_act_fun(fgate_preact) * old_state
      # outgate
      gated_new_cell_state = gate_act_fun(ogate_preact) * new_cell_state
      # NB state is not output: output is gasted state projected to outdim
      output = tf.matmul(gated_new_cell_state,self._kernel_st2out)
      new_state = new_cell_state
    return output,new_state



# _LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))
# class LSTMStateTuple(_LSTMStateTuple):
#   """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.

#   Stores two elements: `(c, h)`, in that order. Where `c` is the hidden state
#   and `h` is the output.

#   Only used when `state_is_tuple=True`.
#   """
#   __slots__ = ()

#   @property
#   def dtype(self):
#     (c, h) = self
#     if c.dtype != h.dtype:
#       raise TypeError("Inconsistent internal state: %s vs %s" %
#                       (str(c.dtype), str(h.dtype)))
#     return c.dtype


# class CustomBasicLSTMCell(LayerRNNCell):

#   def __init__(self,input_size,state_size,output_size,forget_bias=1.0,state_is_tuple=True):
#     """
#     as in arxiv.org/pdf/1409.2329
#     output projection not implemented
#     """
#     super(CustomBasicLSTMCell, self).__init__()
#     self.built = False
#     # dimensions
#     self._input_size = input_size
#     self._state_size = state_size
#     self._output_size = output_size
#     assert input_size == state_size == output_size, 'output projection not implemented'
#     # 
#     self._forget_bias = forget_bias
#     self._state_is_tuple = state_is_tuple

#   @property
#   def input_size(self):
#     return self._input_size
#   @property
#   def state_size(self):
#     return self._state_size
#   @property 
#   def output_size(self):
#     return self._output_size


#   def build(self,inputs_shape):
#     """ stored tensors corresponding to each variable for later inspection
#     but concatenated for matmul
#     """
#     # input gate, 
#     self._kernel_in2igate = self.add_variable('_kernel_in2igate',
#                               shape=[self._input_size, self._state_size])
#     self._kernel_st2igate = self.add_variable('_kernel_st2igate',
#                               shape=[self._state_size, self._state_size])
#     # forget gate
#     self._kernel_in2fgate = self.add_variable('_kernel_in2fgate',
#                               shape=[self._input_size, self._state_size])
#     self._kernel_st2fgate = self.add_variable('_kernel_st2fgate',
#                               shape=[self._state_size, self._state_size])
#     # output gate
#     self._kernel_in2ogate = self.add_variable('_kernel_in2ogate',
#                               shape=[self._input_size, self._state_size])
#     self._kernel_st2ogate = self.add_variable('_kernel_st2ogate',
#                               shape=[self._state_size, self._state_size])
#     # cell state
#     self._kernel_in2state = self.add_variable('_kernel_in2state',
#                               shape=[self._input_size, self._state_size])
#     self._kernel_st2state = self.add_variable('_kernel_st2state',
#                               shape=[self._state_size, self._state_size])

#     # concat [input,state] along input dimension
#     # for performance i do T*[x,h] (concatenate x and h)
#     # the above is left for later inspection
#     self._kernel_igate = tf.concat([self._kernel_in2igate,
#                                     self._kernel_st2igate],
#                                     axis=0)
#     self._kernel_fgate = tf.concat([self._kernel_in2fgate,
#                                     self._kernel_st2fgate],
#                                     axis=0)
#     self._kernel_ogate = tf.concat([self._kernel_in2ogate,
#                                     self._kernel_st2ogate],
#                                     axis=0)
#     self._kernel_state = tf.concat([self._kernel_in2state,
#                                     self._kernel_st2state],
#                                     axis=0)

#     # concat [input_gate, forget_gate, output_gate, state_gate] along output dimension
#     self._kernel = tf.concat([self._kernel_igate,self._kernel_fgate,
#                               self._kernel_ogate,self._kernel_state],
#                               axis=1)
#     # bias
#     self._bias = self.add_variable("BIAS",
#                       shape=[4*self._state_size],
#                       initializer=tf.zeros_initializer)

#     self.built = True
#     return None

#   def zero_state_(self,batch_size,dtype):
#     zero_state = self.zero_state(tf.cast(batch_size,tf.int32),dtype)
#     if self.zero_state is tuple():
#       return zero_state
#     else:
#       return tf.nn.rnn_cell.LSTMStateTuple(state, state)
#     # return zero_state_tuple

#   def call(self,input_,old_state,scope=None):
#     """  """
#     print('call')
#     if not self.built:
#       self.build(input_.shape)

#     gate_act_fun = tf.sigmoid
#     state_act_fun = tf.tanh
      
#     with tf.name_scope('BASIC_LSTM_CELL'):
#       # state is tuple
#       if self._state_is_tuple:
#         old_output,old_state = old_state

#       # concat input and state
#       input_and_state = tf.concat([input_,old_state],axis=1)
#       # compute preactivation 
#       preact = tf.nn.bias_add(tf.matmul(input_and_state,self._kernel),self._bias)
#       # simplit preactivations
#       (igate_preact,fgate_preact,ogate_preact,state_preact
#         ) = tf.split(preact,num_or_size_splits=4,axis=1)
#       # compute new cell state
#       new_cell_state =  gate_act_fun(igate_preact) * state_act_fun(state_preact) \
#                       - gate_act_fun(fgate_preact) * old_state
#       # output
#       output = gate_act_fun(ogate_preact) * new_cell_state
    
#       # state is tuple
#       if self._state_is_tuple:
#         new_state = LSTMStateTuple(new_cell_state, output)

#     return output,new_state






