# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Module for constructing RNN Cells."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from tensorflow.contrib.compiler import jit
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest




class LSTMBaseline(rnn_cell_impl.RNNCell):


  def __init__(self, num_units, forget_bias=1.0,
               input_size=None, activation=math_ops.tanh,
               layer_norm=True, norm_gain=1.0, norm_shift=0.0,
               dropout_keep_prob=1.0, dropout_prob_seed=None,
               reuse=None):

    super(LSTMBaseline, self).__init__(_reuse=reuse)

    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)

    self._num_units = num_units
    self._activation = activation
    self._forget_bias = forget_bias
    self._keep_prob = dropout_keep_prob
    self._seed = dropout_prob_seed
    self._layer_norm = layer_norm
    self._g = norm_gain
    self._b = norm_shift
    self._reuse = reuse

  @property
  def state_size(self):
    return rnn_cell_impl.LSTMStateTuple(self._num_units, self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def _norm(self, inp, scope):
    shape = inp.get_shape()[-1:]
    gamma_init = init_ops.constant_initializer(self._g)
    beta_init = init_ops.constant_initializer(self._b)
    with vs.variable_scope(scope):
      # Initialize beta and gamma for use by layer_norm.
      vs.get_variable("gamma", shape=shape, initializer=gamma_init)
      vs.get_variable("beta", shape=shape, initializer=beta_init)
    normalized = layers.layer_norm(inp, reuse=True, scope=scope)
    return normalized

  def _linear(self, args, num_outs):
    out_size = num_outs * self._num_units
    proj_size = args.get_shape()[-1]
    weights = vs.get_variable("kernel", [proj_size, out_size])
    out = math_ops.matmul(args, weights)
    if not self._layer_norm:
      bias = vs.get_variable("bias", [out_size])
      out = nn_ops.bias_add(out, bias)
    return out

  def call(self, inputs, state):
    """LSTM cell with layer normalization and recurrent dropout."""
    c, h = state
    args = array_ops.concat([inputs, h], 1)
    concat = self._linear(args,num_outs=2)

    j, f = array_ops.split(value=concat, num_or_size_splits=2, axis=1)
    if self._layer_norm:
      j = self._norm(j, "transform")
      f = self._norm(f, "forget")

    g = self._activation(j)
    if (not isinstance(self._keep_prob, float)) or self._keep_prob < 1:
      g = nn_ops.dropout(g, self._keep_prob, seed=self._seed)
    ##
    self.forget_act = math_ops.sigmoid(f)
    ##
    new_c = (c * math_ops.sigmoid(f + self._forget_bias) + g)
    if self._layer_norm:
      new_c = self._norm(new_c, "state")
    new_h = self._activation(new_c) 

    new_state = rnn_cell_impl.LSTMStateTuple(new_c, new_h)
    return new_h, new_state




class LSTMFgatePE(rnn_cell_impl.RNNCell):


  def __init__(self, num_units, forget_bias=1.0,
               input_size=None, activation=math_ops.tanh,
               layer_norm=True, norm_gain=1.0, norm_shift=0.0,
               dropout_keep_prob=1.0, dropout_prob_seed=None,
               reuse=None):

    super(LSTMFgatePE, self).__init__(_reuse=reuse)

    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)

    self._num_units = num_units
    self._activation = activation
    self._forget_bias = forget_bias
    self._keep_prob = dropout_keep_prob
    self._seed = dropout_prob_seed
    self._layer_norm = layer_norm
    self._g = norm_gain
    self._b = norm_shift
    self._reuse = reuse

  @property
  def state_size(self):
    return rnn_cell_impl.LSTMStateTuple(self._num_units, self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def _norm(self, inp, scope):
    shape = inp.get_shape()[-1:]
    gamma_init = init_ops.constant_initializer(self._g)
    beta_init = init_ops.constant_initializer(self._b)
    with vs.variable_scope(scope):
      # Initialize beta and gamma for use by layer_norm.
      vs.get_variable("gamma", shape=shape, initializer=gamma_init)
      vs.get_variable("beta", shape=shape, initializer=beta_init)
    normalized = layers.layer_norm(inp, reuse=True, scope=scope)
    return normalized

  def _linear(self, args, num_outs):
    out_size = num_outs * self._num_units
    proj_size = args.get_shape()[-1]
    weights = vs.get_variable("kernel", [proj_size, out_size])
    out = math_ops.matmul(args, weights)
    if not self._layer_norm:
      bias = vs.get_variable("bias", [out_size])
      out = nn_ops.bias_add(out, bias)
    return out

  def call(self, inputs, state):
    """
    Assume inputs = tf.concat([xbatch,PE])
    LSTM cell with layer normalization and recurrent dropout.
    """
    c, h = state
    x,pe = array_ops.split(value=inputs, num_or_size_splits=2, axis=1)
    args = array_ops.concat([x, h, pe, h], 1)
    concat = self._linear(args,num_outs=2)
    j,f = array_ops.split(value=concat, num_or_size_splits=2, axis=1)
    g = self._activation(j)
    ## layer norm and dropout
    if self._layer_norm:
      j = self._norm(j, "transform")
      f = self._norm(f, "forget")
    if (not isinstance(self._keep_prob, float)) or self._keep_prob < 1:
      g = nn_ops.dropout(g, self._keep_prob, seed=self._seed)
    ## update cell state
    new_c = (c * math_ops.sigmoid(f + self._forget_bias) + g)
    if self._layer_norm:
      new_c = self._norm(new_c, "state")
    new_h = self._activation(new_c) 
    ## strange thing 
    new_state = rnn_cell_impl.LSTMStateTuple(new_c, new_h)
    ## for recording 
    self.forget_act = math_ops.sigmoid(f)
    return new_h, new_state


