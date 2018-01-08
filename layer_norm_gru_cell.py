import collections
import math

from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn_cell

import tensorflow as tf

try:
  linear = tf.nn.rnn_cell.linear
except:
  from tensorflow.python.ops.rnn_cell import _linear as linear


def ln(input, s, b, epsilon = 1e-5, max = 1000):
    """ Layer normalizes a 2D tensor along its second axis, which corresponds to batch """
    m, v = tf.nn.moments(input, [1], keep_dims=True)
    normalised_input = (input - m) / tf.sqrt(v + epsilon)
    return normalised_input * s + b



class LNGRUCell(rnn_cell.RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

  def __init__(self, num_units, input_size=None, activation=tanh, reuse=False):
    if input_size is not None:
      print("%s: The input_size parameter is deprecated." % self)
    self._num_units = num_units
    self._activation = activation
    self._reuse = reuse

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    dim = self._num_units
    with vs.variable_scope(scope or type(self).__name__):  # "GRUCell"
      with vs.variable_scope("gru_cell", reuse=self._reuse):  # Reset gate and update gate.
        # We start with bias of 1.0 to not reset and not update.
        with vs.variable_scope( "Layer_Parameters"):

          s1 = vs.get_variable("s1", initializer=tf.ones([2*dim]), dtype=tf.float32)
          s2 = vs.get_variable("s2", initializer=tf.ones([2*dim]), dtype=tf.float32)
          s3 = vs.get_variable("s3", initializer=tf.ones([dim]), dtype=tf.float32)
          s4 = vs.get_variable("s4", initializer=tf.ones([dim]), dtype=tf.float32)
          b1 = vs.get_variable("b1", initializer=tf.zeros([2*dim]), dtype=tf.float32)
          b2 = vs.get_variable("b2", initializer=tf.zeros([2*dim]), dtype=tf.float32)
          b3 = vs.get_variable("b3", initializer=tf.zeros([dim]), dtype=tf.float32)
          b4 = vs.get_variable("b4", initializer=tf.zeros([dim]), dtype=tf.float32)


          # Code below initialized for all cells
          # s1 = tf.Variable(tf.ones([2 * dim]), name="s1")
          # s2 = tf.Variable(tf.ones([2 * dim]), name="s2")
          # s3 = tf.Variable(tf.ones([dim]), name="s3")
          # s4 = tf.Variable(tf.ones([dim]), name="s4")
          # b1 = tf.Variable(tf.zeros([2 * dim]), name="b1")
          # b2 = tf.Variable(tf.zeros([2 * dim]), name="b2")
          # b3 = tf.Variable(tf.zeros([dim]), name="b3")
          # b4 = tf.Variable(tf.zeros([dim]), name="b4")

        input_below_ = rnn_cell._linear([inputs],
                               2 * self._num_units, False, scope="out_1")
        input_below_ = ln(input_below_, s1, b1)
        state_below_ = rnn_cell._linear([state],
                               2 * self._num_units, False, scope="out_2")
        state_below_ = ln(state_below_, s2, b2)
        out =tf.add(input_below_, state_below_)
        r, u = array_ops.split(1, 2, out)
        r, u = sigmoid(r), sigmoid(u)

      with vs.variable_scope("Candidate"):
          input_below_x = rnn_cell._linear([inputs],
                                           self._num_units, False, scope="out_3")
          input_below_x = ln(input_below_x, s3, b3)
          state_below_x = rnn_cell._linear([state],
                                           self._num_units, False, scope="out_4")
          state_below_x = ln(state_below_x, s4, b4)
          c_pre = tf.add(input_below_x,r * state_below_x)
          c = self._activation(c_pre)
      new_h = u * state + (1 - u) * c
    return new_h, new_h
