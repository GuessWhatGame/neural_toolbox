
# WARNING THIS IS NOT MY CODE: this is a pull request, that, hopefully will be merged soon to tf!

# source : https://github.com/tensorflow/tensorflow/pull/14578

"""Module for constructing RNN Cells."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs


class LayerNormBasicGRUCell(rnn_cell_impl.RNNCell):
  """GRU unit with layer normalization.
    This class adds layer normalization to a
    basic GRU unit. Layer normalization implementation is based on:
      https://arxiv.org/abs/1607.06450.
    "Layer Normalization"
    Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
    and is applied before the internal nonlinearities.
    """

  def __init__(self, num_units,
               activation=math_ops.tanh,
               layer_norm=True,
               norm_gain=1.0,
               norm_shift=0.0,
               reuse=None):
    """Initializes the cell.
        Args:
          num_units: int, The number of units in the GRU cell.
          activation: Activation function of the inner states.
          layer_norm: If `True`, layer normalization will be applied.
          norm_gain: float, The layer normalization gain initial value. If
            `layer_norm` has been set to `False`, this argument will be ignored.
          norm_shift: float, The layer normalization shift initial value. If
            `layer_norm` has been set to `False`, this argument will be ignored.
        """

    super(LayerNormBasicGRUCell, self).__init__(_reuse=reuse)

    self._num_units = num_units
    self._activation = activation
    self._layer_norm = layer_norm
    self._g = norm_gain
    self._b = norm_shift
    self._reuse = reuse

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def _linear(self, args, scope):
    out_size = self._num_units
    proj_size = args.get_shape()[-1]
    with vs.variable_scope(scope):
      weights = vs.get_variable("kernel", [proj_size, out_size])
      out = math_ops.matmul(args, weights)
      if not self._layer_norm:
        bias = vs.get_variable("bias", [out_size])
        out = nn_ops.bias_add(out, bias)
    return out

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

  def call(self, inputs, state):
    """GRU cell with layer normalization."""

    args = array_ops.concat([inputs, state], 1)

    z = self._linear(args, scope="update")
    r = self._linear(args, scope="reset")

    if self._layer_norm:
      z = self._norm(z, "update")
      r = self._norm(r, "reset")

    z = math_ops.sigmoid(z)
    r = math_ops.sigmoid(r)

    _x = self._linear(inputs, scope="candidate_linear_x")
    _h = self._linear(state, scope="candidate_linear_h")

    if self._layer_norm:
      _x = self._norm(_x, scope="candidate_linear_x")
      _h = self._norm(_h, scope="candidate_linear_h")

    candidate = self._activation(_x + r * _h)

    new_h = (1 - z) * state + z * candidate

    return new_h, new_h


