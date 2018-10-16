
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


class LayerNormBasicGRUCell(rnn_cell_impl.LayerRNNCell):
  """GRU unit with layer normalization.
    This class adds layer normalization to a
    basic GRU unit. Layer normalization implementation is based on:
      https://arxiv.org/abs/1607.06450.
    "Layer Normalization"
    Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
    and is applied before the internal nonlinearities.
    """

  def __init__(self,
               num_units,
               activation=math_ops.tanh,
               norm_gain=1.0,
               norm_shift=0.0,
               reuse=None):
    """Initializes the cell.
        Args:
          num_units: int, The number of units in the GRU cell.
          activation: Activation function of the inner states.
          norm_gain: float, The layer normalization gain initial value.
          norm_shift: float, The layer normalization shift initial value.
        """

    super(LayerNormBasicGRUCell, self).__init__(_reuse=reuse)

    self._num_units = num_units
    self._activation = activation
    self._g = norm_gain
    self._b = norm_shift
    self._reuse = reuse

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def _norm(self, inp, scope):
    # layer_norm is using gamma and beta variables already initialized in build method
    # this allows to parametrize gamma/beta initializations
    # reuse is therefore set to True
    return layers.layer_norm(inp, reuse=True, scope=scope)

  def build(self, inputs_shape):
    if inputs_shape[1].value is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)

    input_depth = inputs_shape[1].value

    # Initialize beta and gamma for use by layer_norm.
    scopes = ["update_gate",
              "reset_gate",
              "candidate_linear_x",
              "candidate_linear_h"]
    for scope in scopes:
      self.add_variable(scope + "/gamma",
                        shape=[self._num_units],
                        initializer=init_ops.constant_initializer(self._g))
      self.add_variable(scope + "/beta",
                        shape=[self._num_units],
                        initializer=init_ops.constant_initializer(self._b))

    self._update_gate_kernel = self.add_variable(
      "update_gate/kernel",
      shape=[input_depth + self._num_units, self._num_units])
    self._reset_gate_kernel = self.add_variable(
      "reset_gate/kernel",
      shape=[input_depth + self._num_units, self._num_units])
    self._candidate_linear_x_kernel = self.add_variable(
      "candidate_linear_x/kernel",
      shape=[input_depth, self._num_units])
    self._candidate_linear_h_kernel = self.add_variable(
      "candidate_linear_h/kernel",
      shape=[self._num_units, self._num_units])

    self.built = True

  def call(self, inputs, state):
    """GRU cell with layer normalization."""

    args = array_ops.concat([inputs, state], 1)

    z = math_ops.matmul(args, self._update_gate_kernel)
    r = math_ops.matmul(args, self._reset_gate_kernel)

    z = self._norm(z, "update_gate")
    r = self._norm(r, "reset_gate")

    z = math_ops.sigmoid(z)
    r = math_ops.sigmoid(r)

    _x = math_ops.matmul(inputs, self._candidate_linear_x_kernel)
    _h = math_ops.matmul(state, self._candidate_linear_h_kernel)

    _x = self._norm(_x, scope="candidate_linear_x")
    _h = self._norm(_h, scope="candidate_linear_h")

    candidate = self._activation(_x + r * _h)

    new_h = (1 - z) * state + z * candidate

    return new_h, new_h

