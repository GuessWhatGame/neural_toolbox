import tensorflow as tf
from tensorflow.python.ops.init_ops import UniformUnitScaling, Constant

#TODO slowly delete those modules


from tensorflow.python.layers import base

class MultiLayers(base.Layer):
   """ Multi Layer Class - Applies multiple layers sequentially """

   def __init__(self, layers, **kwargs):
       """ Constructor
           :param layers: A list of layers to apply
           :param kwargs: Optional. Keyword arguments
       """
       super(MultiLayers, self).__init__(**kwargs)
       self.layers = layers

   def call(self, inputs, **kwargs):
       """ Sequentially calls each layer with the output of the previous one """
       outputs = inputs
       for layer in self.layers:
           outputs = layer(outputs)
       return outputs

   def _compute_output_shape(self, input_shape):
       """ Computes the output shape given the input shape """
       output_shape = input_shape
       for layer in self.layers:
           output_shape = layer._compute_output_shape(output_shape)
       return output_shape



def masked_softmax(scores, mask):

    # subtract max for stability
    scores = scores - tf.tile(tf.reduce_max(scores, axis=(1,), keep_dims=True), [1, tf.shape(scores)[1]])

    # compute padded softmax
    exp_scores = tf.exp(scores)
    exp_scores *= mask
    exp_sum_scores = tf.reduce_sum(exp_scores, axis=1, keep_dims=True)
    return exp_scores / tf.tile(exp_sum_scores, [1, tf.shape(exp_scores)[1]])