import tensorflow as tf
from tensorflow.python.ops.init_ops import UniformUnitScaling, Constant
import cocoapi.PythonAPI.pycocotools.mask as cocoapi

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

def iou_accuracy(box1, box2):
    """
    Args:
        box1: shape (batch, 4) x1, y1, w, h
        box2: shape (batch, 4) x1, y1, w, h

    Reurns:
        Tensor with shape (batch)
        accuracy of the IoU (intersection over union) between box1 and box2 (element wise)
        (If Iou > 0.5 the object is considered as found)
    """

    x11, y11, width1, height1 = tf.split(box1, 4, axis=1)
    x21, y21, width2, height2 = tf.split(box2, 4, axis=1)

    x12 = x11 + width1
    y12 = y11 - height1

    x22 = x21 + width2
    y22 = y21 - height2

    xI1 = tf.maximum(x11, tf.transpose(x21))
    yI1 = tf.maximum(y11, tf.transpose(y21))

    xI2 = tf.minimum(x12, tf.transpose(x22))
    yI2 = tf.minimum(y12, tf.transpose(y22))

    inter_area = (xI2 - xI1 + 1) * (yI2 - yI1 + 1)

    box1_area = (x12 - x11 + 1) * (y12 - y11 + 1)
    box2_area = (x22 - x21 + 1) * (y22 - y21 + 1)

    union = (box1_area + tf.transpose(box2_area)) - inter_area
    all_scores = tf.maximum(inter_area / union, 0)
    bbox_score = tf.diag_part(all_scores)

    # If Iou>0.5 the IoU is considered as found.
    accuracy = tf.where(bbox_score > 0.5, tf.ones_like(bbox_score), tf.zeros_like(bbox_score))
    return accuracy


def main():

    import numpy as np

    with tf.Session() as sess:

        sq1 = [[0, 0, 10, 10], [39, 63, 203, 112]]
        sq2 = [[3, 4, 24, 32], [54, 66, 198, 114]]

        in1 = tf.placeholder(tf.float64, shape=[None, 4])
        in2 = tf.placeholder(tf.float64, shape=[None, 4])

        scores = iou_accuracy(in1,in2)

        print("perso score", sess.run(fetches=[scores], feed_dict={in1:sq1,in2:sq2}))
        print("coco score", cocoapi.iou(sq1,sq2,[]))

if __name__ == "__main__":
    main()