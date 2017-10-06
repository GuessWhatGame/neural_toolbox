import tensorflow as tf
import tensorflow.contrib.slim as slim

import neural_toolbox.ft_utils as ft_utils


class FiLMResblock(object):
    def __init__(self, features, context, is_training,
                 kernel1=list([1, 1]),
                 kernel2=list([3, 3]),
                 spatial_mask=True, reuse=None):

        # Append a mask with spatial location to the feature map
        if spatial_mask:
            features = ft_utils.append_spatial_location(features)

        # Retrieve the size of the feature map
        feature_size = int(features.get_shape()[3])

        # First convolution
        self.conv1 = slim.conv2d(features,
                                 num_outputs=feature_size,
                                 kernel_size=kernel1,
                                 stride=1,
                                 activation_fn=tf.nn.relu,
                                 padding='SAME',
                                 scope='conv1',
                                 reuse=reuse)

        # Second convolution
        self.conv2 = slim.conv2d(self.conv1,
                                 num_outputs=feature_size,
                                 kernel_size=kernel2,
                                 stride=1,
                                 activation_fn=None,
                                 padding='SAME',
                                 scope='conv2',
                                 reuse=reuse)

        # Center/reduce output (Batch Normalization with no training parameters)
        self.bn = slim.batch_norm(self.conv2,
                                  center=False,
                                  scale=False,
                                  is_training=is_training,
                                  scope="bn",
                                  reuse=reuse)

        # Apply FILM layer Residual connection
        with tf.variable_scope("FiLM", reuse=reuse):
            self.film = film_layer(features, context, reuse=reuse)

        # Apply ReLU
        self.out = tf.nn.relu(self.film)

        # Residual connection
        self.output = self.out + self.conv1

    def get(self):
        return self.output


def film_layer(features, context, reuse=False):
    """
    A very basic FiLM layer with a linear transformation from context to FiLM parameters
    :param features: features map to modulate. Must be a 3-D input vector (+batch size)
    :param context: conditioned FiLM parameters. Must be a 1-D input vector (+batch size)
    :param reuse: reuse variable, e.g, multi-gpu
    :param scope: tensorflow scope
    :return: modulated features
    """

    height = int(features.get_shape()[1])
    width = int(features.get_shape()[2])
    feature_size = int(features.get_shape()[3])

    film_params = slim.fully_connected(context,
                                       num_outputs=2 * feature_size,
                                       activation_fn=tf.nn.relu,
                                       scope=scope,
                                       reuse=reuse)

    film_params = tf.expand_dims(film_params, axis=[1])
    film_params = tf.expand_dims(film_params, axis=[1])
    film_params = tf.tile(film_params, [1, height, width, 1])

    gammas = film_params[:, :, :, :feature_size]
    betas = film_params[:, :, :, feature_size:]

    output = (1 + gammas) * features + betas

    return output


if __name__ == '__main__':
    feature_maps = tf.placeholder(tf.float32, shape=[None, 8, 7, 256])
    lstm_state = tf.placeholder(tf.float32, shape=[None, 80])

    modulated_feat1 = film_layer(features=feature_maps, context=lstm_state)
    modulated_feat2 = FiLMResblock(features=feature_maps, context=lstm_state, is_training=True).get()
    print(modulated_feat1)
