import tensorflow as tf
import tensorflow.contrib.layers as tfc_layers


def fuse_by_concat(input1, input2):
    with tf.variable_scope('concat'):
        return tf.concat([input1, input2], axis=-1)


def fuse_by_dot_product(input1, input2):
    with tf.variable_scope('dot'):
        assert input1.get_shape()[1] == input2.get_shape()[1]
        return input1 * input2


def fuse_by_brut_force(input1, input2):
    with tf.variable_scope('brut_force_fusion'):
        assert tf.size(input1) == tf.size(input2) and tf.size(input1) == 2

        tf.concat([input1, input2, tf.abs(input1 - input2), input1 * input2], axis=-1)


def fuse_by_vis(input1, input2, projection_size, output_size, dropout_keep, activation=tf.nn.tanh, reuse=False):
    with tf.variable_scope('vis_fusion'):

        input1_projection = tfc_layers.fully_connected(input1,
                                                       num_outputs=projection_size,
                                                       activation_fn=activation,
                                                       reuse=reuse,
                                                       scope="input1_projection")

        input2_projection = tfc_layers.fully_connected(input2,
                                                       num_outputs=projection_size,
                                                       activation_fn=activation,
                                                       reuse=reuse,
                                                       scope="input2_projection")

        full_projection = input1_projection * input2_projection
        full_projection = tf.nn.dropout(full_projection, dropout_keep)

        output = tfc_layers.fully_connected(full_projection,
                                            num_outputs=output_size,
                                            activation_fn=activation,
                                            reuse=reuse,
                                            scope="final_projection")

        return output

def fuse_by_vis_left(input1, input2, projection_size, output_size, dropout_keep, activation=tf.nn.tanh, reuse=False):
    with tf.variable_scope('vis_fusion_left'):

        input1_projection = input1

        input2_projection = tfc_layers.fully_connected(input2,
                                                       num_outputs=projection_size,
                                                       activation_fn=activation,
                                                       reuse=reuse,
                                                       scope="input2_projection")

        full_projection = input1_projection * input2_projection
        full_projection = tf.nn.dropout(full_projection, dropout_keep)

        output = tfc_layers.fully_connected(full_projection,
                                            num_outputs=output_size,
                                            activation_fn=activation,
                                            reuse=reuse,
                                            scope="final_projection")

        return output