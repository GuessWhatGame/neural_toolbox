import tensorflow as tf
from tensorflow.python.ops.init_ops import UniformUnitScaling, Constant

#TODO slowly delete those modules


def get_embedding(lookup_indices, n_words, n_dim,
                  scope="embedding", reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        with tf.control_dependencies([tf.assert_non_negative(n_words - tf.reduce_max(lookup_indices))]):
            embedding_matrix = tf.get_variable(
                'W', [n_words, n_dim],
                initializer=tf.random_uniform_initializer(-0.08, 0.08))
            embedded = tf.nn.embedding_lookup(embedding_matrix, lookup_indices)
            return embedded


def fully_connected(inp, n_out, activation=None, scope="fully_connected",
                    weight_initializer=UniformUnitScaling(),
                    init_bias=0.0, use_bias=True, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        inp_size = int(inp.get_shape()[1])
        shape = [inp_size, n_out]
        weight = tf.get_variable(
            "W", shape,
            initializer=weight_initializer)
        out = tf.matmul(inp, weight)

        if use_bias:
            bias = tf.get_variable(
                "b", [n_out],
                initializer=Constant(init_bias))
            out += bias

    if activation == 'relu':
        return tf.nn.relu(out)
    if activation == 'softmax':
        return tf.nn.softmax(out)
    if activation == 'tanh':
        return tf.tanh(out)
    return out


def rank(inp):
    return len(inp.get_shape())


def cross_entropy(y_hat, y):
    if rank(y) == 2:
        return -tf.reduce_mean(y * tf.log(y_hat))
    if rank(y) == 1:
        ind = tf.range(tf.shape(y_hat)[0]) * tf.shape(y_hat)[1] + y
        flat_prob = tf.reshape(y_hat, [-1])
        return -tf.log(tf.gather(flat_prob, ind))
    raise ValueError('Rank of target vector must be 1 or 2')


def error(y_hat, y):
    if rank(y) == 1:
        mistakes = tf.not_equal(
            tf.argmax(y_hat, 1), tf.cast(y, tf.int64))
    elif rank(y) == 2:
        mistakes = tf.not_equal(
            tf.argmax(y_hat, 1), tf.argmax(y, 1))
    else:
        assert False
    return tf.cast(mistakes, tf.float32)
