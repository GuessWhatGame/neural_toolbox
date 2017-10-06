import tensorflow as tf


def variable_length_LSTM(inp, num_hidden, seq_length,
                         dropout_keep_prob=1.0, scope="lstm", depth=1,
                         layer_norm=False, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        states = []
        output = inp
        for d in range(depth):
            with tf.variable_scope('lstmcell'+str(d)):
                cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                    num_hidden,
                    layer_norm=layer_norm,
                    dropout_keep_prob=dropout_keep_prob,
                    reuse=reuse)


                output, _ = tf.nn.dynamic_rnn(
                    cell,
                    output,
                    dtype=tf.float32,
                    sequence_length=seq_length,
                )
                states.append(output)
        last_states = tf.concat([last_relevant(s, seq_length) for s in states], 1)

        return last_states, tf.concat(states, axis=2)


def last_relevant(output, length):
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        output_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)
        return relevant
