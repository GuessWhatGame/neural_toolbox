import tensorflow as tf
import tensorflow.contrib.rnn as tfc_rnn

# For some reason, it is faster than MultiCell on tf
def variable_length_LSTM(inp, num_hidden, seq_length,
                         dropout_keep_prob=1.0, scope="lstm", depth=1,
                         layer_norm=False, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        states = []
        last_states = []
        rnn_states = inp
        for d in range(depth):
            with tf.variable_scope('lstmcell'+str(d)):

                cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                    num_hidden,
                    layer_norm=layer_norm,
                    dropout_keep_prob=dropout_keep_prob,
                    reuse=reuse)

                rnn_states, rnn_last_states = tf.nn.dynamic_rnn(
                    cell,
                    rnn_states,
                    dtype=tf.float32,
                    sequence_length=seq_length,
                )
                states.append(rnn_states)
                last_states.append(rnn_last_states.h)

        states = tf.concat(states, axis=2)
        last_states = tf.concat(last_states, axis=1)

        return states, last_states


def create_cell(num_hidden, reuse=False, scope="gru"):

    with tf.variable_scope(scope):
        GRUCell = tfc_rnn.GRUCell
        # if layer_norm:
        #     from neural_toolbox.layer_norm_gru_cell import LNGRUCell
        #     GRUCell = LNGRUCell
        # simple implem -> https://theneuralperspective.com/2016/10/27/gradient-topics/

        rnn_cell = GRUCell(
            num_units=num_hidden,
            activation=tf.nn.tanh,
            reuse=reuse)

    return rnn_cell


def gru_factory(inputs, num_hidden, seq_length,
                bidirectional=False,
                 max_pool=False,
                initial_state_fw=None, initial_state_bw=None,
                reuse=False):

    if bidirectional:

        num_hidden = num_hidden / 2

        rnn_cell_forward = create_cell(num_hidden, reuse=reuse, scope="forward")
        rnn_cell_backward = create_cell(num_hidden, reuse=reuse, scope="backward")

        rnn_states, last_rnn_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=rnn_cell_forward,
            cell_bw=rnn_cell_backward,
            initial_state_fw=initial_state_fw,
            initial_state_bw=initial_state_bw,
            inputs=inputs,
            sequence_length=seq_length,
            dtype=tf.float32)

        # Concat forward/backward
        rnn_states = tf.concat(rnn_states, axis=2)
        last_rnn_state = tf.concat(last_rnn_state, axis=1)

    else:

        rnn_cell_forward = create_cell(num_hidden, reuse=reuse, scope="forward")

        rnn_states, last_rnn_state = tf.nn.dynamic_rnn(
            cell=rnn_cell_forward,
            inputs=inputs,
            dtype=tf.float32,
            sequence_length=seq_length)

    if max_pool:
        last_rnn_state = tf.reduce_max(rnn_states, axis=1)

    return rnn_states, last_rnn_state



