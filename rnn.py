import tensorflow as tf
import tensorflow.contrib.rnn as tfc_rnn


def create_cell(num_units, reuse=False, layer_norm=False, cell="gru", scope="rnn"):

    with tf.variable_scope(scope):

        if cell == "gru":

            if layer_norm:

                 from neural_toolbox.LayerNormBasicGRUCell import LayerNormBasicGRUCell

                 rnn_cell = LayerNormBasicGRUCell(
                     num_units=num_units,
                     layer_norm=layer_norm,
                     activation=tf.nn.tanh,
                     reuse=reuse)

            else:
                rnn_cell = tfc_rnn.GRUCell(
                    num_units=num_units,
                    activation=tf.nn.tanh,
                    reuse=reuse)

        elif cell == "lstm":
            rnn_cell = tfc_rnn.LayerNormBasicLSTMCell(
                num_units=num_units,
                layer_norm=layer_norm,
                activation=tf.nn.tanh,
                reuse=reuse)

        else:
            assert False, "Invalid RNN cell"

    return rnn_cell


def rnn_factory(inputs, num_hidden, seq_length,
                cell="gru",
                bidirectional=False,
                max_pool=False,
                layer_norm=False,
                initial_state_fw=None, initial_state_bw=None,
                reuse=False):

    if bidirectional:

        num_hidden = num_hidden / 2

        rnn_cell_forward = create_cell(num_hidden, layer_norm=layer_norm, reuse=reuse, cell=cell, scope="forward")
        rnn_cell_backward = create_cell(num_hidden, layer_norm=layer_norm, reuse=reuse, cell=cell,  scope="backward")

        rnn_states, last_rnn_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=rnn_cell_forward,
            cell_bw=rnn_cell_backward,
            initial_state_fw=initial_state_fw,
            initial_state_bw=initial_state_bw,
            inputs=inputs,
            sequence_length=seq_length,
            dtype=tf.float32)

        if cell == "lstm":
            last_rnn_state = tuple(last_state.h for last_state in last_rnn_state)

        # Concat forward/backward
        rnn_states = tf.concat(rnn_states, axis=2)
        last_rnn_state = tf.concat(last_rnn_state, axis=1)

    else:

        rnn_cell_forward = create_cell(num_hidden, layer_norm=layer_norm, reuse=reuse, cell=cell,  scope="forward")

        rnn_states, last_rnn_state = tf.nn.dynamic_rnn(
            cell=rnn_cell_forward,
            inputs=inputs,
            dtype=tf.float32,
            sequence_length=seq_length)

        if cell == "lstm":
            last_rnn_state = last_rnn_state.h

    if max_pool:
        last_rnn_state = tf.reduce_max(rnn_states, axis=1)

    return rnn_states, last_rnn_state




