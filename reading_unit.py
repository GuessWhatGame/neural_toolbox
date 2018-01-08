import tensorflow as tf
from neural_toolbox import attention
import tensorflow.contrib.seq2seq  as tfc_seq
import tensorflow.contrib.rnn as tfc_rnn

class DummyReadingUnit(object):
    def __init__(self, last_state):
        self.last_state = last_state

        self.scope = "dummy_reading_unit"

    def forward(self, **_):
        return self.last_state


class BasicReadingUnit(object):

    def __init__(self, states, init_cell_state, no_units, shared_attention):
        self.context_cell = init_cell_state
        self.states = states
        self.no_units = no_units
        self.shared_attention = shared_attention

        self.already_forward = False
        self.scope = "basic_reading_unit"

    def forward(self, **_):

        # Should we reuse attention from one level to another
        reuse = (self.already_forward and self.shared_attention)

        with tf.variable_scope(self.scope, reuse=reuse):
            self.context_cell = attention.compute_attention(self.states,
                                                            context=self.context_cell,
                                                            no_mlp_units=self.no_units,
                                                            fuse_mode="dot",
                                                            reuse=reuse)

        self.already_forward = True

        return self.context_cell


# https://openreview.net/pdf?id=S1Euwz-Rb
class CLEVRReadingUnit(object):

    def __init__(self, states, last_state, init_cell_state, no_units, shared_attention):
        self.context_state = init_cell_state
        self.states = states
        self.last_state = last_state
        self.no_units = no_units
        self.shared_attention = shared_attention

        self.already_forward = False
        self.scope = "clevr_reading_unit"

    def forward(self, **_):

        # Should we reuse attention from one level to another
        reuse = (self.already_forward and self.shared_attention)



        with tf.variable_scope(self.scope, reuse=reuse):

            projected_state = tfc_layers.fully_connected(
                self.context_state,
                num_outputs=int(self.context_state.get_shape()[-1]),
                activation=None)

            proj_context_state = tfc_layers.fully_connected(
               tf.concat([projected_state, self.context_state], axis=-1),
               num_outputs=int(self.context_state.get_shape()[-1]),
               activation=None)


            self.context_state = attention.compute_attention(self.states,
                                                             context=proj_context_state,
                                                             no_mlp_units=self.no_units,
                                                             fuse_mode="dot",
                                                             reuse=reuse)

        self.already_forward = True

        return self.context_state


class RecurrentReadingUnit(object):
    def __init__(self, states, rnn_cell):
        self.rnn_cell = rnn_cell
        self.states = states

        self.scope = "rnn_reading_unit"

    def forward(self, **_):
        with tf.variable_scope(self.scope):
            return self.rnn_cell(self.states)


class RecurrentAttReadingUnit(object):
    def __init__(self, states, seq_length, rnn_cell, no_units, attention_mechanism_cst=tfc_seq.LuongAttention):

        self.input = states
        self.rnn_cell = rnn_cell

        self.scope = "attrnn_reading_unit"

        with tf.variable_scope(self.scope):
            self.attention_states = tf.transpose(states, [1, 0, 2])
            self.attention_mechanism = attention_mechanism_cst(
                no_units=no_units,
                memory=self.attention_states,
                memory_sequence_length=seq_length)

            self.decoder_cell = tfc_seq.DynamicAttentionWrapper(
                rnn_cell,
                self.attention_mechanism,
                attention_size=24,
                output_attention=False)

            self.cell_state = self.decoder_cell.zero_state(batch_size=seq_length.get_shape(), dtype=tf.float32)

    def forward(self, input=None):

        with tf.variable_scope(self.scope):

            if input is not None:
                self.input = tf.concat([self.input, input], axis=-1)

            output, self.cell_state = self.decoder_cell(self.input, self.cell_state)
            self.input = output

        return output




def create_reading_unit(last_state, states, seq_length, config):

    unit_type = config["reading_unit"]

    if unit_type == "none":
        return DummyReadingUnit(last_state)
    elif unit_type == "basic":

        no_units = config["att_units"]
        shared_attention = config["shared_attention"]

        return BasicReadingUnit(states=states, init_cell_state=last_state,
                                no_units=no_units,
                                shared_attention=shared_attention)

    elif unit_type == "basic":

        no_units = config["reading_unit"]
        shared_attention = config["shared_attention"]

        return BasicReadingUnit(states=states, init_cell_state=last_state,
                                no_units=no_units,
                                shared_attention=shared_attention)

    elif unit_type == "rnn":

        rnn_cell = tfc_rnn.GRUCell(
            num_units=config["rnn_size"],
            activation=tf.nn.tanh)

        return RecurrentReadingUnit(states=states, rnn_cell=rnn_cell)

    elif unit_type == "att_rnn":

        no_units = config["att_units"]
        rnn_cell = tfc_rnn.GRUCell(
            num_units=config["rnn_size"],
            activation=tf.nn.tanh)

        return RecurrentAttReadingUnit(states=states, seq_length=seq_length,
                                       rnn_cell=rnn_cell,
                                       no_units=no_units)


if __name__ == "__main__":

    import neural_toolbox.rnn as rnn
    import tensorflow.contrib.layers as tfc_layers

    _question = tf.placeholder(tf.int32, [None, None], name='question')
    _seq_length = tf.placeholder(tf.int32, [None], name='seq_length')

    no_words = 100

    word_emb = tfc_layers.embed_sequence(
        ids=_question,
        vocab_size=no_words,
        embed_dim=64,
        scope="word_embedding")


    rnn_states, last_rnn_state = rnn.gru_factory(
        inputs=word_emb,
        seq_length=_seq_length,
        num_hidden=1024,
        bidirectional=False,
        max_pool=False,
        reuse=False)

    # reading_unit = BasicReadingUnit(states=rnn_states,
    #                                 init_cell_state=last_rnn_state,
    #                                 no_units=128, shared_attention=False)



    out = reading_unit.forward()

