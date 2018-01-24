import tensorflow as tf
from neural_toolbox import attention
import tensorflow.contrib.seq2seq as tfc_seq
import tensorflow.contrib.rnn as tfc_rnn
from neural_toolbox.film_layer import film_layer


class EmptyReadingUnit(object):
    def __init__(self, last_state):
        self.last_state = last_state

        self.scope = "empty"

    # Create a dummy 0-vector (ugly but works!)
    def forward(self, _):
        zero_state = tf.zeros([1, 1])  #
        zero_state = tf.tile(zero_state, [tf.shape(self.last_state)[0], 1])  # trick to do a dynamic size 0 tensors

        return zero_state


class NoReadingUnit(object):
    def __init__(self, last_state):
        self.last_state = last_state

        self.scope = "no_reading_unit"

    def forward(self, _):
        return self.last_state



class BasicReadingUnit(object):

    def __init__(self, states, init_cell_state, no_units, shared_attention, fuse_mode="dot", keep_dropout=1., feedback_loop=False):
        self.memory_cell = init_cell_state
        self.states = states
        self.no_units = no_units
        self.shared_attention = shared_attention

        self.already_forward = False
        self.scope = "basic_reading_unit"
        self.fuse_mode = fuse_mode
        self.keep_dropout = keep_dropout
        self.feedback_loop = feedback_loop

    def forward(self, image_feat):

        # Should we reuse attention from one level to another
        reuse = (self.already_forward and self.shared_attention)

        with tf.variable_scope(self.scope, reuse=reuse):
            self.memory_cell = attention.compute_attention(self.states,
                                                           context=self.memory_cell,
                                                           no_mlp_units=self.no_units,
                                                           fuse_mode=self.fuse_mode,
                                                           reuse=reuse)
            output = self.memory_cell

        self.already_forward = True

        if self.feedback_loop:
            image_feat = tf.reduce_mean(image_feat, axis=[1, 2])
            output = tf.concat([output, image_feat], axis=-1)

        output = tf.layers.dropout(output, self.keep_dropout)

        return output


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

    def forward(self, _):

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
    def __init__(self, states, rnn_cell, keep_dropout):
        self.rnn_cell = rnn_cell
        self.states = states
        self.keep_dropout = keep_dropout

        self.scope = "rnn_reading_unit"

    def forward(self):
        with tf.variable_scope(self.scope):
            output = self.rnn_cell(self.states)
            output = tf.layers.dropout(output, self.keep_dropout)
            return output


# class RecurrentAttReadingUnit(object):
#     def __init__(self, states, seq_length, rnn_cell, no_units, attention_mechanism_cst=tfc_seq.LuongAttention):
#
#         self.input = states
#         self.rnn_cell = rnn_cell
#
#         self.scope = "attrnn_reading_unit"
#
#         with tf.variable_scope(self.scope):
#             self.attention_states = tf.transpose(states, [1, 0, 2])
#             self.attention_mechanism = attention_mechanism_cst(
#                 num_units=no_units,
#                 memory=self.attention_states,
#                 memory_sequence_length=seq_length)
#
#             self.decoder_cell = tfc_seq.AttentionWrapper(
#                 rnn_cell,
#                 self.attention_mechanism,
#                 attention_layer_size=config,
#                 output_attention=False)
#
#             self.cell_state = self.decoder_cell.zero_state(batch_size=seq_length.get_shape(), dtype=tf.float32)
#
#     def forward(self, input=None):
#
#         with tf.variable_scope(self.scope):
#
#             if input is not None:
#                 self.input = tf.concat([self.input, input], axis=-1)
#
#             output, self.cell_state = self.decoder_cell(self.input, self.cell_state)
#             self.input = output
#
#         return output

# this factory method create a film layer by calling the reading unit one for every FiLM Resblock
def create_film_layer_with_reading_unit(reading_unit):

    def film_layer_with_reading_unit(ft, context, reuse=False):

        # retrieve
        question_emb = reading_unit.forward(ft)

        if len(context) > 0:
            context = tf.concat([question_emb, context], axis=-1)
        else:
            context = question_emb

        return film_layer(ft, context, reuse)

    return film_layer_with_reading_unit


def create_reading_unit(last_state, states, config, keep_dropout):

    unit_type = config["reading_unit_type"]

    if unit_type == "no_question":
        return EmptyReadingUnit(last_state)

    if unit_type == "only_question":
        return NoReadingUnit(last_state)

    elif unit_type == "basic":

        no_units = config["attention_units"]
        shared_attention = config["shared_attention"]
        feedback_loop = config["feedback_loop"]

        return BasicReadingUnit(states=states, init_cell_state=last_state,
                                no_units=no_units,
                                shared_attention=shared_attention,
                                keep_dropout=keep_dropout,
                                feedback_loop=feedback_loop)

    elif unit_type == "clevr":

        no_units = config["attention_units"]
        shared_attention = config["shared_attention"]

        return CLEVRReadingUnit(states=states,
                                last_state=last_state,
                                init_cell_state=last_state,
                                no_units=no_units,
                                shared_attention=shared_attention)

    elif unit_type == "rnn":

        rnn_cell = tfc_rnn.GRUCell(
            num_units=config["rnn_size"],
            activation=tf.nn.tanh)

        return RecurrentReadingUnit(states=states, rnn_cell=rnn_cell, keep_dropout=keep_dropout)

    # elif unit_type == "att_rnn":
    #
    #     no_units = config["att_units"]
    #     rnn_cell = tfc_rnn.GRUCell(
    #         num_units=config["rnn_size"],
    #         activation=tf.nn.tanh)
    #
    #     return RecurrentAttReadingUnit(states=states, seq_length=seq_length,
    #                                    rnn_cell=rnn_cell,
    #                                    no_units=no_units)


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



