import tensorflow as tf
from neural_toolbox import attention

import tensorflow.contrib.seq2seq as tfc_seq
import tensorflow.contrib.layers as tfc_layers
from neural_toolbox import rnn

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

    def __init__(self, states, seq_length, init_cell_state, no_units, shared_attention, fuse_mode="dot", keep_dropout=1., feedback_loop=False, reuse=False):
        self.memory_cell = init_cell_state
        self.states = states
        self.seq_length = seq_length
        self.no_units = no_units
        self.shared_attention = shared_attention

        self.already_forward = False
        self.scope = "basic_reading_unit"
        self.reuse = reuse

        self.fuse_mode = fuse_mode
        self.keep_dropout = keep_dropout
        self.feedback_loop = feedback_loop

    def forward(self, image_feat):

        # Should we reuse attention from one level to another
        reuse = (self.already_forward and self.shared_attention) or self.reuse

        with tf.variable_scope(self.scope, reuse=reuse) as scope:
            self.memory_cell = attention.compute_attention(self.states,
                                                           seq_length=self.seq_length,
                                                           context=self.memory_cell,
                                                           no_mlp_units=self.no_units,
                                                           fuse_mode=self.fuse_mode,
                                                           reuse=reuse)
            output = self.memory_cell

            if self.shared_attention:
                self.scope = scope
                self.already_forward = True

        if self.feedback_loop:
            image_feat = tf.reduce_mean(image_feat, axis=[1, 2])
            output = tf.concat([output, image_feat], axis=-1)

        output = tf.layers.dropout(output, self.keep_dropout)

        return output


class BasicReadingUnit2(object):

    def __init__(self, states, seq_length, init_cell_state, no_units, shared_attention, fuse_mode="dot", gating="relu", feedback_loop=False, keep_dropout=1., reuse=False):
        self.memory_cell = init_cell_state
        self.states = states
        self.seq_length = seq_length
        self.no_units = no_units
        self.shared_attention = shared_attention
        self.gating = gating

        self.already_forward = False
        self.scope = "basic_reading_unit"
        self.fuse_mode = fuse_mode
        self.keep_dropout = keep_dropout
        self.feedback_loop = feedback_loop

        self.reuse = reuse  # This reuse is used by multi-gpu computation, this does not encode weight sharing inside memory cells

    def forward(self, image_feat):

        if self.feedback_loop:
            with tf.variable_scope("feedback_loop", reuse=self.reuse):
                image_feat = tf.reduce_mean(image_feat, axis=[1, 2])

                new_memory = tf.concat([self.memory_cell, image_feat], axis=-1)
                new_memory = tfc_layers.fully_connected(new_memory,
                                                        num_outputs=int(self.memory_cell.get_shape()[1]),
                                                        activation_fn=eval("tf.nn.{}".format(self.gating)),
                                                        scope='hidden_layer',
                                                        reuse=self.reuse)  # reuse: multi-gpu computation

                self.memory_cell = tf.layers.dropout(new_memory, self.keep_dropout)

        # Should we reuse attention from one level to another
        reuse = (self.already_forward and self.shared_attention) or self.reuse

        with tf.variable_scope(self.scope, reuse=reuse) as scope:
            self.memory_cell = attention.compute_attention(self.states,
                                                           seq_length=self.seq_length,
                                                           context=self.memory_cell,
                                                           no_mlp_units=self.no_units,
                                                           fuse_mode=self.fuse_mode,
                                                           reuse=reuse)
            if self.shared_attention:
                self.scope = scope
                self.already_forward = True

            output = self.memory_cell

        self.already_forward = True

        output = tf.layers.dropout(output, self.keep_dropout)

        return output



class BasicReadingUnit3(object):

    def __init__(self, states, seq_length, init_cell_state, no_units, shared_attention, fuse_mode="dot", feedback_loop=False, gating="relu", keep_dropout=1., reuse=False):
        self.memory_cell = init_cell_state
        self.states = states
        self.seq_length = seq_length
        self.no_units = no_units
        self.shared_attention = shared_attention
        self.gating = gating

        self.already_forward = False
        self.scope = "basic_reading_unit"
        self.fuse_mode = fuse_mode
        self.keep_dropout = keep_dropout
        self.feedback_loop = feedback_loop

        self.reuse = reuse  # This reuse is used by multi-gpu computation, this does not encode weight sharing inside memory cells

    def forward(self, image_feat):

        if self.feedback_loop:
            with tf.variable_scope("feedback_loop", reuse=self.reuse):
                image_feat = tf.reduce_mean(image_feat, axis=[1, 2])

                image_feat = tfc_layers.fully_connected(image_feat,
                                                        num_outputs=int(self.memory_cell.get_shape()[1]),
                                                        activation_fn=eval("tf.nn.{}".format(self.gating)),
                                                        scope='hidden_layer',
                                                        reuse=self.reuse)  # reuse: multi-gpu computation

                self.memory_cell = self.memory_cell * image_feat

        # Should we reuse attention from one level to another
        reuse = (self.already_forward and self.shared_attention) or self.reuse

        with tf.variable_scope(self.scope, reuse=reuse) as scope:
            self.memory_cell = attention.compute_attention(self.states,
                                                           seq_length=self.seq_length,
                                                           context=self.memory_cell,
                                                           no_mlp_units=self.no_units,
                                                           fuse_mode=self.fuse_mode,
                                                           reuse=reuse)
            if self.shared_attention:
                self.scope = scope
                self.already_forward = True

            output = self.memory_cell

        self.already_forward = True

        output = tf.layers.dropout(output, self.keep_dropout)

        return output


class NoMemoryReadingUnit(object):

    def __init__(self, states, seq_length, init_cell_state, no_units, shared_attention, fuse_mode="dot", feedback_loop=False, gating="relu", keep_dropout=1., reuse=False):
        self.memory_cell = init_cell_state
        self.states = states
        self.seq_length = seq_length
        self.no_units = no_units
        self.shared_attention = shared_attention
        self.gating = gating

        self.already_forward = False
        self.scope = "no_memory_reading_unit"
        self.fuse_mode = fuse_mode
        self.keep_dropout = keep_dropout
        self.feedback_loop = feedback_loop

        self.reuse = reuse  # This reuse is used by multi-gpu computation, this does not encode weight sharing inside memory cells

    def forward(self, image_feat):

        if self.feedback_loop:
            with tf.variable_scope("feedback_loop", reuse=self.reuse):
                image_feat = tf.reduce_mean(image_feat, axis=[1, 2])

                image_feat = tfc_layers.fully_connected(image_feat,
                                                        num_outputs=int(self.memory_cell.get_shape()[1]),
                                                        activation_fn=eval("tf.nn.{}".format(self.gating)),
                                                        scope='hidden_layer',
                                                        reuse=self.reuse)  # reuse: multi-gpu computation

        # Should we reuse attention from one level to another
        reuse = (self.already_forward and self.shared_attention) or self.reuse

        with tf.variable_scope(self.scope, reuse=reuse) as scope:
            output = attention.compute_attention(self.states,
                                                           seq_length=self.seq_length,
                                                           context=image_feat,
                                                           no_mlp_units=self.no_units,
                                                           fuse_mode=self.fuse_mode,
                                                           reuse=reuse)
            if self.shared_attention:
                self.scope = scope
                self.already_forward = True

        self.already_forward = True

        output = tf.layers.dropout(output, self.keep_dropout)

        return output




# https://openreview.net/pdf?id=S1Euwz-Rb
class CLEVRReadingUnit(object):

    def __init__(self, states, seq_length, last_state, init_cell_state, no_units, shared_attention):
        self.context_state = init_cell_state
        self.states = states
        self.seq_length = seq_length
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
                activation_fn=None)

            proj_context_state = tfc_layers.fully_connected(
                tf.concat([projected_state, self.context_state], axis=-1),
                num_outputs=int(self.context_state.get_shape()[-1]),
                activation_fn=None)

            self.context_state = attention.compute_attention(self.states,
                                                             seq_length=self.seq_length,
                                                             context=proj_context_state,
                                                             no_mlp_units=self.no_units,
                                                             fuse_mode="dot",
                                                             reuse=reuse)

        self.already_forward = True

        return self.context_state


class RecurrentReadingUnit(object):
    def __init__(self, last_state, keep_dropout, img_prj_units, reuse):
        self.rnn_cell = rnn.create_cell(
            num_units=int(last_state.get_shape()[-1]),
            layer_norm=False,  # TODO use layer norm if it works!
            reuse=reuse)

        self.state = last_state
        self.keep_dropout = keep_dropout
        self.img_prj_units = img_prj_units

        self.reuse = reuse
        self.scope = "rnn_reading_unit"

    def forward(self, ft):
        image_feat = tf.reduce_mean(ft, axis=[1, 2])

        if self.img_prj_units > 0:
            with tf.variable_scope("feedback_loop", reuse=self.reuse):
                image_feat = tfc_layers.fully_connected(image_feat, num_outputs=self.img_prj_units, activation_fn=tf.nn.relu)
                image_feat = tf.layers.dropout(image_feat, self.keep_dropout)

        with tf.variable_scope(self.scope):
            self.state, _ = self.rnn_cell(inputs=image_feat, state=self.state)
            # TODO add layer_norm?

            output = tf.layers.dropout(self.state, self.keep_dropout)
            return output


class RecurrentAttReadingUnit(object):
    def __init__(self, states, seq_length, keep_dropout, img_prj_units, reuse):
        self.input = states
        self.rnn_cell = rnn.create_cell(
            num_units=int(states.get_shape()[-1]),
            layer_norm=False,  # TODO use layer norm if it works!
            reuse=reuse)

        self.scope = "attrnn_reading_unit"

        self.img_prj_units = img_prj_units
        self.keep_dropout = keep_dropout
        self.reuse = reuse

        with tf.variable_scope(self.scope, reuse=reuse):
            self.attention_mechanism = tfc_seq.LuongAttention(
                num_units=int(states.get_shape()[-1]),
                memory=states,
                memory_sequence_length=seq_length)

            # TODO missing dropout
            self.decoder_cell = tfc_seq.AttentionWrapper(
                self.rnn_cell,
                self.attention_mechanism,
                output_attention=False)

            self.cell_state = self.decoder_cell.zero_state(batch_size=tf.shape(states)[0], dtype=tf.float32)

    def forward(self, ft):
        image_feat = tf.reduce_mean(ft, axis=[1, 2])

        if self.img_prj_units > 0:
            with tf.variable_scope("feedback_loop", reuse=self.reuse):
                image_feat = tfc_layers.fully_connected(image_feat, num_outputs=self.img_prj_units, activation_fn=tf.nn.relu)
                image_feat = tf.layers.dropout(image_feat, self.keep_dropout)

        with tf.variable_scope(self.scope, reuse=self.reuse):
            output, self.cell_state = self.decoder_cell(image_feat, self.cell_state)
            output = tf.layers.dropout(output, self.keep_dropout)

        return output


# this factory method create a film layer by calling the reading unit one for every FiLM Resblock
def create_film_layer_with_reading_unit(reading_unit):
    def film_layer_with_reading_unit(ft, context, reuse=False):

        question_emb = reading_unit.forward(ft)

        if len(context) > 0:
            context = tf.concat([question_emb] + context, axis=-1)
        else:
            context = question_emb

        return film_layer(ft, context, reuse)

    return film_layer_with_reading_unit


def create_reading_unit(last_state, states, seq_length, config, keep_dropout, reuse):
    unit_type = config["reading_unit_type"]

    if unit_type == "no_question":
        return EmptyReadingUnit(last_state)

    if unit_type == "only_question":
        return NoReadingUnit(last_state)

    elif unit_type == "basic":

        no_units = config["attention_units"]
        shared_attention = config["shared_attention"]
        feedback_loop = config["feedback_loop"]
        fuse_mode = config["fuse_mode"]

        return BasicReadingUnit(states=states,
                                seq_length=seq_length,
                                init_cell_state=last_state,
                                no_units=no_units,
                                shared_attention=shared_attention,
                                fuse_mode=fuse_mode,
                                keep_dropout=keep_dropout,
                                feedback_loop=feedback_loop,
                                reuse=reuse)

    elif unit_type == "basic2":

        no_units = config["attention_units"]
        shared_attention = config["shared_attention"]
        feedback_loop = config["feedback_loop"]
        fuse_mode = config["fuse_mode"]
        gating = config["gating"]

        return BasicReadingUnit2(states=states,
                                 seq_length=seq_length,
                                 init_cell_state=last_state,
                                 no_units=no_units,
                                 shared_attention=shared_attention,
                                 fuse_mode=fuse_mode,
                                 gating=gating,
                                 keep_dropout=keep_dropout,
                                 feedback_loop=feedback_loop,
                                 reuse=reuse)

    elif unit_type == "basic3":

        no_units = config["attention_units"]
        shared_attention = config["shared_attention"]
        feedback_loop = config["feedback_loop"]
        fuse_mode = config["fuse_mode"]
        gating = config["gating"]


        return BasicReadingUnit3(states=states,
                                 seq_length=seq_length,
                                 init_cell_state=last_state,
                                 no_units=no_units,
                                 shared_attention=shared_attention,
                                 fuse_mode=fuse_mode,
                                 gating=gating,
                                 keep_dropout=keep_dropout,
                                 feedback_loop=feedback_loop,
                                 reuse=reuse)

    elif unit_type == "no_mem":

        no_units = config["attention_units"]
        shared_attention = config["shared_attention"]
        feedback_loop = config["feedback_loop"]
        fuse_mode = config["fuse_mode"]
        gating = config["gating"]

        return NoMemoryReadingUnit(states=states,
                                 seq_length=seq_length,
                                 init_cell_state=last_state,
                                 no_units=no_units,
                                 shared_attention=shared_attention,
                                 fuse_mode=fuse_mode,
                                 gating=gating,
                                 keep_dropout=keep_dropout,
                                 feedback_loop=feedback_loop,
                                 reuse=reuse)

    elif unit_type == "clevr":

        no_units = config["attention_units"]
        shared_attention = config["shared_attention"]

        return CLEVRReadingUnit(states=states,
                                seq_length=seq_length,
                                last_state=last_state,
                                init_cell_state=last_state,
                                no_units=no_units,
                                shared_attention=shared_attention)

    elif unit_type == "rnn":

        no_units = config["attention_units"]
        return RecurrentReadingUnit(last_state=last_state, img_prj_units=no_units, keep_dropout=keep_dropout, reuse=reuse)

    elif unit_type == "att_rnn":

        no_units = config["attention_units"]
        return RecurrentAttReadingUnit(states=states,
                                       img_prj_units=no_units,
                                       seq_length=seq_length,
                                       keep_dropout=keep_dropout,
                                       reuse=reuse)
    else:
        assert False, "Invalid reading unit: ".format(unit_type)


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
        layer_norm=True,
        reuse=False)

    # reading_unit = BasicReadingUnit(states=rnn_states,
    #                                 init_cell_state=last_rnn_state,
    #                                 no_units=128, shared_attention=False)
