import tensorflow as tf
from neural_toolbox import attention

import tensorflow.contrib.layers as tfc_layers
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

    def __init__(self, states, seq_length, init_cell_state, no_units, shared_attention, feedback_loop=False, sum_memory=False, reuse=False):
        self.memory_cell = init_cell_state
        self.states = states
        self.seq_length = seq_length
        self.no_units = no_units
        self.shared_attention = shared_attention

        self.already_forward = False
        self.scope = "basic_reading_unit"
        self.reuse = reuse

        self.feedback_loop = feedback_loop

    def forward(self, image_feat):

        # Should we reuse attention from one level to another
        reuse = (self.already_forward and self.shared_attention) or self.reuse

        with tf.variable_scope(self.scope, reuse=reuse) as scope:

            self.memory_cell = attention.compute_attention(self.states,
                                                           seq_length=self.seq_length,
                                                           context=self.memory_cell,
                                                           no_mlp_units=self.no_units,
                                                           reuse=reuse)

            self.memory_cell = tfc_layers.layer_norm(self.memory_cell, reuse=self.reuse)

            output = self.memory_cell

            if self.shared_attention:
                self.scope = scope
                self.already_forward = True

        if self.feedback_loop:
            image_feat = tf.reduce_mean(image_feat, axis=[1, 2])
            output = tf.concat([output, image_feat], axis=-1)

        return output


# this factory method create a film layer by calling the reading unit once for every FiLM Resblock
def create_film_layer_with_reading_unit(reading_unit):
    def film_layer_with_reading_unit(ft, context, reuse=False):

        question_emb = reading_unit.forward(ft)

        if len(context) > 0:
            context = tf.concat([question_emb] + context, axis=-1)
        else:
            context = question_emb

        return film_layer(ft, context, reuse)

    return film_layer_with_reading_unit


def create_reading_unit(last_state, states, seq_length, config, reuse):
    unit_type = config["reading_unit_type"]

    if unit_type == "no_question":
        return EmptyReadingUnit(last_state)

    if unit_type == "only_question":
        return NoReadingUnit(last_state)

    elif unit_type == "basic":

        no_units = config["attention_units"]
        shared_attention = config["shared_attention"]
        feedback_loop = config["feedback_loop"]

        return BasicReadingUnit(states=states,
                                seq_length=seq_length,
                                init_cell_state=last_state,
                                no_units=no_units,
                                shared_attention=shared_attention,
                                feedback_loop=feedback_loop,
                                reuse=reuse)

    else:
        assert False, "Invalid reading unit: ".format(unit_type)
