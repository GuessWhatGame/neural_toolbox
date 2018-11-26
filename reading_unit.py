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

    def __init__(self, states, seq_length, init_cell_state,
                 attention_hidden_units=0,
                 shared_attention=True,
                 sum_memory=False,
                 inject_img_before=False,
                 inject_img_before2=False,
                 inject_img_after=False,
                 reuse=False):

        self.memory_cell = init_cell_state
        self.states = states
        self.seq_length = seq_length
        self.attention_hidden_units = attention_hidden_units
        self.shared_attention = shared_attention

        self.sum_memory = sum_memory
        self.already_forward = False
        self.scope = "basic_reading_unit"

        self.inject_img_before = inject_img_before
        self.inject_img_before2 = inject_img_before2
        self.inject_img_after = inject_img_after

        self.reuse = reuse  # This reuse is used by multi-gpu computation, this does not encode weight sharing inside memory cells

    def forward(self, image_feat):

        if self.inject_img_before:
            with tf.variable_scope("feedback_loop", reuse=self.reuse):
                image_feat = tf.reduce_mean(image_feat, axis=[1, 2])

                new_memory = tf.concat([self.memory_cell, image_feat], axis=-1)
                new_memory = tfc_layers.fully_connected(new_memory,
                                                        num_outputs=int(self.memory_cell.get_shape()[1]),
                                                        scope='hidden_layer',
                                                        reuse=self.reuse)  # reuse: multi-gpu computation

                self.memory_cell = tfc_layers.layer_norm(new_memory, reuse=self.reuse)

        if self.inject_img_before2:
            with tf.variable_scope("feedback_loop", reuse=self.reuse):
                image_feat = tf.reduce_mean(image_feat, axis=[1, 2])

                image_emb = tfc_layers.fully_connected(image_feat,
                                                       num_outputs=int(self.memory_cell.get_shape()[1]),
                                                       scope='hidden_layer',
                                                       reuse=self.reuse)  # reuse: multi-gpu computation
                image_emb = tf.nn.relu(image_emb)

                self.memory_cell += image_emb

        # Should we reuse attention from one level to another
        reuse = (self.already_forward and self.shared_attention) or self.reuse

        with tf.variable_scope(self.scope, reuse=reuse) as scope:
            new_memory_cell = attention.compute_attention(self.states,
                                                          seq_length=self.seq_length,
                                                          context=self.memory_cell,
                                                          no_mlp_units=self.attention_hidden_units,
                                                          fuse_mode="dot",
                                                          reuse=reuse)

        if self.sum_memory:
            self.memory_cell = self.memory_cell + new_memory_cell
        else:
            self.memory_cell = new_memory_cell

        self.memory_cell = tfc_layers.layer_norm(new_memory_cell, reuse=self.reuse)

        output = self.memory_cell

        if self.shared_attention:
            self.scope = scope
            self.already_forward = True

        if self.inject_img_after:
            image_feat = tf.reduce_mean(image_feat, axis=[1, 2])
            output = tf.concat([output, image_feat], axis=-1)

        return output


# this factory method create a film layer by calling the reading unit once for every FiLM Resblock
def create_film_layer_with_reading_unit(reading_unit):

    def film_layer_with_reading_unit(ft, context, reuse=False):
        question_emb = reading_unit.forward(ft)
        film_embedding = tf.concat([question_emb] + context, axis=-1)
        return film_layer(ft, film_embedding, reuse)

    return film_layer_with_reading_unit


def create_reading_unit(last_state, states, seq_length, config, reuse):
    unit_type = config["reading_unit_type"]

    if unit_type == "no_question":
        return EmptyReadingUnit(last_state)

    if unit_type in ["only_question", "none"]:
        return NoReadingUnit(last_state)

    elif unit_type == "basic":

        return BasicReadingUnit(states=states,
                                seq_length=seq_length,
                                init_cell_state=last_state,
                                attention_hidden_units=config["attention_hidden_units"],
                                shared_attention=config["shared_attention"],
                                inject_img_before=config["inject_img_before"],
                                inject_img_before2=config["inject_img_before2"],
                                inject_img_after=config["inject_img_after"],
                                sum_memory=config["sum_memory"],
                                reuse=reuse)

    else:
        assert False, "Invalid reading unit: ".format(unit_type)
