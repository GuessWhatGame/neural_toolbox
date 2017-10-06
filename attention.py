import tensorflow as tf

from neural_toolbox import utils


def compute_attention(feature_maps, context, no_mlp_units):
    with tf.variable_scope("attention"):

        if len(feature_maps.get_shape()) == 3:
            h = tf.shape(feature_maps)[1]  # when the shape is dynamic (attention over lstm)
            w = 1
            c = int(feature_maps.get_shape()[2])
        else:
            h = int(feature_maps.get_shape()[1])
            w = int(feature_maps.get_shape()[2])
            c = int(feature_maps.get_shape()[3])

        s = int(context.get_shape()[1])

        feature_maps = tf.reshape(feature_maps, shape=[-1, h * w, c])

        context = tf.expand_dims(context, axis=1)
        context = tf.tile(context, [1, h * w, 1])

        embedding = tf.concat([feature_maps, context], axis=2)
        embedding = tf.reshape(embedding, shape=[-1, s + c])

        # compute the evidence from the embedding
        with tf.variable_scope("mlp"):
            e = utils.fully_connected(embedding, no_mlp_units, scope='hidden_layer', activation="relu")
            e = utils.fully_connected(e, 1, scope='out')

        e = tf.reshape(e, shape=[-1, h * w, 1])

        # compute the softmax over the evidence
        alpha = tf.nn.softmax(e, dim=1)

        # apply soft attention
        soft_attention = feature_maps * alpha
        soft_attention = tf.reduce_sum(soft_attention, axis=1)

    return soft_attention


# cf https://arxiv.org/abs/1610.04325
def compute_glimpse(feature_maps, context, no_glims, glimse_embedding_size, keep_dropout):
    with tf.variable_scope("glimps"):
        h = int(feature_maps.get_shape()[1])
        w = int(feature_maps.get_shape()[2])
        c = int(feature_maps.get_shape()[3])

        # reshape state to perform batch operation
        context = tf.nn.dropout(context, keep_dropout)
        projected_context = utils.fully_connected(context, glimse_embedding_size,
                                                  scope='hidden_layer', activation="tanh",
                                                  use_bias=False)

        projected_context = tf.expand_dims(projected_context, axis=1)
        projected_context = tf.tile(projected_context, [1, h * w, 1])
        projected_context = tf.reshape(projected_context, [-1, glimse_embedding_size])

        feature_maps = tf.reshape(feature_maps, shape=[-1, h * w, c])

        glimpses = []
        with tf.variable_scope("glimpse"):
            g_feature_maps = tf.reshape(feature_maps, shape=[-1, c])  # linearise the feature map as as single batch
            g_feature_maps = tf.nn.dropout(g_feature_maps, keep_dropout)
            g_feature_maps = utils.fully_connected(g_feature_maps, glimse_embedding_size, scope='picture_projection',
                                                   activation="tanh", use_bias=False)

            hadamard = g_feature_maps * projected_context
            hadamard = tf.nn.dropout(hadamard, keep_dropout)

            e = utils.fully_connected(hadamard, no_glims, scope='hadamard_projection')
            e = tf.reshape(e, shape=[-1, h * w, no_glims])

            for i in range(no_glims):
                ev = e[:, :, i]
                alpha = tf.nn.softmax(ev)
                # apply soft attention
                soft_glimpses = feature_maps * tf.expand_dims(alpha, -1)
                soft_glimpses = tf.reduce_sum(soft_glimpses, axis=1)

                glimpses.append(soft_glimpses)

        full_glimpse = tf.concat(glimpses, axis=1)

    return full_glimpse
