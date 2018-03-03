import tensorflow as tf
import tensorflow.contrib.layers as tfc_layers
import neural_toolbox.ft_utils as ft_utils

import neural_toolbox.film_layer as film
from generic.tf_factory.attention_factory import get_attention


def append_spatial_location(features, config):
    if config["spatial_location"]:
        return ft_utils.append_spatial_location(features)


class FiLM_Stack(object):

    def __init__(self, image, film_input, config, is_training, dropout_keep,
                 film_layer_fct=film.film_layer,
                 append_extra_features=append_spatial_location,
                 attention_input=None,
                 reuse=False):

        #####################
        #   STEM
        #####################

        with tf.variable_scope("stem", reuse=reuse):

            stem_features = image
            stem_features = append_extra_features(stem_features, config["stem"])

            self.stem_conv = tfc_layers.conv2d(stem_features,
                                               num_outputs=config["stem"]["conv_out"],
                                               kernel_size=config["stem"]["conv_kernel"],
                                               normalizer_fn=tfc_layers.batch_norm,
                                               normalizer_params={"center": True, "scale": True,
                                                                  "decay": 0.9,
                                                                  "is_training": is_training,
                                                                  "reuse": reuse},
                                               activation_fn=tf.nn.relu,
                                               reuse=reuse,
                                               scope="stem_conv")

        #####################
        #   FiLM Layers
        #####################

        with tf.variable_scope("resblocks", reuse=reuse):

            res_output = self.stem_conv
            self.resblocks = []

            for i, ft_size in enumerate(config["resblock"]["feature_size"]):
                with tf.variable_scope("ResBlock_{}".format(i), reuse=reuse):

                    res_output = append_extra_features(res_output, config["resblock"])

                    resblock = film.FiLMResblock(res_output, film_input,
                                                 feature_size=ft_size,
                                                 film_layer_fct=film_layer_fct,
                                                 kernel1=config["resblock"]["kernel1"],
                                                 kernel2=config["resblock"]["kernel2"],
                                                 spatial_location=False,
                                                 is_training=is_training,
                                                 reuse=reuse)

                    self.resblocks.append(resblock)
                    res_output = resblock.get()

        #####################
        #   Classifier
        #####################

        if config["head"]["conv_out"] > 0:

            with tf.variable_scope("head", reuse=reuse):

                classif_features = res_output
                classif_features = append_extra_features(classif_features, config["resblock"])

                # 2D-Conv
                self.classif_conv = tfc_layers.conv2d(classif_features,
                                                      num_outputs=config["head"]["conv_out"],
                                                      kernel_size=config["head"]["conv_kernel"],
                                                      normalizer_fn=tfc_layers.batch_norm,
                                                      normalizer_params={"center": True, "scale": True,
                                                                         "decay": 0.9,
                                                                         "is_training": is_training,
                                                                         "reuse": reuse},
                                                      activation_fn=tf.nn.relu,
                                                      reuse=reuse,
                                                      scope="head_conv")
        else:
            self.classif_conv = res_output

        with tf.variable_scope("pooling", reuse=reuse):
            self.pooling = get_attention(self.classif_conv, attention_input,
                                         config=config["head"]["attention"],
                                         dropout_keep=dropout_keep,
                                         is_training=is_training,
                                         reuse=reuse)

    def get(self):
        return self.pooling

