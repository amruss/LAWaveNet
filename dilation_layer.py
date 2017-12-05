import tensorflow as tf
import numpy as np
import math


class SimpleWavenet(object):
    def __init__(self, batch_size, dilation_width, dilation_layer_widths,
                 dilation_channels, residual_channels,):

        # TODO: possibly add possibility of bias term
        # TODO: possibly add skip channels, quantized channels
        # TODO: possibly add possibility of using scalar input
        # TODO: add possibility of global conditioning


        self.batch_size = batch_size
        self.dilations = [[dilation_width**i for i in range(len(n))] for n in dilation_layer_widths]
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels


    def make_causal_conv(self):
        pass

    def convolutional_layer(self):
        pass

    def make_dilation_layer(self,
                            layer_input,
                            dilation_size,
                            output_size,
                            input_length,
                            ):
        # Layer input should already be formatted in batch format

        gate_weights = [dilation_size,
                             self.residual_channels,
                             self.dilation_channels]
        filter_weights = [dilation_size,
                             self.residual_channels,
                             self.dilation_channels]

        with tf.variable_scope("dilated_conv"):
            convolutional_filter = self.convolutional_layer(layer_input,
                                              dilation_size,
                                              self.residual_channels,
                                              )
            convolutional_gate = self.convolutional_layer()

            output = tf.tanh(convolutional_filter) + tf.sigmoid(convolutional_gate)

            # Need to split into output for the skip channel and the normal channel
            residual_output = tf.nn.conv1d() #TODO
            residual_output = residual_output + layer_input

            # Skipped channel
            skipped_output = tf.nn.conv1d() #TODO

            return skipped_output, residual_output
