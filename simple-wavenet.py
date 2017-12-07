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

    def get_loss_function(self):
        #TODO: calculate predictions (is probably max of softmax)
        loss = tf.losses.mean_squared_error(lables, predictions)

    def calculate_prediction(self, network_output):
        return np.argmax(network_output) #TODO: check the format of the output vector (aka, is it batched?)


    #TODO: This probs should take in more params
    def create_network(self):
        #Initialize?
        skip_connections = []
        #create causal layer
        next_input = self.create_causal_layer(layer_input, filter_width, n_residual_channels)
        #for each dilation layer
        with tf.name_scope('dilation_layers'):
            for i in range(len(self.dilations)): #TODO: check this
                layer_num = i
                dilation = self.dilations[layer_num][i] #todo: layer number
                #create new dilation layer with previous outputs as inputs
                #get list of skips
                skip_connection, next_input = self.create_dilation_layer(next_input)
                skip_connections.append(skip_connection)
        with tf.name_scope('final_steps'):
            #sum skips
            skip_sum = tf.add_n(skip_connections)
            #apply relu
            skip_relu_1 = tf.relu(skip_sum)
            #1x1 convolutional filter
            skip_conv_1 = self.create_1x1_conv_layer(skip_relu_1, 1)
            #apply relu
            skip_relu_2 = tf.relu(skip_conv_1)
            #1x1 convolutional filter
            skip_conv_2 = self.create_1x1_conv_layer(skip_relu_2, 2)
            #softmax
            #TODO: Do I have to do preprocessing ln(1+mu*x)/ln(1+mu)??
            output = tf.softmax(skip_conv_2)
            return output

    def convolutional_layer(self):
        pass

    def create_causal_layer(layer_input, filter_width, n_residual_channels):
        #create scope
        with tf.name_scope('causal_convolution'):
            #determine shape
            shape = [filter_width, 1, n_residual_channels]

            #create layer & layer variable
            init = tf.contrib.layers.xavier_initializer_conv2d()

            #create 2d convolutional layer filter w/initialization for specified shape
            filter_weights = tf.Variable(init(shape=shape), 'causal_layer')

            #pad filter_width-1 to input
            padding = [[0,0] , [filter_width-1, 0],[0,0]]

            new_input = tf.pad(layer_input, padding)

            #do convolution
            return tf.nn.conv1d(new_input, filter_weights, stride=1, padding='VALID')

    def dialated_convolution(layer_input, dialation, n_residual_channels, name):
        #create scope
        with tf.name_scope(name):
            #determine shape
            shape = [dialation, 1, n_residual_channels]

            #create layer & layer variable
            init = tf.contrib.layers.xavier_initializer_conv2d()

            #create 2d convolutional layer filter w/initialization for specified shape
            filter_weights = tf.Variable(init(shape=shape), name)

            #pad dialation to input
            padding = [[0,0] , [dialation, 0],[0,0]]

            new_input = tf.pad(layer_input, padding)

            #do convolution
            #NOTE: I think passing in dilation_rate should be fine, but unsure
            return tf.nn.conv1d(new_input, filter_weights, stride=1, padding='VALID', dilation_rate=dialation)

    def create_dilation_layer(self,
                            layer_input,
                            dilation_size,
                            output_size,
                            input_length,
                            layer_num):
        # Layer input should already be formatted in batch format

        gate_weights = [dilation_size,
                             self.residual_channels,
                             self.dilation_channels]
        filter_weights = [dilation_size,
                             self.residual_channels,
                             self.dilation_channels]

        with tf.variable_scope("dilated_conv"):
            convolutional_filter = self.dilated_conv(layer_input,
                                              dilation_size,
                                              self.residual_channels,
                                              )
            convolutional_gate = self.dilated_conv(layer_input,
                                              dilation_size,
                                              self.residual_channels,
                                              )

            output = tf.tanh(convolutional_filter) + tf.sigmoid(convolutional_gate)

            # Need to split into output for the skip channel and the normal channel

            #TODO: split into skip and residual

            #residual_output = tf.nn.conv1d() #TODO
            residual_output = self.create_1x1_conv_layer(output, 'dilated_conv_residual_' + str(layer_num))
            residual_output = residual_output + layer_input

            # Skipped channel
            skipped_output = self.create_1x1_conv_layer(output, 'dilated_conv_skipped_' + str(layer_num))
            #skipped_output = tf.nn.conv1d() #TODO

            return skipped_output, residual_output

    def create_1x1_conv_layer(self, inputs, name):
        init = tf.contrib.layers.xavier_initializer_conv2d()

        #create 2d convolutional layer filter w/initialization for specified shape
        filter_weights = tf.Variable(init(shape=(1,1)), name)

        #do convolution
        return tf.nn.conv1d(inputs, filter_weights, stride=1, padding='VALID')

    def create_final_layer(self): #TODO ABBEY
        # Linear + Softmax

        #outputs = _output_linear(h)
        ut_ops = [tf.nn.softmax(outputs)]
        pass