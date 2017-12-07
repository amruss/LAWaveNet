import tensorflow as tf
import numpy as np
import math


#########Variables################
# dilation_channels: How many filters to learn for the dilated convolution.
#################################


class SimpleWavenet(object):
    def __init__(self, batch_size, dilation_width,
                 dilation_layer_widths,
                 dilation_channels=32,
                 residual_channels=16,
                 skip_channels=16
                 ):

        # TODO: possibly add possibility of bias term
        # TODO: possibly add skip channels, quantized channels
        # TODO: possibly add possibility of using scalar input
        # TODO: add possibility of global conditioning


        self.batch_size = batch_size
        self.dilations = [[dilation_width**i for i in range(len(n))] for n in dilation_layer_widths]
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels =  skip_channels

        self.filter_width = 2

        self.inputs = tf.placeholder(tf.float32,
                                shape=(None, num_time_samples, num_channels))
        targets = tf.placeholder(tf.int32, shape=(None, num_time_samples))


    def get_loss_function(self):
        return tf.losses.mean_squared_error

    def initial_loss_function(self, target, name='loss'):
        with tf.name_scope(name):
            prediction = self.calculate_prediction()


    def calculate_prediction(self, network_output):
        return np.argmax(network_output) #TODO: check the format of the output vector (aka, is it batched?)


    def create_network(self, network_input):
        #Initialize
        skip_connections = []
        #create causal layer
        next_input = self.create_causal_layer(network_input, self.filter_width, self.residual_channels)
        #for each dilation layer
        with tf.name_scope('dilation_layers'):
            for i, dilation in enumerate(self.dilations): #TODO: check this
                with tf.name_scope('layer{}'.format(i)):
                    #create new dilation layer with previous outputs as inputs
                    #get list of skips
                    skip_connection, next_input = self.create_dilation_layer(next_input, dilation, i)
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
        prediction = np.argmax(output)
        return prediction

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

    def dialated_convolution(layer_input, dilation_filter, dilation, name):
        #create scope
        with tf.name_scope(name):
            #pad dialation to input
            padding = [[0,0] , [dilation, 0],[0,0]]
            padded_input = tf.pad(layer_input, padding)

            #Convert to dilated sequence: unsure if necessary
            # padded_shape = shape = tf.shape(padded_input)
            # pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
            # padded = tf.pad(padded_input, [[0, 0], [pad_elements, 0], [0, 0]])
            # reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
            # transposed = tf.transpose(reshaped, perm=[1, 0, 2])
            # dilated_sequence = tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])

            return tf.nn.conv1d(padded_input, dilation_filter, stride=1, padding='VALID', dilation_rate=dilation)

    def create_dilation_layer(self,
                            layer_input,
                            dilation_size,
                            layer_num):
        shape = [dilation_size, self.residual_channels, self.dilation_channels]
        init = tf.contrib.layers.xavier_initializer_conv2d()

        with tf.variable_scope("dilated_conv"):
            gate_weights = tf.Variable(init(shape=shape), "GateWeightL{}".format(layer_num))
            filter_weights = tf.Variable(init(shape=shape), "FilterWeightL{}".format(layer_num))

            convolutional_filter = self.dilated_conv(layer_input,
                                                     filter_weights,
                                                     dilation_size,
                                              )
            convolutional_gate = self.dilated_conv(layer_input,
                                                   gate_weights,
                                                   dilation_size,
                                              )

            output = tf.mul(tf.tanh(convolutional_filter), tf.sigmoid(convolutional_gate)) #TODO: check multiply or add? seen it both ways

            # Need to split into output for the skip channel and the normal channel
            output_shape = [1, self.dilation_channels, self.skip_channels]
            dense_weight = tf.Variable(init(shape=output_shape), "DenseWeightL{}".format(layer_num))
            skip_weight = tf.Variable(init(shape=output_shape), "SkipWeightL{}".format(layer_num))

            dense_output = tf.nn.conv1d(output, dense_weight, stride=1, padding='VALID')
            dense_output = dense_output + layer_input

            # Skipped channel
            skipped_output = self.nn.conv1d(output, skip_weight, stride=1, padding='VALID')

            return skipped_output, dense_output

    def create_1x1_conv_layer(self, inputs, name):
        init = tf.contrib.layers.xavier_initializer_conv2d()

        #create 2d convolutional layer filter w/initialization for specified shape
        filter_weights = tf.Variable(init(shape=(1,1)), name)

        #do convolution
        return tf.nn.conv1d(inputs, filter_weights, stride=1, padding='VALID')
