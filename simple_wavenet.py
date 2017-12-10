import tensorflow as tf
import numpy as np
import math


#########Variables################
# dilation_channels: How many filters to learn for the dilated convolution.
#################################
SEQUENCE_LENGTH = 32

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
        self.dilations = dilation_layer_widths #[[dilation_width**i for i in range(n)] for n in dilation_layer_widths]
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels =  skip_channels

        self.filter_width = 2
        x_placeholder = tf.placeholder('float32', [SEQUENCE_LENGTH, 1])
        y_placeholder = tf.placeholder('float32', [1, 1])

        self.inputs = x_placeholder
        self.targets = y_placeholder

        self.predict = self.init_predict(self.inputs)
        self.loss_func = self.init_loss(self.targets)

    def pred(self):
        return self.predict

    def loss(self):
        return self.loss_func

    def init_predict(self, x):
        x = tf.reshape(tf.cast(x, tf.float32), [self.batch_size, -1, 1])
        out = self.create_network(x)
        #out = tf.reshape(tf.slice(tf.reshape(out, [-1]), begin=[tf.shape(out)[1] - 1], size=[1]), [-1, 1])
        return out #self.calculate_prediction(out)

    def init_loss(self, y, name='loss'):
        with tf.name_scope(name):
            out = self.pred()
            reduced_loss = tf.reduce_sum(tf.square(tf.subtract(out, y)))
            return reduced_loss

    def calculate_prediction(self, network_output):
        return np.argmax(network_output) #TODO: check the format of the output vector (aka, is it batched?)


    def create_network(self, network_input):
        initializer = tf.contrib.layers.xavier_initializer_conv2d()
        network_input = tf.reshape(tf.cast(network_input, tf.float32), [self.batch_size, -1, 1])
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
            skip_relu_1 = tf.nn.relu(skip_sum)
            #1x1 convolutional filter
            skip_density_1 = tf.Variable(initializer(shape=[1, self.skip_channels, self.skip_channels]), name="skip_1")
            skip_conv_1 = tf.nn.conv1d(skip_relu_1, skip_density_1, stride=1, padding='VALID')
            #apply relu
            skip_relu_2 = tf.nn.relu(skip_conv_1)
            #1x1 convolutional filter
            skip_density_2 = tf.Variable(initializer(shape=[1, self.skip_channels, 1]), name="skip_2")
            skip_conv_2 = tf.nn.conv1d(skip_relu_2, skip_density_2, stride=1, padding='VALID')
            #softmax
            #TODO: Do I have to do preprocessing ln(1+mu*x)/ln(1+mu)??
            output = tf.nn.softmax(skip_conv_2)
        return output


    def convolutional_layer(self):
        pass

    def variables(self, ):
        pass

    def create_causal_layer(self, layer_input, filter_width, n_residual_channels):
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

    def dialated_convolution(self, layer_input, dilation_filter, dilation, name):
        #create scope
        with tf.name_scope(name):
            #pad dialation to input
            padding = [[0,0] , [dilation, 0],[0,0]]
            padded_input = tf.pad(layer_input, padding)

            #Convert to dilated sequence: unsure if necessary
            padded_shape = shape = tf.shape(padded_input)
            pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
            padded = tf.pad(padded_input, [[0, 0], [pad_elements, 0], [0, 0]])
            reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
            transposed = tf.transpose(reshaped, perm=[1, 0, 2])
            dilated_sequence = tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])

            layer_output = tf.nn.conv1d(dilated_sequence, dilation_filter, stride=1, padding='VALID')

            shape = tf.shape(layer_output)
            prepared = tf.reshape(layer_output, [dilation, -1, shape[2]])
            transposed = tf.transpose(prepared, perm=[1, 0, 2])
            return tf.reshape(transposed, [tf.div(shape[0], dilation), -1, shape[2]])

    def create_dilation_layer(self,
                            layer_input,
                            dilation_size,
                            layer_num):
        shape = [dilation_size, self.residual_channels, self.dilation_channels]
        init = tf.contrib.layers.xavier_initializer_conv2d()

        with tf.variable_scope("dilated_conv"):
            gate_weights = tf.Variable(init(shape=shape), "GateWeightL{}".format(layer_num))
            filter_weights = tf.Variable(init(shape=shape), "FilterWeightL{}".format(layer_num))

            convolutional_filter = self.dialated_convolution(layer_input,
                                                     filter_weights,
                                                     dilation_size,
                                                     "conv_filter",
                                              )
            convolutional_gate = self.dialated_convolution(layer_input,
                                                   gate_weights,
                                                   dilation_size,
                                                   "conv_gate",
                                              )

            output =  tf.multiply(tf.tanh(convolutional_filter), tf.sigmoid(convolutional_gate)) #TODO: check multiply or add? seen it both ways

            # Need to split into output for the skip channel and the normal channel
            output_shape = [1, self.dilation_channels, self.skip_channels]
            dense_weight = tf.Variable(init(shape=output_shape), "DenseWeightL{}".format(layer_num))
            skip_weight = tf.Variable(init(shape=output_shape), "SkipWeightL{}".format(layer_num))

            dense_output = tf.nn.conv1d(output, dense_weight, stride=1, padding='VALID')
            dense_output = dense_output + layer_input

            # Skipped channel
            skipped_output = tf.nn.conv1d(output, skip_weight, stride=1, padding='VALID')

            return skipped_output, dense_output

    def create_1x1_conv_layer(self, inputs, name):
        init = tf.contrib.layers.xavier_initializer_conv2d()

        #create 2d convolutional layer filter w/initialization for specified shape
        filter_weights = tf.Variable(init(shape=(1,1)), name)

        #do convolution
        return tf.nn.conv1d(inputs, filter_weights, stride=1, padding='VALID')
