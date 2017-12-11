import tensorflow as tf
import numpy as np
import math


#########Variables################
# dilation_channels: How many filters to learn for the dilated convolution.
#################################

class BasicWavenet(object):
    def __init__(self, x_placeholder, y_placeholder,
                 num_classes=256,
                 num_layers=2,
                 num_hidden=128,
                 ):

        inputs = x_placeholder
        targets = y_placeholder

        layer_input = inputs
        init = tf.contrib.layers.xavier_initializer_conv2d()

        layers = []
        skips = []
        for l in range(num_layers):
            rate = 2**l
            name_f = 'layer{}filter'.format(l)
            name_g = 'layer{}gate'.format(l)
            filter_input = self.create_dilated_conv1d(layer_input, num_hidden, rate=rate, name=name_f)
            gate_input = self.create_dilated_conv1d(layer_input, num_hidden, rate=rate, name=name_g)
            layer_output = tf.multiply(tf.tanh(filter_input), tf.sigmoid(gate_input))

            # Need to split into output for the skip channel and the normal channel
            output_shape = [1, num_hidden, num_hidden]
            dense_weight = tf.get_variable(shape=output_shape, initializer=init, name="DenseWeightL{}".format(l))
            skip_weight = tf.get_variable(shape=output_shape, initializer=init, name="SkipWeightL{}".format(l))
            dense_output = tf.nn.conv1d(layer_output, dense_weight, stride=1, padding='VALID', data_format='NHWC')
            dense_output = dense_output + layer_input

            # Skipped channel
            skipped_output = tf.nn.conv1d(layer_output, skip_weight, stride=1, padding='VALID')

            layer_input = dense_output
            layers.append(layer_input)
            skips.append(skipped_output)

        skip_sum = tf.add_n(skips)
        skip_relu_1 = tf.nn.relu(skip_sum)
        skip_density_1 = tf.get_variable(shape=[1, num_hidden, num_classes], initializer=init, name="skip1")
        skip_conv_1 = tf.nn.conv1d(skip_relu_1, skip_density_1, stride=1, padding='VALID')
        skip_relu_2 = tf.nn.relu(skip_conv_1)
        skip_density_2 = tf.get_variable(shape=[1, num_classes, num_classes], initializer=init, name="skip_2")
        skip_conv_2 = tf.nn.conv1d(skip_relu_2, skip_density_2, stride=1, padding='VALID')
        # TODO: Do I have to do preprocessing ln(1+mu*x)/ln(1+mu)??
        outputs = tf.nn.softmax(skip_conv_2)

        costs = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=outputs, labels=targets)
        cost = tf.reduce_mean(costs)

        train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=1.0)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess.run(tf.initialize_all_variables())

        self.inputs = inputs
        self.targets = targets
        self.outputs = outputs
        self.hs = layers
        self.costs = costs
        self.cost = cost
        self.train_step = train_step
        self.sess = sess


    def transform_convolution(self, inputs, rate):
        ''' reshapes input for dilated convolutions
        '''
        permutation = (1, 0, 2)
        _, width, num_channels = inputs.get_shape().as_list()

        padded_width = int(rate * np.ceil((width + rate) * 1.0 / rate))
        pad_left = padded_width - width
        new_shape = (int(padded_width / rate), -1, num_channels)
        padded = tf.pad(inputs, [[0, 0], [pad_left, 0], [0, 0]])
        transposed = tf.transpose(padded, permutation)
        reshaped = tf.reshape(transposed, new_shape)
        outputs = tf.transpose(reshaped, permutation)
        return outputs


    def transform_back(self, inputs, rate, crop_left=0):
        permutation = (1, 0, 2)
        old_shape = tf.shape(inputs)
        width = old_shape[1]
        out_width = tf.to_int32(width * rate)
        _, _, num_channels = inputs.get_shape().as_list()

        new_shape = (out_width, -1, num_channels)  # missing dim: batch_size
        transposed = tf.transpose(inputs, permutation)
        reshaped = tf.reshape(transposed, new_shape)
        outputs = tf.transpose(reshaped, permutation)
        cropped = tf.slice(outputs, [0, crop_left, 0], [-1, -1, -1])
        return cropped


    def create_conv1d(self, inputs,
               out_channels,
               filter_width=2,
               stride=1,
               padding='VALID',
               gain=np.sqrt(2),
               activation=tf.nn.relu,
               ):
        in_channels = inputs.get_shape().as_list()[-1]
        w_init = tf.contrib.layers.xavier_initializer_conv2d()
        w = tf.get_variable(name='w',
                            shape=(filter_width, in_channels, out_channels),
                            initializer=w_init)

        layer_outputs = tf.nn.conv1d(inputs,
                               w,
                               stride=stride,
                               padding=padding,
                               data_format='NHWC')

        outputs = activation(layer_outputs)

        return outputs


    def create_dilated_conv1d(self, inputs,
                       out_channels,
                       filter_width=2,
                       rate=1,
                       padding='VALID',
                       name=None,
                       gain=np.sqrt(2),
                       activation=tf.nn.relu):
        assert name
        with tf.variable_scope(name):
            _, width, _ = inputs.get_shape().as_list()
            transformed_inputs = self.transform_convolution(inputs, rate=rate)
            transformed_outputs = self.create_conv1d(transformed_inputs,
                              out_channels=out_channels,
                              filter_width=filter_width,
                              padding=padding,
                              gain=gain,
                              activation=activation)
            _, out_width, _ = transformed_outputs.get_shape().as_list()
            new_width = out_width * rate
            diff = new_width - width
            outputs = self.transform_back(transformed_outputs, rate=rate, crop_left=diff)
            tensor_shape = [tf.Dimension(None),
                            tf.Dimension(width),
                            tf.Dimension(out_channels)]
            outputs.set_shape(tf.TensorShape(tensor_shape))

        return outputs
