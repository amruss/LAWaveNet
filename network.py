import tensorflow as tf
form causal_layer import *

DILATION_LAYERS = [1,2,8]
CAUSAL_FILTER_WIDTH = ??
N_RESIDUAL_CHANNELS = ??
def create_network():
    #Initialize
    skip_connections = []
    #create causal layer
    next_input = create_causal_layer(layer_input, filter_width, n_residual_channels)
    #for each dilation layer
    with tf.name_scope('dilation_layers'):
        for i in range(DILATION_LAYERS):
            layer_num = i
            dilation = DILATION_LAYERS[i]
            #create new dilation layer with previous outputs as inputs
            #get list of skips
            skip_connection, next_input = dilation_layer(next_input)
            skip_connections.append(skip_connection)
    with tf.name_scope('final_steps'):
        #sum skips
        skip_sum = tf.add_n(skip_connections)
        #apply relu
        skip_relu_1 = tf.relu(skip_sum)
        #1x1 convolutional filter
        skip_conv_1 = create_1x1_conv_layer(skip_relu_1, 1)
        #apply relu
        skip_relu_2 = tf.relu(skip_conv_1)
        #1x1 convolutional filter
        skip_conv_2 = create_1x1_conv_layer(skip_relu_2, 2)
        #softmax
        #TODO: Do I have to do preprocessing ln(1+mu*x)/ln(1+mu)??
        output = tf.softmax(skip_conv_2)
        return output

def get_loss_function():
    loss = tf.losses.mean_squared_error(lables, predictions)
    #loss = tf.reduce_sum(tf.square(tf.sub(out, y)))


def create_1x1_conv_layer(inputs, i):
    init = tf.contrib.layers.xavier_initializer_conv2d()

    #create 2d convolutional layer filter w/initialization for specified shape
    filter_weights = tf.Variable(init(shape=(1,1)), 'final_conv_layer_'+ str(i))

    #do convolution
    return tf.nn.conv1d(inputs, filter_weights, stride=1, padding='VALID')
