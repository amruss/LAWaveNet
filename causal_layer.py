
import tensoflow as tf

#TODO: is there anything with scoping I'm missing?

def create_causal_layer(layer_input, filter_width, n_residual_channels):
    #create scope
    with tf.name_scope('causal_convolution'):
        #determine shape
        shape = [filter_width, 1, n_residual_channels]

        #create layer & layer variable
        init = tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32)

        #create 2d convolutional layer filter w/initialization for specified shape
        filter_weights = tf.Variable(init(shape=shape), 'causal_layer')

        #pad filter_width-1 to input
        padding = [[0,0] , [filter_width-1, 0],[0,0]]

        new_input = tf.pad(layer_input, padding)

        #do convolution
        return tf.nn.conv1d(new_input, filter_weights, stride=1, padding='VALID')
