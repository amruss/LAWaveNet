import tensorflow as tf
import numpy as np
import matplotlib as plt
import argparse
from train_basic_wavenet import *
import librosa
argparser = argparse.ArgumentParser()
argparser.add_argument('checkpoint', type=str)
argparser.add_argument('--samples_to_generate', type=int, default=160000)
argparser.add_argument('--log_directory', type=str, default="test_model3/")
argparser.add_argument('--layers', type=int, default=2)
argparser.add_argument('--hidden_size', type=int, default=128)
argparser.add_argument('--data_directory', type=str, default="fma_small_wav/015/electronic_noises/")

args = argparser.parse_args()

class Generator(object):
    def __init__(self, model, batch_size=1, input_size=1):
        self.model = model
        inputs = tf.placeholder(tf.float32, [batch_size, input_size],
                                name='inputs')
        enq = []
        push_ops = []
        layer_input = inputs
        state_size = 1

        for l in range(self.model.num_layers):
            rate = 2**l
            q = tf.FIFOQueue(rate, dtypes=tf.float32,
                             shapes=(batch_size, state_size))
            enqueued = q.enqueue_many(tf.zeros((rate, batch_size, state_size)))
            enq.append(enqueued)
            state = q.dequeue()
            enq_input = q.enqueue([layer_input])
            push_ops.append(enq_input)
            name = "layer{}filter".format(l)

            layer_input = self._causal_linear(layer_input, state, name=name, activation=tf.nn.relu)
            state_size = self.model.num_hidden


        outputs = self._output_linear(layer_input, name=name)

        out_ops = [tf.nn.softmax(outputs)]
        out_ops.extend(push_ops)

        self.inputs = inputs
        self.init_ops = enq
        self.out_ops = out_ops

        # Initialize queues.
        self.model.sess.run(self.init_ops)

    def _causal_linear(self, inputs, state, name=None):
        with tf.variable_scope(name, reuse=True) as scope:
            weight = tf.get_variable('w')
            w_1 = weight[0, :, :]
            w_2 = weight[1, :, :]
            output = tf.matmul(inputs, w_2) + tf.matmul(state, w_1)
        return output

    def _output_linear(self, layer_input, name=''):
        with tf.variable_scope(name, reuse=True):
            w = tf.get_variable('w')[0, :, :]
            w_0 = tf.get_variable('b')
            output = tf.matmul(layer_input, w) + tf.expand_dims(w_0, 0)
        return output


    def run(self, input, num_samples, wav_filename):
        predictions = []
        for step in range(num_samples):

            feed_dict = {self.inputs: input}
            output = self.model.sess.run(self.out_ops, feed_dict=feed_dict)[0]  # ignore push ops
            value = np.argmax(output[0, :])

            bin = np.linspace(-1, 1, self.model.num_classes)
            input = np.array(bin[value])[None, None]
            predictions.append(input)

            if step % 1000 == 0:
                predictions_ = np.concatenate(predictions, axis=1)
                librosa.output.write_wav(predictions_, wav_filename, 44100)
                print('Updated wav file at '.format(wav_filename))

        predictions_ = np.concatenate(predictions, axis=1)
        return predictions_



if __name__ == "__main__":
    print "Restoring Model..."
    inputs, targets = make_dataset(args.data_directory)
    num_time_samples = inputs[0].shape[1]
    num_channels = 1
    gpu_fraction = 1.0

    x_placeholder = tf.placeholder(tf.float32,
                                   shape=(None, num_time_samples, num_channels))
    y_placeholder = tf.placeholder(tf.int32, [None, num_time_samples])
    wavenet = BasicWavenet(x_placeholder, y_placeholder, args.log_directory, num_layers=args.layers,
                           num_hidden=args.hidden_size)
    variables_to_restore = {
        var.name[:-2]: var for var in tf.global_variables()
        if not ('state_buffer' in var.name or 'pointer' in var.name)}

    saver = tf.train.Saver(variables_to_restore)

