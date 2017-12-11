import argparse
import os

import numpy as np
import tensorflow as tf
from scipy.io import wavfile
from tqdm import tqdm
import sys

from basic_wavenet import BasicWavenet

argparser = argparse.ArgumentParser()
argparser.add_argument('model_name', type=str, default="tester")
argparser.add_argument('--model', type=str, default="simple-wavenet")
argparser.add_argument('--learning_rate', type=float, default=1e-5)
argparser.add_argument('--iterations', type=int, default=10000)
argparser.add_argument('--layers', type=int, default=2)
argparser.add_argument('--hidden_size', type=int, default=128)
argparser.add_argument('--momentum', type=float, default=0.9)
argparser.add_argument('--save_every', type=int, default=50)
argparser.add_argument('--data_directory', type=str, default="fma_small_wav/015/electronic_noises/")
argparser.add_argument('--log_directory', type=str, default="test_model/")
argparser.add_argument('--max_checkpoints', type=int, default=100)
argparser.add_argument('--checkpoint_every', type=int, default=100)
argparser.add_argument('--num_time_samples', type=int, default=1000000)
args = argparser.parse_args()


def train(model, inputs, targets):
    losses = []
    for i in tqdm(range(args.iterations)):
        index = i % len(inputs)
        input = inputs[index]
        target = targets[index]
        feed_dictionary = {model.inputs: input, model.targets: target}
        loss, _ = model.sess.run([model.cost, model.train_step], feed_dict=feed_dictionary)
        losses.append(loss)
        print loss
        if i % args.checkpoint_every == 0:
            save(saver, model.sess, i)

def save(saver, sess, step):
    model_name = args.model_name
    checkpoint_path = os.path.join("basic_wavenet_log/" + args.log_directory, model_name)
    print('Storing checkpoint to {} ...'.format(args.log_directory))
    sys.stdout.flush()

    if not os.path.exists(args.log_directory):
        os.makedirs(args.log_directory)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')

def make_dataset(data_directory):
    inputs = []
    targets = []

    for filename in os.listdir(data_directory):
        if filename[-4:] != ".wav":
            continue

        # Read File
        print "Adding data for " + data_directory + filename
        data = wavfile.read(data_directory + filename)[1][:, 0]

        # Normalize the data
        temp = np.float32(data) - np.min(data)
        data_normalized = (temp / np.max(temp) - 0.5) * 2

        # Quantize inputs and targets.
        x_s = np.digitize(data_normalized[0:-1], np.linspace(-1, 1, 256), right=False) - 1
        x_s = np.linspace(-1, 1, 256)[x_s][None, :, None]
        y_s = (np.digitize(data_normalized[1::], np.linspace(-1, 1, 256), right=False) - 1)[None, :]
        for i in range (x_s.shape[1]/args.num_time_samples):
            begin = args.num_time_samples * i
            end = args.num_time_samples * i + args.num_time_samples
            inputs.append(x_s[:, begin:end, :])
            targets.append(y_s[:, begin:end])

    return inputs, targets

if __name__ == "__main__":
    inputs, targets = make_dataset(args.data_directory)
    num_time_samples = inputs[0].shape[1]
    print num_time_samples
    num_channels = 1
    gpu_fraction = 1.0

    # Set up logging for TensorBoard.
    writer = tf.summary.FileWriter(args.log_directory)
    writer.add_graph(tf.get_default_graph())
    run_metadata = tf.RunMetadata()
    summaries = tf.summary.merge_all()

    x_placeholder = tf.placeholder(tf.float32,
                             shape=(None, num_time_samples, num_channels))
    y_placeholder = tf.placeholder(tf.int32, [None, num_time_samples])
    wavenet = BasicWavenet(x_placeholder, y_placeholder, num_layers=args.layers, num_hidden=args.hidden_size)
    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=args.max_checkpoints)
    try:
        train(wavenet, inputs, targets)
    except KeyboardInterrupt:
        pass