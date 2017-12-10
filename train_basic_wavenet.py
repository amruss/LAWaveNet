import argparse
import os

import numpy as np
import tensorflow as tf
from scipy.io import wavfile
from tqdm import tqdm

from basic_wavenet import BasicWavenet

#import matplotlib.pyplot as plt


argparser = argparse.ArgumentParser()
argparser.add_argument('model_name', type=str, default="tester")
argparser.add_argument('--model', type=str, default="simple-wavenet")
argparser.add_argument('--learning_rate', type=float, default=1e-5)
argparser.add_argument('--iterations', type=int, default=10000)
argparser.add_argument('--layers', type=int, default=2)
argparser.add_argument('--momentum', type=float, default=0.9)
argparser.add_argument('--save_every', type=int, default=50)
argparser.add_argument('--data_directory', type=str, default="fma_small_wav/015/electronic_noises/")
args = argparser.parse_args()


def _train(model, inputs, targets):
    feed_dict = {model.inputs: inputs, model.targets: targets}
    cost, _ = model.sess.run(
        [model.cost, model.train_step],
        feed_dict=feed_dict)
    return cost


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
        inputs.append(x_s)
        targets.append(y_s)

    return inputs, targets

if __name__ == "__main__":
    inputs, targets = make_dataset(args.data_directory)
    num_time_samples = inputs[0].shape[1]
    num_channels = 1
    gpu_fraction = 1.0
    x_placeholder = tf.placeholder(tf.float32,
                            shape=(None, num_time_samples, num_channels))
    y_placeholder = tf.placeholder(tf.int32, [None, num_time_samples])
    wavenet = BasicWavenet(
                            x_placeholder, y_placeholder,
                             num_time_samples=num_time_samples)
    train(wavenet, inputs, targets)