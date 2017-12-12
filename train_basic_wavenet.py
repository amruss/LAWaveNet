import argparse
import os

import numpy as np
import tensorflow as tf
from scipy.io import wavfile
from tqdm import tqdm
from basic_generator import *
import sys

from basic_wavenet import *

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
argparser.add_argument('--log_directory', type=str, default="test_model5/")
argparser.add_argument('--checkpoint_every', type=int, default=3)
argparser.add_argument('--num_time_samples', type=int, default=1000000)
argparser.add_argument('--wave_name', type=str, default="generated")
args = argparser.parse_args()



def train(model, inputs, targets):

    losses = []

    for i in tqdm(range(args.iterations)):
        index = i % len(inputs)
        input = inputs[index]
        target = targets[index]
        feed_dictionary = {model.inputs: input, model.targets: target}
        loss, _, summ = model.sess.run([model.cost, model.train_step, model.summaries], run_metadata=model.run_metadata, feed_dict=feed_dictionary)
        losses.append(loss)
        model.writer.add_run_metadata(model.run_metadata, 'step_{:04d}'.format(i))
        model.writer.add_summary(summ, i)
        if i % args.checkpoint_every == 0:
            print(loss)
            save(model.saver, model.sess, i)

        if i % 10000 == 0:
            generator = Generator(model)
            input_ = input[:, 0:1, 0]
            predictions = generator.run(input_, 32000, args.wav_name)
            tf.summary.histogram("Generated iteration " + str(i), predictions)

def save(saver, sess, step):
    model_name = args.model_name
    #checkpoint_path = os.path.join(args.log_directory, model_name)
    print('Storing checkpoint to {} ...'.format(args.log_directory))
    sys.stdout.flush()


    if not os.path.exists(args.log_directory):
        os.makedirs(args.log_directory)

    saver.save(sess, args.log_directory, global_step=step)
    print(' Done.')

def make_dataset(data_directory):
    inputs = []
    targets = []

    for filename in os.listdir(data_directory):
        if filename[-4:] != ".wav":
            continue

        # Read File
        print("Adding data for " + data_directory + filename)
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
    num_channels = 1
    gpu_fraction = 1.0

    x_placeholder = tf.placeholder(tf.float32,
                             shape=(None, num_time_samples, num_channels))
    y_placeholder = tf.placeholder(tf.int32, [None, num_time_samples])
    wavenet = BasicWavenet(x_placeholder, y_placeholder, args.log_directory, num_layers=args.layers, num_hidden=args.hidden_size)
    try:
        train(wavenet, inputs, targets)
    except KeyboardInterrupt:
        pass