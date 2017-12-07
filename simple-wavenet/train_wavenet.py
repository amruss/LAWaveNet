import argparse
import os

import numpy as np
import tensorflow as tf
from scipy.io import wavfile
from tqdm import tqdm

from simple_wavenet import SimpleWavenet

import matplotlib.pyplot as plt


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


DILATIONS =  [1, 2, 4, 8, 16]

#TODO: test code
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

def train(model, inputs, targets, session):
    saver = tf.train.Saver()
    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate, epsilon=1e-4) #TODO: use momentum
    trainable = tf.trainable_variables()
    loss = model.loss()
    grad_update = optimizer.minimize(loss, var_list=trainable)
    losses = []

    for epoch in tqdm(range(1, args.iterations + 1)):
        index = epoch % len(inputs)
        input = inputs[index]
        target = targets[index]

        cost, _ = session.run([loss, optimizer], feed_dict={model.inputs: input, model.targets: target})
        losses.append(cost)
        if epoch % 50 == 0:
            print "Loss for iteration " + str(epoch) + " is " + str(cost)
        if epoch % 250 ==0:
            plt.plot(losses)
            plt.show()
        if epoch % args.save_every == 0:
            #source: https://stackoverflow.com/questions/33759623/tensorflow-how-to-save-restore-a-model
            saver.save(sess, args.model_name, global_step=epoch)



if __name__ == "__main__":
    inputs, targets = make_dataset(args.data_directory)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(tf.initialize_all_variables())
    wavenet = SimpleWavenet(1, 2, DILATIONS)
    train(wavenet, inputs, targets, sess)


