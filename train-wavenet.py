import argparse
import os
from scipy.io import wavfile
import numpy as np


argparser = argparse.ArgumentParser()
argparser.add_argument('--model', type=str, default="simple-wavenet")
argparser.add_argument('--learning_rate', type=float, default=0.0005)
argparser.add_argument('--iterations', type=int, default=10000)
argparser.add_argument('--layers', type=int, default=2)
argparser.add_argument('--data_directory', type=str, default="fma_small_wav/015/electronic_noises/")
args = argparser.parse_args()


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
    make_dataset(args.data_directory)


