import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

from gpflow.likelihoods import MultiClass
from gpflow.kernels import RBF, White
from gpflow.models.svgp import SVGP
# from gpflow.training import AdamOptimizer

from scipy.stats import mode
from scipy.cluster.vq import kmeans2

# from doubly_stochastic_dgp.dgp import DGP

import tensorflow_datasets as tfds
from tensorflow.keras.datasets import mnist
import os

def _seperate_X_Y(ds):
    X = []
    Y = []
    for i, ex in enumerate(tfds.as_numpy(ds)):
        if i < 2000:
            X.append(ex['image'].reshape(784))
            Y.append(ex['label'])
        else:
            break

    X = np.stack(X).astype(float)
    Y = np.stack(Y)
    data = (X,Y)
    return data

def load_mnist_from_tf():
    train_ds, test_ds = tfds.load('mnist', split=['train', 'test'])
    train_data = _seperate_X_Y(train_ds)
    test_data = _seperate_X_Y(test_ds)
    return train_data, test_data

def load_mnist_from_local():
    current_dir = os.getcwd()
    data = mnist.load_data(path=current_dir+'/mnist.npz')
    return data

def svgp_model():
    pass


if __name__ == '__main__':
    train_data, test_data = load_mnist_from_tf()
    X, Y = test_data
    print(X.shape)
