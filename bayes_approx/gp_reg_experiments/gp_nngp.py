import jax.numpy as np
from jax import random
from jax.example_libraries import optimizers
from jax import jit, grad, vmap
import functools
import neural_tangents as nt
from neural_tangents import stax
import matplotlib.pyplot as plt
import torch

import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import datasets.gp_reg_dataset.gp_regression as d

torch.manual_seed(1)
key = random.PRNGKey(10)

train, test, noise_std = d.import_train_test()

X = train[:,0].reshape(-1, 1)
Y = train[:,1].reshape(-1, 1)
Xt = test[:,0].reshape(-1, 1)
Yt = test[:,1].reshape(-1, 1)

plt.plot(X, Y, "kx", mew=2, label='noisy sample points')
plt.legend()
plt.show()

# nngp init
h_dim = 50
num_layers = 2

x_dim, y_dim = 1, 1
init_fn, apply_fn, kernel_fn = stax.serial(
    stax.Dense(h_dim), stax.Relu(),
    stax.Dense(h_dim), stax.Relu(),
    stax.Dense(y_dim)
)

apply_fn = jit(apply_fn)
kernel_fn = jit(kernel_fn, static_argnames='get')

prior_draws = []
for _ in range(10):
    key, net_key = random.split(key)
    _, params = init_fn(net_key, (-1, 1))
    prior_draws += [apply_fn(params, Xt)]


def format_plot(x=None, y=None):
    # plt.grid(False)
    ax = plt.gca()
    if x is not None:
        plt.xlabel(x, fontsize=20)
    if y is not None:
        plt.ylabel(y, fontsize=20)


def finalize_plot(shape=(1, 1)):
    plt.gcf().set_size_inches(
        shape[0] * 1.5 * plt.gcf().get_size_inches()[1],
        shape[1] * 1.5 * plt.gcf().get_size_inches()[1])
    plt.tight_layout()


legend = functools.partial(plt.legend, fontsize=10)


def plot_fn(X, Y, Xt, Yt, *fs):
    plt.plot(X, Y, 'ro', markersize=10, label='train')

    if test is not None:
        plt.plot(Xt, Yt, 'k--', linewidth=3, label='$f(x)$')

        for f in fs:
            plt.plot(Xt, f(Xt), '-', linewidth=3)

    plt.xlim([-np.pi, np.pi])
    plt.ylim([-1.5, 1.5])

    format_plot('$x$', '$f$')


plot_fn(X, Y, Xt, Yt)


for p in prior_draws:
    plt.plot(Xt, p, linewidth=3, color=[1, 0.65, 0.65])

legend(['train', '$f(x)$', 'random draw'], loc='upper left')

finalize_plot((0.85, 0.6))

plt.show()
