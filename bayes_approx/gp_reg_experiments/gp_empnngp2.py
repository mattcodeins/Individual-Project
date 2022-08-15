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
key = random.PRNGKey(1)


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
    plt.show()


def plot_fn(X, Y, Xt, Yt, *fs):
    plt.plot(X, Y, 'ro', markersize=10, label='train')

    if test is not None:
        plt.plot(Xt, Yt, 'k--', linewidth=3, label='$f(x)$')

        for f in fs:
            plt.plot(Xt, f(Xt), '-', linewidth=3)

    format_plot('$x$', '$f$')


def plot_prior(X, Y, Xt, Yt, params, key=random.PRNGKey(1)):
    prior_draws = []
    for _ in range(10):
        key, net_key = random.split(key)
        _, params = init_fn(net_key, (0, 1))
        prior_draws += [apply_fn(params, Xt)]

    legend = functools.partial(plt.legend, fontsize=10)

    plot_fn(X, Y, Xt, Yt)

    for p in prior_draws:
        plt.plot(Xt, p, linewidth=3, color=[1, 0.65, 0.65])

    legend(['train', '$f(x)$', 'random draw'], loc='upper left')

    finalize_plot((0.85, 0.6))


def plot_prior_std(X, Y, Xt, Yt, params, kernel, key=random.PRNGKey(1)):
    std_dev = np.sqrt(np.diag(kernel))

    plot_fn(X, Y, Xt, Yt)

    plt.fill_between(np.reshape(X, (-1,)), 2*std_dev, -2*std_dev, alpha=0.4)

    prior_draws = []
    for _ in range(10):
        key, net_key = random.split(key)
        _, params = init_fn(net_key, (0, 1))
        prior_draws += [apply_fn(params, Xt)]

    for p in prior_draws:
        plt.plot(Xt, p, linewidth=3, color=[1, 0.65, 0.65])

    finalize_plot((0.85, 0.6))
    print('hey')


def plot_posterior(X, Y, Xt, Yt, nngp_mean, nngp_std):
    plot_fn(X, Y, Xt, Yt)

    plt.plot(Xt, nngp_mean, 'r-', linewidth=3)
    plt.fill_between(
        np.reshape(Xt, (-1)),
        nngp_mean - 2 * nngp_std,
        nngp_mean + 2 * nngp_std,
        color='red', alpha=0.2)

    # plt.ylim((-5, 5))

    legend = functools.partial(plt.legend, fontsize=10)
    legend(['Train', 'f(x)', 'Bayesian Inference'], loc='upper left')

    finalize_plot((0.85, 0.6))


train, test, noise_std = d.import_train_test()

X = train[:,0].reshape(-1, 1)
Y = train[:,1].reshape(-1, 1)
train = (X, Y)
Xt = test[:,0].reshape(-1, 1)
Yt = test[:,1].reshape(-1, 1)
test = (Xt, Yt)

# plt.plot(X, Y, "kx", mew=2, label='noisy sample points')
# plt.legend()
# plt.show()

# nngp init
h_dim = 50
num_layers = 2

# x_dim, y_dim = 1, 1
# init_fn, apply_fn, kernel_fn = stax.serial(
#     stax.Dense(h_dim, W_std=3.2, b_std=0.5), stax.Relu(),
#     stax.Dense(h_dim, W_std=3.2, b_std=0.5), stax.Relu(),
#     stax.Dense(h_dim, W_std=3.2, b_std=0.5), stax.Relu(),
#     stax.Dense(h_dim, W_std=3.2, b_std=0.5), stax.Relu(),
#     stax.Dense(h_dim, W_std=3.2, b_std=0.5), stax.Relu(),
#     stax.Dense(y_dim)
# )

x_dim, y_dim = 1, 1
init_fn, f, _ = stax.serial(
    stax.Dense(h_dim), stax.Relu(),
    stax.Dense(h_dim), stax.Relu(),
    stax.Dense(h_dim), stax.Relu(),
    stax.Dense(h_dim), stax.Relu(),
    stax.Dense(y_dim)
)

_, params = init_fn(key, X.shape)

kernel_fn = nt.empirical_kernel_fn(
    f, trace_axes=(-1,), vmap_axes=0,
    implementation=nt.NtkImplementation.JACOBIAN_CONTRACTION)

# apply_fn = jit(f)
# kernel_fn = jit(kernel_fn, static_argnames='get')

# plot_prior(X, Y, Xt, Yt, params, key)

k_train_train = kernel_fn(X, X, 'nngp', params)
k_test_train = kernel_fn(Xt, X, 'nngp', params)
k_test_test = kernel_fn(Xt, Xt, 'nngp', params)

# plot_prior_std(X, Y, Xt, Yt, params, kernel, key)

# predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, X, Y, diag_reg=1e-4)
predict_fn = nt.predict.gp_inference(k_train_train, Y, diag_reg=1e-4)
# nngp_fn = nt.empirical_nngp_fn(f, diagonal_axes=(0,))

# nngp_mean, nngp_covariance = predict_fn(get='nngp', compute_cov=True)
nngp_mean, nngp_covariance = predict_fn('nngp', k_test_train, k_test_test)

print(nngp_mean.shape)
print(nngp_covariance.shape)

nngp_mean = np.reshape(nngp_mean, (-1,))
nngp_std = np.sqrt(np.diag(nngp_covariance))

plot_posterior(X, Y, Xt, Yt, nngp_mean, nngp_std)
