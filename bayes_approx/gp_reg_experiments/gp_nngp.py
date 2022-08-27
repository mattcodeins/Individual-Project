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


def make_nngp(num_layers, h_dim, w_std, b_std):
    if num_layers == 1:
        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(h_dim, W_std=w_std, b_std=b_std), stax.Relu(),
            stax.Dense(1, W_std=w_std, b_std=b_std)
        )
    elif num_layers == 2:
        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(h_dim, W_std=w_std, b_std=b_std), stax.Relu(),
            stax.Dense(h_dim, W_std=w_std, b_std=b_std), stax.Relu(),
            stax.Dense(1, W_std=w_std, b_std=b_std)
        )
    elif num_layers == 3:
        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(h_dim, W_std=w_std, b_std=b_std), stax.Relu(),
            stax.Dense(h_dim, W_std=w_std, b_std=b_std), stax.Relu(),
            stax.Dense(h_dim, W_std=w_std, b_std=b_std), stax.Relu(),
            stax.Dense(1, W_std=w_std, b_std=b_std)
        )
    elif num_layers == 4:
        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(h_dim, W_std=w_std, b_std=b_std), stax.Relu(),
            stax.Dense(h_dim, W_std=w_std, b_std=b_std), stax.Relu(),
            stax.Dense(h_dim, W_std=w_std, b_std=b_std), stax.Relu(),
            stax.Dense(h_dim, W_std=w_std, b_std=b_std), stax.Relu(),
            stax.Dense(1, W_std=w_std, b_std=b_std)
        )
    elif num_layers == 5:
        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(h_dim, W_std=w_std, b_std=b_std), stax.Relu(),
            stax.Dense(h_dim, W_std=w_std, b_std=b_std), stax.Relu(),
            stax.Dense(h_dim, W_std=w_std, b_std=b_std), stax.Relu(),
            stax.Dense(h_dim, W_std=w_std, b_std=b_std), stax.Relu(),
            stax.Dense(h_dim, W_std=w_std, b_std=b_std), stax.Relu(),
            stax.Dense(1, W_std=w_std, b_std=b_std)
        )
    return init_fn, apply_fn, kernel_fn


def predict(model, x, K=50, key=random.PRNGKey(1)):
    init_fn, apply_fn = model

    if K == 1:
        key, net_key = random.split(key)
        _, params = init_fn(net_key, (0, 1))
        return apply_fn(params, test.x), torch.tensor([0])


    return y_pred.mean(0), y_pred.std(0)


def full_training(exp_name=None, n_epochs=10000,
                  num_layers=2, h_dim=50,
                  w_std=1.0, b_std=1.0, lik_std=0.05):
    torch.manual_seed(1)
    key = random.PRNGKey(1)

    train_loader, test_loader, train, test, noise_std = d.create_regression_dataset()

    # nngp init
    init_fn, apply_fn, kernel_fn = make_nngp(num_layers, h_dim, w_std, b_std)

    # kwargs = dict(
    #     f=apply_fn,
    #     trace_axes=(),
    #     vmap_axes=0
    # )

    apply_fn = jit(apply_fn)
    kernel_fn = jit(kernel_fn, static_argnames='get')

    k_train_train = kernel_fn(train.x, train.x, 'nngp', params)
    k_test_train = kernel_fn(test.x, train.x, 'nngp', params)
    k_test_test = kernel_fn(test.x, test.x, 'nngp', params)

    # predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, X, Y, diag_reg=1e-4)
    predict_fn = nt.predict.gp_inference(k_train_train, train.y, diag_reg=1e-4)

    nngp_mean, nngp_covariance = predict_fn('nngp', k_test_train, k_test_test)
    # nngp_mean, nngp_covariance = predict_fn(x_test=test.x, get='nngp', compute_cov=True)

    nngp_mean = np.reshape(nngp_mean, (-1,))
    nngp_std = np.sqrt(np.diag(nngp_covariance))

    plot_posterior(train, test, nngp_mean, nngp_std)


if __name__ == '__main__':
    full_training()
