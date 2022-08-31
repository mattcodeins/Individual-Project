import gpflow
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from gpflow.utilities import print_summary

import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import datasets.gp_reg_dataset.gp_regression as d
from modules.bnn.utils import uniquify


def log_marg_lik(train, mean_func, kernel, likelihood):
    K = kernel(train.x)
    k_diag = tf.linalg.diag_part(K)
    ks = tf.linalg.set_diag(K, k_diag + (likelihood.variance.numpy()))
    L = tf.linalg.cholesky(ks)
    m = mean_func(train.x)

    # [R,] log-likelihoods for each independent dimension of Y
    log_prob = gpflow.logdensities.multivariate_normal(train.y, m, L)
    return tf.reduce_sum(log_prob)


def full_training(kernel='arccosine', lik_learnable=True, lik_std=0.1, exp_name=None):
    np.random.seed(1)
    tf.random.set_seed(1)

    if exp_name == 'hyper':
        exp_name = (f'GP_GPtoyreg_kernel{kernel}_ll{lik_learnable}_ls{lik_std}')
    exp_name = uniquify(exp_name)

    # import dataset
    _, _, train, test, noise_std = d.create_regression_dataset()

    if kernel == 'arccosine':
        k = gpflow.kernels.ArcCosine(1)
    elif kernel == 'sqexp':
        k = gpflow.kernels.SquaredExponential()
    elif kernel == 'matern':
        k = gpflow.kernels.Matern52()
    print_summary(k)

    m = gpflow.models.GPR(data=(train.x, train.y), kernel=k, mean_function=None)
    if not lik_learnable:
        m.likelihood.variance.assign(lik_std**2)
        gpflow.set_trainable(m.likelihood, False)
    lik_var = m.likelihood.variance.numpy()
    print_summary(m)
    d.gpflow_plot_bnn_pred_post(
        m, train, test, lik_var, f'GP with {kernel} kernel before optimisation', exp_name+'_before'
    )

    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=1000))
    # print(opt_logs)
    print_summary(m)
    lik_var = m.likelihood.variance.numpy()

    d.gpflow_plot_bnn_pred_post(
        m, train, test, lik_var, f'GP with {kernel} kernel after optimisation', exp_name
    )

    d.gpflow_mse_train(m, train, lik_var)
    d.gpflow_test_step(m, test, train, 'mse', lik_var)
    d.gpflow_test_step(m, test, train, 'gnll', lik_var)

    print('gpflow ML: {}'.format(m.log_marginal_likelihood()))

    # print(log_marg_lik(train, m.mean_function, m.kernel, m.likelihood))


if __name__ == '__main__':
    full_training(kernel='matern', lik_learnable=True, lik_std=0.1, exp_name='hyper')
