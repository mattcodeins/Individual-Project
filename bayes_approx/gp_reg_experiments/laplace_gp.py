from re import X
import torch
import torch.nn as nn
import numpy as np
import logging
import csv

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from external.scalemarglik import marglik_optimization
from modules.bnn.utils import uniquify, to_numpy

from datasets.gp_reg_dataset import gp_regression as d

from laplace.curvature import BackPackGGN
from laplace import FullLaplace, KronLaplace


# construct a BNN
def make_linear_nn(layer_sizes, num_layers):
    """
    Create initial NN model.
    Had an issue using loops, so this method is very unpythonic, and only works up to 5 hidden layers.
    """
    # nonlinearity = getattr(nn, activation)() if isinstance(activation, str) else activation
    if num_layers == 1:
        net = nn.Sequential(
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(layer_sizes[1], layer_sizes[2]),
        )
    elif num_layers == 2:
        net = nn.Sequential(
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(layer_sizes[1], layer_sizes[2]),
            nn.ReLU(),
            nn.Linear(layer_sizes[2], layer_sizes[3]),
        )
    elif num_layers == 3:
        net = nn.Sequential(
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(layer_sizes[1], layer_sizes[2]),
            nn.ReLU(),
            nn.Linear(layer_sizes[2], layer_sizes[3]),
            nn.ReLU(),
            nn.Linear(layer_sizes[3], layer_sizes[4]),
        )
    elif num_layers == 4:
        net = nn.Sequential(
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(layer_sizes[1], layer_sizes[2]),
            nn.ReLU(),
            nn.Linear(layer_sizes[2], layer_sizes[3]),
            nn.ReLU(),
            nn.Linear(layer_sizes[3], layer_sizes[4]),
            nn.ReLU(),
            nn.Linear(layer_sizes[4], layer_sizes[5]),
            nn.ReLU(),
        )
    elif num_layers == 5:
        net = nn.Sequential(
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(layer_sizes[1], layer_sizes[2]),
            nn.ReLU(),
            nn.Linear(layer_sizes[2], layer_sizes[3]),
            nn.ReLU(),
            nn.Linear(layer_sizes[3], layer_sizes[4]),
            nn.ReLU(),
            nn.Linear(layer_sizes[4], layer_sizes[5]),
            nn.ReLU(),
            nn.Linear(layer_sizes[5], layer_sizes[6]),
        )
    return net


def predict(lap, x_test, K):
    """
    Monte Carlo sampling of BNN using K samples.
    """
    x_test = torch.tensor(x_test, device='cuda:0')
    f_mu, f_var = lap(x_test)
    f_sigma = f_var.squeeze().sqrt()[:, None]
    # pred_std = np.sqrt(f_sigma**2 + lap.sigma_noise.item()**2)
    f_mu = f_mu
    return f_mu, f_sigma


def save_logs(exp_name, rows):
    if exp_name is not None:
        with open(f"./bayes_approx/results/{exp_name}.csv", 'w') as f:
            writer = csv.writer(f)
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)


def full_training(exp_name=None, n_epochs=1000, num_layers=2, h_dim=50, activation='relu'):
    torch.manual_seed(1)
    if exp_name == 'hyper':
        exp_name = (f'gp_laplacebnn/nl{num_layers}_ne{n_epochs}')
    exp_name = uniquify(exp_name)

    # import dataset
    train_loader, test_loader, train, test, noise_std = d.create_regression_dataset()

    # create bnn
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    x_dim, y_dim = 1, 1
    layer_sizes = [x_dim] + [h_dim for _ in range(num_layers)] + [y_dim]
    if activation == 'relu':
        activation = nn.ReLU()
    elif activation == 'tanh':
        activation = nn.Tanh()
    model = make_linear_nn(layer_sizes, num_layers).to(device)
    print("BNN architecture: \n", model)

    # d.plot_bnn_pred_post(lap, predict, train, test, log_noise_var, noise_std,
    #                      'BNN init (before training, MFVI)', device)

    logging.basicConfig(level=logging.INFO)

    lap, model, margliks, losses = marglik_optimization(
        model, train_loader, likelihood='regression', sigma_noise_init=0.05, backend=BackPackGGN,
        laplace=FullLaplace, n_epochs=n_epochs
    )
    log_lik_var = torch.log(lap.sigma_noise)
    # d.plot_bnn_pred_post(
    #     lap, predict, train, test, log_lik_var, 'Laplace BNN approximate posterior', exp_name, device)

    rows = zip(margliks, losses)
    save_logs(exp_name, rows)

    test_mse = d.mse_test_step(lap, test_loader, train, predict, log_lik_var)
    test_gnll = d.gnll_test_step(lap, test_loader, train, predict, log_lik_var)
    train_mse = d.mse_train(lap, train_loader, predict, log_lik_var)
    train_gnll = d.gnll_train(lap, train_loader, predict, log_lik_var)
    return test_mse, test_gnll, train_mse, train_gnll, margliks[-1], losses[-1]


if __name__ == '__main__':
    # x = full_training(exp_name='hyper', n_epochs=1, num_layers=2, h_dim=50, activation='relu')
    # print(x)
    x = full_training(exp_name='hyper', n_epochs=1, num_layers=3, h_dim=50, activation='relu')
    print(x)
    x = full_training(exp_name='hyper', n_epochs=500, num_layers=4, h_dim=50, activation='relu')
    print(x)
    x = full_training(exp_name='hyper', n_epochs=500, num_layers=5, h_dim=50, activation='relu')
    print(x)
