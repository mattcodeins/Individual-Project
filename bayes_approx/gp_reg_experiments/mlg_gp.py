import torch
import torch.nn as nn
import torchvision
import numpy as np
import logging
# import matplotlib.pyplot as plt

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from external.scalemarglik import marglik_optimization

from datasets.gp_reg_dataset import gp_regression as d

from laplace.curvature import BackPackGGN
from laplace import FullLaplace, KronLaplace


# construct a BNN
def make_linear_nn(layer_sizes, **layer_kwargs):
    # nonlinearity = getattr(nn, activation)() if isinstance(activation, str) else activation
    net = nn.Sequential(
        nn.Linear(layer_sizes[0], layer_sizes[1]),
        nn.ReLU(),
        nn.Linear(layer_sizes[1], layer_sizes[2]),
        nn.ReLU(),
        nn.Linear(layer_sizes[2], layer_sizes[2]),
        nn.ReLU(),
        nn.Linear(layer_sizes[2], layer_sizes[3]),
    )
    return net


def predict(lap, x_test, K):
    """
    Monte Carlo sampling of BNN using K samples.
    """
    f_mu, f_var = lap(x_test)
    f_mu = f_mu.squeeze().detach().cpu().numpy()
    f_sigma = f_var.squeeze().sqrt().cpu().numpy()
    # pred_std = np.sqrt(f_sigma**2 + lap.sigma_noise.item()**2)
    return f_mu, f_sigma


torch.manual_seed(1)
experiment_name = 'mlg_nn_gp_reg_30_07'

# import dataset
train_loader, test_loader, train, test, noise_std = d.create_regression_dataset()

# create bnn
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
x_dim, y_dim = 1, 1
h_dim = 100
layer_sizes = [x_dim, h_dim, h_dim, y_dim]
layer_kwargs = {'device': device}
model = make_linear_nn(layer_sizes, **layer_kwargs)
print("BNN architecture: \n", model)

# d.plot_bnn_pred_post(lap, predict, train, test, log_noise_var, noise_std,
#                      'BNN init (before training, MFVI)', device)

lap, model, margliks, losses = marglik_optimization(
    model, train_loader, likelihood='regression', sigma_noise_init=0.05, backend=BackPackGGN,
    laplace=KronLaplace, n_epochs=100
)

d.plot_bnn_pred_post(lap, predict, train, test, np.log(lap.sigma_noise), noise_std,
                     'BNN approx. posterior (MFVI)', device)

# print(margliks)
# print(losses)
