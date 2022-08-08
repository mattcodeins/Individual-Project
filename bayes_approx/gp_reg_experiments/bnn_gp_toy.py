import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from datasets.gp_reg_dataset import gp_regression as d
from modules.bnn.modules.linear import make_linear_bnn
from modules.bnn.modules.loss import GaussianKLLoss, nELBO
from modules.bnn.utils import *


def full_training(experiment_name=None, n_epochs=10000, num_layers=2, h_dim=50,
                  prior_weight_std=1.0, prior_bias_std=1.0, init_std=1.0):
    torch.manual_seed(1)
    experiment_name = uniquify(experiment_name)

    # import dataset
    train_loader, test_loader, train, test, noise_std = d.create_regression_dataset()

    # create bnn
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    x_dim, y_dim = 1, 1
    layer_sizes = [x_dim] + [h_dim for _ in range(num_layers)] + [y_dim]
    activation = nn.ReLU()
    layer_kwargs = {'prior_weight_std': prior_weight_std,
                    'prior_bias_std': prior_bias_std,
                    'sqrt_width_scaling': True,
                    'init_std': init_std,
                    'device': device}
    model = make_linear_bnn(layer_sizes, activation, **layer_kwargs)
    log_noise_var = torch.ones(size=(), device=device)*-4.6  # Gaussian likelihood -4.6 == std 0.1
    print("BNN architecture: \n", model)

    d.plot_bnn_pred_post(model, predict, train, test, log_noise_var, 'BNN init (before training, MFVI)', device)

    # training hyperparameters
    learning_rate = 1e-2
    params = list(model.parameters())  # + [log_noise_var]
    opt = torch.optim.Adam(params, lr=learning_rate)
    lr_sch = torch.optim.lr_scheduler.StepLR(opt, 10000, gamma=0.1)

    gnll_loss = nn.GaussianNLLLoss(full=True, reduction='sum')
    kl_loss = GaussianKLLoss()
    nelbo = nELBO(nll_loss=gnll_loss, kl_loss=kl_loss)
    # nelbo = nELBO(nll_loss=gnll_loss, kl_loss=lambda a: torch.tensor([0]))

    logs = training_loop(
        model, n_epochs, opt, lr_sch, nelbo, train_loader, test_loader, log_noise_var,
        d.test_step, train, experiment_name, device
    )
    plot_training_loss(logs)

    d.plot_bnn_pred_post(model, predict, train, test, log_noise_var, 'BNN approx. posterior (MFVI)', device)

    return d.test_step(model, test_loader, train, predict)


if __name__ == "__main__":
    print(full_training(
        n_epochs=100000, num_layers=4, h_dim=50, prior_weight_std=1.0, prior_bias_std=1.0, init_std=0.1
    ))
