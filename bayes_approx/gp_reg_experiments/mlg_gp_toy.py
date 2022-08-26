import torch
import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt

import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from modules.bnn.modules.mlg_linear import make_mlg_linear_bnn
from modules.bnn.modules.loss import GaussianKLLoss, nELBO
from modules.bnn.utils import *

from datasets.gp_reg_dataset import gp_regression as d


def full_training(exp_name=None, n_epochs=10000, num_layers=2, h_dim=50, activation='relu',
                  init_std=0.1, lik_std=0.1, init_prior_std=1.0):
    torch.manual_seed(1)
    if exp_name == 'hyper':
        exp_name = (f'mlgBNN_GPtoyreg_nl{num_layers}_ls{lik_std}')
    exp_name = uniquify(exp_name)

    # import dataset
    train_loader, test_loader, train, test, noise_std = d.create_regression_dataset()

    # create bnn
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    x_dim, y_dim = 1, 1
    layer_sizes = [x_dim] + [h_dim for _ in range(num_layers)] + [y_dim]
    if activation == 'relu':
        activation = nn.ReLU()
    elif activation == 'tanh':
        activation = nn.Tanh()
    layer_kwargs = {'prior_weight_std': init_prior_std,
                    'prior_bias_std': init_prior_std,
                    'sqrt_width_scaling': True,
                    'init_std': init_std,
                    'device': device}
    model = make_mlg_linear_bnn(layer_sizes, activation=activation, **layer_kwargs)
    if lik_std is not None:
        log_lik_var = torch.ones(size=(), device=device)*np.log(lik_std**2)
    # elif torch.cuda.is_available():
    #     normal_lik_std = torch.cuda.FloatTensor(d.normalise_data(0.05, 0, train.y_std))
    #     log_lik_var = nn.Parameter(torch.ones(size=(), device=device)*torch.log(normal_lik_std**2))
    else:
        normal_lik_std = d.normalise_data(0.05, 0, train.y_std)
        log_lik_var = nn.Parameter(torch.ones(size=(), device=device)*np.log(normal_lik_std**2))
    print("BNN architecture: \n", model)

    d.plot_bnn_pred_post(model, predict, train, test, log_lik_var,
                         'BNN init (before training, MFVI)', device)

    # training hyperparameters
    learning_rate = 1e-3
    params = list(model.parameters()) + [log_lik_var]
    opt = torch.optim.Adam(params, lr=learning_rate)
    lr_sch = torch.optim.lr_scheduler.StepLR(opt, n_epochs/2, gamma=0.1)

    # define loss function (-ELBO)
    gnll_loss = nn.GaussianNLLLoss(full=True, reduction='sum')
    kl_loss = GaussianKLLoss()
    nelbo = nELBO(nll_loss=gnll_loss, kl_loss=kl_loss)

    logs = training_loop(
        model, n_epochs, opt, lr_sch, nelbo, train_loader, test_loader, log_lik_var,
        d.mse_test_step, train, exp_name, device
    )
    plot_training_loss(logs)

    d.plot_bnn_pred_post(model, predict, train, test, log_lik_var,
                         'BNN approx. posterior (MFVI)', device)


if __name__ == '__main__':
    full_training(exp_name='hyper', n_epochs=10000, num_layers=3, h_dim=50, activation='relu',
                  init_std=0.1, lik_std=0.1, init_prior_std=0.1)
