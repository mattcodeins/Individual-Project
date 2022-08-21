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

from modules.bnn.modules.cmv_linear import make_linear_cmv_bnn
from modules.bnn.modules.loss import CollapsedMeanVarLoss, nELBO
from modules.bnn.utils import *

from datasets.gp_reg_dataset import gp_regression as d


def full_training(exp_name=None, n_epochs=10000,
                  num_layers=2, h_dim=50, activation='relu', init_std=0.05,
                  init_lik_std=0.05, alpha=0.5, beta=0.01, delta=0.1):
    torch.manual_seed(1)
    if exp_name == 'hyper':
        exp_name = (f'cmvBNN_GPtoyreg_nl{num_layers}_ils{init_lik_std}_ia{alpha}'
                    + f'_ib{beta}_id{delta}')
    exp_name = uniquify(exp_name)

    # import dataset
    train_loader, test_loader, train, test, noise_std = d.create_regression_dataset()

    # create bnn
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    x_dim, y_dim = 1, 1
    h_dim = 50
    layer_sizes = [x_dim] + [h_dim for _ in range(num_layers)] + [y_dim]
    if activation == 'relu':
        activation = nn.ReLU()
    elif activation == 'tanh':
        activation = nn.Tanh()
    layer_kwargs = {'sqrt_width_scaling': True,
                    'init_std': init_std,
                    'device': device}
    model = make_linear_cmv_bnn(layer_sizes, alpha, beta, delta, activation, **layer_kwargs)
    if init_lik_std is None:
        log_lik_var = torch.ones(size=(), device=device)*np.log(0.05**2)
    elif torch.cuda.is_available():
        normal_lik_std = torch.cuda.FloatTensor(d.normalise_data(init_lik_std, 0, train.y_std))
        log_lik_var = nn.Parameter(torch.ones(size=(), device=device)*torch.log(normal_lik_std**2))
    else:
        normal_lik_std = d.normalise_data(init_lik_std, 0, train.y_std)
        log_lik_var = nn.Parameter(torch.ones(size=(), device=device)*np.log(normal_lik_std**2))
    print("BNN architecture: \n", model)

    # d.plot_bnn_pred_post(model, predict, train, test, log_lik_var, 'cmvBNN initialisation', device)

    # training hyperparameters
    learning_rate = 1e-3
    params = list(model.parameters()) + [log_lik_var]
    opt = torch.optim.Adam(params, lr=learning_rate)
    lr_sch = torch.optim.lr_scheduler.StepLR(opt, n_epochs/3, gamma=0.1)

    gnll_loss = nn.GaussianNLLLoss(full=True, reduction='sum')
    cmv_loss = CollapsedMeanVarLoss()
    ncelbo = nELBO(nll_loss=gnll_loss, kl_loss=cmv_loss)

    logs = training_loop(
        model, n_epochs, opt, lr_sch, ncelbo, train_loader, test_loader, log_lik_var,
        d.mse_test_step, train, exp_name, device
    )

    # plot_training_loss_together(logs, exp_name=exp_name)

    # d.plot_bnn_pred_post(model, predict, train, test, log_lik_var, 'cmvBNN approximate posterior', device)

    return d.mse_test_step(model, test_loader, train, predict), logs[-1][1]


def load_test_model(exp_name):
    model = load_model(model, exp_name)
    # test_step(model, nelbo, test_loader, predict=predict, device=device)
    print(model._hyperprior_alpha_param)
    print(model._hyperprior_beta_param)
    print(model._hyperprior_t_param)


if __name__ == "__main__":
    # full_training(exp_name='hyper', n_epochs=30000,
    #               num_layers=3, h_dim=50, activation='relu', init_std=0.05,
    #               init_lik_std=None, alpha=0.5, beta=0.01, delta=0.1)
    full_training(exp_name='hyper', n_epochs=30000,
                  num_layers=1, h_dim=50, activation='relu', init_std=0.05,
                  init_lik_std=None, alpha=0.5, beta=0.01, delta=0.1)
    full_training(exp_name='hyper', n_epochs=30000,
                  num_layers=2, h_dim=50, activation='relu', init_std=0.05,
                  init_lik_std=None, alpha=0.5, beta=0.01, delta=0.1)
    full_training(exp_name='hyper', n_epochs=30000,
                  num_layers=4, h_dim=50, activation='relu', init_std=0.05,
                  init_lik_std=None, alpha=0.5, beta=0.01, delta=0.1)
    full_training(exp_name='hyper', n_epochs=30000,
                  num_layers=5, h_dim=50, activation='relu', init_std=0.05,
                  init_lik_std=None, alpha=0.5, beta=0.01, delta=0.1)
