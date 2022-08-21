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

from modules.bnn.modules.cm_linear import make_linear_cm_bnn
from modules.bnn.modules.loss import CollapsedMeanLoss, nELBO
from modules.bnn.utils import *

from datasets.gp_reg_dataset import gp_regression as d


def full_training(exp_name=None, n_epochs=10000, num_layers=2, h_dim=50, activation='relu',
                  init_std=0.05, hyperprior_learnable=False,
                  lik_std=0.05, init_prior_std=1.0, init_prior_hyperstd=0.5):
    torch.manual_seed(1)
    if exp_name == 'hyper':
        exp_name = (f'cmBNN_GPtoyreg_nl{num_layers}_ls{lik_std}_ips{init_prior_std}'
                    + f'_hl{hyperprior_learnable}_ipmhs{init_prior_hyperstd}')
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
    layer_kwargs = {'prior_weight_std': init_prior_std,
                    'prior_bias_std': init_prior_std,
                    'sqrt_width_scaling': True,
                    'init_std': init_std,
                    'device': device}
    model = make_linear_cm_bnn(
        layer_sizes, init_prior_hyperstd, hyperprior_learnable, activation, **layer_kwargs
    )
    if lik_std is not None:
        log_lik_var = torch.ones(size=(), device=device)*np.log(lik_std**2)
    elif torch.cuda.is_available():
        normal_lik_std = torch.cuda.FloatTensor(d.normalise_data(0.05, 0, train.y_std))
        log_lik_var = nn.Parameter(torch.ones(size=(), device=device)*torch.log(normal_lik_std**2))
    else:
        normal_lik_std = d.normalise_data(0.05, 0, train.y_std)
        log_lik_var = nn.Parameter(torch.ones(size=(), device=device)*np.log(normal_lik_std**2))
    print("BNN architecture: \n", model)

    # d.plot_bnn_pred_post(model, predict, train, test, log_lik_var, 'cmBNN initialisation', device)

    # training hyperparameters
    learning_rate = 1e-3
    params = list(model.parameters()) + [log_lik_var]
    opt = torch.optim.Adam(params, lr=learning_rate)
    lr_sch = torch.optim.lr_scheduler.StepLR(opt, n_epochs/2, gamma=0.1)

    gnll_loss = nn.GaussianNLLLoss(full=True, reduction='sum')
    cm_loss = CollapsedMeanLoss()
    ncelbo = nELBO(nll_loss=gnll_loss, kl_loss=cm_loss)

    logs = training_loop(
        model, n_epochs, opt, lr_sch, ncelbo, train_loader, test_loader, log_lik_var,
        d.mse_test_step, train, exp_name, device
    )

    # plot_training_loss_together(logs, exp_name=exp_name)

    # d.plot_bnn_pred_post(model, predict, train, test, log_lik_var, 'cmBNN approximate posterior', device)

    return d.mse_test_step(model, test_loader, train, predict), logs[-1][1]


def load_test_model(exp_name=None, n_epochs=10000, num_layers=2, h_dim=50, activation='relu',
                    init_std=0.05, hyperprior_learnable=False,
                    init_lik_std=0.05, init_prior_std=1.0, init_prior_hyperstd=0.05):
    # create bnn
    train_loader, test_loader, train, test, noise_std = d.create_regression_dataset()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    x_dim, y_dim = 1, 1
    h_dim = 50
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
    model = make_linear_cm_bnn(
        layer_sizes, init_prior_hyperstd, hyperprior_learnable, activation, **layer_kwargs
    )
    if init_lik_std is None:
        log_lik_var = torch.ones(size=(), device=device)*np.log(0.05**2)
    elif torch.cuda.is_available():
        normal_lik_std = torch.cuda.FloatTensor(d.normalise_data(init_lik_std, 0, train.y_std))
        log_lik_var = nn.Parameter(torch.ones(size=(), device=device)*torch.log(normal_lik_std**2))
    else:
        normal_lik_std = d.normalise_data(init_lik_std, 0, train.y_std)
        log_lik_var = nn.Parameter(torch.ones(size=(), device=device)*np.log(normal_lik_std**2))
    print("BNN architecture: \n", model)

    model = load_model(model, exp_name)
    logs = load_logs(exp_name)
    plot_training_loss_together(logs, exp_name=exp_name)
    d.plot_bnn_pred_post(model, predict, train, test, log_lik_var, None, exp_name, device)
    d.mse_test_step(model, test_loader, train, predict, log_lik_var)
    d.gnll_test_step(model, test_loader, train, predict, log_lik_var)
    d.mse_train(model, train_loader, predict, log_lik_var)
    d.gnll_train(model, train_loader, predict, log_lik_var)


if __name__ == "__main__":
    # full_training(exp_name='hyper', n_epochs=30000, num_layers=3,
    #               init_std=0.2, hyperprior_learnable=True,
    #               lik_std=0.05, init_prior_std=1.0, init_prior_hyperstd=0.5)
    # full_training(exp_name='hyper', n_epochs=30000, num_layers=3,
    #               init_std=0.05, hyperprior_learnable=False,
    #               lik_std=0.05, init_prior_std=1.0, init_prior_hyperstd=0.5)
    # full_training(exp_name='hyper', n_epochs=30000, num_layers=4,
    #               init_std=0.05, hyperprior_learnable=False,
    #               lik_std=0.05, init_prior_std=1.0, init_prior_hyperstd=0.5)
    # full_training(exp_name='hyper', n_epochs=30000, num_layers=2,
    #               init_std=0.05, hyperprior_learnable=False,
    #               lik_std=0.05, init_prior_std=1.0, init_prior_hyperstd=0.5)
    # full_training(exp_name='hyper', n_epochs=30000, num_layers=4,
    #               init_std=0.05, hyperprior_learnable=True,
    #               lik_std=0.05, init_prior_std=1.0, init_prior_hyperstd=0.5)
    # full_training(exp_name='hyper', n_epochs=30000, num_layers=2,
    #               init_std=0.05, hyperprior_learnable=True,
    #               lik_std=0.05, init_prior_std=1.0, init_prior_hyperstd=0.5)
    # full_training(exp_name='hyper', n_epochs=30000, num_layers=3,
    #               init_std=0.05, hyperprior_learnable=True,
    #               lik_std=None, init_prior_std=1.0, init_prior_hyperstd=0.5)
    # full_training(exp_name='hyper', n_epochs=30000, num_layers=3,
    #               init_std=0.05, hyperprior_learnable=False,
    #               lik_std=0.05, init_prior_std=1.0, init_prior_hyperstd=2.0)
    # full_training(exp_name='hyper', n_epochs=30000, num_layers=4,
    #               init_std=0.05, hyperprior_learnable=False,
    #               lik_std=0.05, init_prior_std=1.0, init_prior_hyperstd=2.0)
    # full_training(exp_name='hyper', n_epochs=30000, num_layers=4,
    #               init_std=0.05, hyperprior_learnable=True,
    #               lik_std=0.05, init_prior_std=1.0, init_prior_hyperstd=2.0)
    # full_training(exp_name='hyper', n_epochs=30000, num_layers=1,
    #               init_std=0.2, hyperprior_learnable=True,
    #               lik_std=0.05, init_prior_std=1.0, init_prior_hyperstd=0.5)
    full_training(exp_name='hyper', n_epochs=30000, num_layers=1,
                  init_std=0.05, hyperprior_learnable=False,
                  lik_std=0.05, init_prior_std=1.0, init_prior_hyperstd=0.5)