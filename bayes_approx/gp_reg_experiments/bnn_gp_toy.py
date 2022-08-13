import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

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


def full_training(experiment_name=None, n_epochs=10000,
                  num_layers=2, h_dim=50, activation='relu', init_std=0.05,
                  lik_std=0.05, prior_weight_std=1.0, prior_bias_std=1.0):
    torch.manual_seed(1)
    experiment_name = uniquify(experiment_name)

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
    layer_kwargs = {'prior_weight_std': prior_weight_std,
                    'prior_bias_std': prior_bias_std,
                    'sqrt_width_scaling': True,
                    'init_std': init_std,
                    'device': device}
    model = make_linear_bnn(layer_sizes, activation, **layer_kwargs)
    log_lik_var = torch.ones(size=(), device=device)*np.log(lik_std**2)  # Gaussian likelihood -4.6 == std 0.1
    print("BNN architecture: \n", model)

    # d.plot_bnn_prior(model)
    d.plot_bnn_pred_post(model, predict, train, test, log_lik_var, 'BNN init (before training, MFVI)', device)

    # training hyperparameters
    learning_rate = 1e-3
    params = list(model.parameters())  # + [log_noise_var]
    opt = torch.optim.Adam(params, lr=learning_rate)
    lr_sch = torch.optim.lr_scheduler.StepLR(opt, 10000, gamma=0.1)

    gnll_loss = nn.GaussianNLLLoss(full=True, reduction='sum')
    kl_loss = GaussianKLLoss()
    nelbo = nELBO(nll_loss=gnll_loss, kl_loss=kl_loss)
    # nelbo = nELBO(nll_loss=gnll_loss, kl_loss=lambda a: torch.tensor([0]))

    logs = training_loop(
        model, n_epochs, opt, lr_sch, nelbo, train_loader, test_loader, log_lik_var,
        d.test_step, train, experiment_name, device
    )
    # plot_training_loss(logs)

    d.plot_bnn_pred_post(model, predict, train, test, log_lik_var, 'BNN approx. posterior (MFVI)', device)

    return d.test_step(model, test_loader, train, predict)


def hyper_training_iter(train_loader, test_loader, train, test,
                        num_layers=2, h_dim=50, activation='relu', init_std=0.1,
                        likelihood_std=0.1, prior_weight_std=1.0, prior_bias_std=1.0):
    torch.manual_seed(1)

    n_epochs = 10000

    # create bnn
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    x_dim, y_dim = 1, 1
    layer_sizes = [x_dim] + [h_dim for _ in range(num_layers)] + [y_dim]
    if activation == 'relu':
        activation = nn.ReLU()
    elif activation == 'tanh':
        activation = nn.Tanh()
    layer_kwargs = {'prior_weight_std': prior_weight_std,
                    'prior_bias_std': prior_bias_std,
                    'sqrt_width_scaling': True,
                    'init_std': init_std,
                    'device': device}
    model = make_linear_bnn(layer_sizes, activation, **layer_kwargs)
    log_lik_var = torch.ones(size=(), device=device)*np.log(likelihood_std**2)  # Gaussian likelihood -4.6 == std 0.1
    # print("BNN architecture: \n", model)

    # d.plot_bnn_pred_post(model, predict, train, test, log_lik_var, 'BNN init (before training, MFVI)', device)

    # training hyperparameters
    learning_rate = 1e-3
    params = list(model.parameters())  # + [log_lik_var]
    opt = torch.optim.Adam(params, lr=learning_rate)
    lr_sch = torch.optim.lr_scheduler.StepLR(opt, 5000, gamma=0.1)

    gnll_loss = nn.GaussianNLLLoss(full=True, reduction='sum')
    kl_loss = GaussianKLLoss()
    nelbo = nELBO(nll_loss=gnll_loss, kl_loss=kl_loss)
    # nelbo = nELBO(nll_loss=gnll_loss, kl_loss=lambda a: torch.tensor([0]))

    logs = training_loop(
        model, n_epochs, opt, lr_sch, nelbo, train_loader, test_loader, log_lik_var,
        d.test_step, train, None, device
    )
    # plot_training_loss(logs)

    # d.plot_bnn_pred_post(model, predict, train, test, log_noise_var, 'BNN approx. posterior (MFVI)', device)

    return d.test_step(model, test_loader, train, predict)


def bnn_cross_val():
    n_splits = 5
    init_std_list = [0.02]
    lik_var_list = [0.02]
    num_layers_list = [2]
    prior_w_std_list = [2.0, 1.0]
    prior_b_std_list = [4.0, 6.0]
    kf = KFold(n_splits=n_splits, shuffle=True)

    (train_loader_list, val_loader_list, tesst_loader,
        normalised_train_list, val_list, test, noise_std) = d.create_regression_dataset_kf(kf)

    best_loss = float('inf')
    for init_std in init_std_list:
        for lik_var in lik_var_list:
            for num_layers in num_layers_list:
                for prior_w_std in prior_w_std_list:
                    for prior_b_std in prior_b_std_list:
                        t_val_loss = 0
                        print(f'Current Model: lv={lik_var}, nl={num_layers}, pws={prior_w_std}, pbs={prior_b_std}')
                        for i in range(n_splits):
                            t_val_loss += hyper_training_iter(
                                train_loader_list[i], val_loader_list[i],
                                normalised_train_list[i], val_list[i],
                                num_layers=num_layers, h_dim=50, activation='relu',
                                init_std=init_std, likelihood_std=lik_var,
                                prior_weight_std=prior_w_std, prior_bias_std=prior_b_std)/n_splits
                        print(f'CV Loss={t_val_loss}')
                        if t_val_loss < best_loss:
                            best_model = {'init_std': init_std, 'likelihood var': lik_var,
                                          'num_layers': num_layers,
                                          'prior w std': prior_w_std, 'prior b std': prior_b_std}
                            best_loss = t_val_loss
                        print(f'Best CV Loss:{best_loss}. Best Model:{best_model}')
                        with open('bayes_approx/results/gp_bnn/auto_cross_val.txt', 'a') as f:
                            f.write(f'{init_std} {lik_var} {num_layers} {prior_w_std} {prior_b_std} {t_val_loss} \n')


if __name__ == "__main__":
    # print(full_training(
    #     experiment_name="bnn_2l3pstd0.04initstd", n_epochs=40000,
    #     num_layers=4, h_dim=50, prior_weight_std=8.0, prior_bias_std=8.0, init_std=0.03,
    # ))
    # bnn_cross_val()
    full_training(experiment_name=None, n_epochs=50000,
                  num_layers=4, h_dim=50, activation='relu', init_std=0.05,
                  lik_std=0.02, prior_weight_std=10.0, prior_bias_std=1.0)
