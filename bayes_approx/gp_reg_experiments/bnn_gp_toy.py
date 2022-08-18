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


def full_training(exp_name=None, n_epochs=10000,
                  num_layers=2, h_dim=50, activation='relu', init_std=0.05,
                  likelihood_std=0.05, prior_weight_std=1.0, prior_bias_std=1.0):
    torch.manual_seed(1)
    if exp_name == 'hyper':
        exp_name = (f'nl{num_layers}_hdim{h_dim}_likstd{likelihood_std}_pws{prior_weight_std}_pbs{prior_bias_std}'
                    + '_BNN_GPtoyreg')
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
    layer_kwargs = {'prior_weight_std': prior_weight_std,
                    'prior_bias_std': prior_bias_std,
                    'sqrt_width_scaling': True,
                    'init_std': init_std,
                    'device': device}
    model = make_linear_bnn(layer_sizes, activation, **layer_kwargs)
    normal_lik_std = torch.cuda.FloatTensor(d.normalise_data(likelihood_std, 0, train.y_std))
    print(normal_lik_std)
    log_lik_var = torch.ones(size=(), device=device)*torch.log(normal_lik_std**2)
    print("BNN architecture: \n", model)

    d.plot_bnn_pred_post(model, predict, train, test, log_lik_var, 'BNN initialisation (MVFI)', device)

    # training hyperparameters
    learning_rate = 1e-3
    params = list(model.parameters())  # + [log_lik_var]
    opt = torch.optim.Adam(params, lr=learning_rate)
    lr_sch = torch.optim.lr_scheduler.StepLR(opt, n_epochs/4, gamma=0.1)

    gnll_loss = nn.GaussianNLLLoss(full=True, reduction='sum')
    kl_loss = GaussianKLLoss()
    nelbo = nELBO(nll_loss=gnll_loss, kl_loss=kl_loss)
    # nelbo = nELBO(nll_loss=gnll_loss, kl_loss=lambda a: torch.tensor([0]))

    logs = training_loop(
        model, n_epochs, opt, lr_sch, nelbo, train_loader, test_loader, log_lik_var,
        d.mse_test_step, train, exp_name, device
    )
    # plot_training_loss(logs)

    # d.plot_bnn_pred_post(model, predict, train, test, log_lik_var, 'BNN approx. posterior (MFVI)', device)

    return d.mse_test_step(model, test_loader, train, predict), logs[-1][1]


def hyper_training_iter(train_loader, test_loader, train, test,
                        num_layers=2, h_dim=50, activation='relu', init_std=0.1,
                        likelihood_std=0.1, prior_weight_std=1.0, prior_bias_std=1.0):
    n_epochs = 16000

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
    normal_lik_std = torch.cuda.FloatTensor(d.normalise_data(likelihood_std, 0, train.y_std))
    log_lik_var = torch.ones(size=(), device=device)*torch.log(normal_lik_std**2)  # Gaussian likelihood -4.6 == std 0.1
    # print("BNN architecture: \n", model)

    # d.plot_bnn_pred_post(model, predict, train, test, log_lik_var, 'BNN init (before training, MFVI)', device)

    # training hyperparameters
    learning_rate = 1e-3
    params = list(model.parameters())  # + [log_lik_var]
    opt = torch.optim.Adam(params, lr=learning_rate)
    lr_sch = torch.optim.lr_scheduler.StepLR(opt, 7000, gamma=0.15)

    gnll_loss = nn.GaussianNLLLoss(full=True, reduction='sum')
    kl_loss = GaussianKLLoss()
    nelbo = nELBO(nll_loss=gnll_loss, kl_loss=kl_loss)
    # nelbo = nELBO(nll_loss=gnll_loss, kl_loss=lambda a: torch.tensor([0]))

    logs = training_loop(
        model, n_epochs, opt, lr_sch, nelbo, train_loader, test_loader, log_lik_var,
        d.mse_test_step, train, None, device
    )
    # plot_training_loss(logs)

    # d.plot_bnn_pred_post(model, predict, train, test, log_lik_var, 'BNN approx. posterior (MFVI)', device)

    return d.mse_test_step(model, test_loader, train, predict, log_lik_var), logs[-1][1]


def bnn_cross_val():
    init_std_list = [0.05, 'prior']
    lik_var_list = [0.02, 0.01]
    num_layers_list = [5, 4]
    prior_std_list = [(0.5, 0.5), (0.5, 2.0), (5.0, 10.0)]

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True)

    (train_loader_list, val_loader_list, tesst_loader,
        normalised_train_list, val_list, test, noise_std) = d.create_regression_dataset_kf(kf)

    completed_cvs = 0

    best_loss = float('inf')
    for init_std in init_std_list:
        for lik_var in lik_var_list:
            for num_layers in num_layers_list:
                for p_w_std, p_b_std in prior_std_list:
                    if completed_cvs > 0:
                        completed_cvs -= 1
                        continue
                    t_val_loss = 0
                    t_elbo = 0
                    print(f'Current Model: lv={lik_var}, nl={num_layers}, pws={p_w_std}, pbs={p_b_std}')
                    for i in range(n_splits):
                        val_loss, elbo = hyper_training_iter(
                            train_loader_list[i], val_loader_list[i],
                            normalised_train_list[i], val_list[i],
                            num_layers=num_layers, h_dim=50, activation='relu',
                            init_std=init_std, likelihood_std=lik_var,
                            prior_weight_std=p_w_std, prior_bias_std=p_b_std
                        )
                        t_val_loss += val_loss/n_splits
                        t_elbo += elbo/n_splits
                    print(f'CV Loss={t_val_loss}')
                    if t_val_loss < best_loss:
                        best_model = {'init_std': init_std, 'likelihood var': lik_var,
                                      'num_layers': num_layers,
                                      'prior w std': p_w_std, 'prior b std': p_b_std}
                        best_loss = t_val_loss
                    print(f'Best CV Loss:{best_loss}. Best Model:{best_model}')
                    with open('bayes_approx/results/gp_bnn/new_auto_cv.txt', 'a') as f:
                        f.write(f'{init_std} {lik_var} {num_layers} {p_w_std} {p_b_std} {t_val_loss} {t_elbo} \n')


def load_test_model(exp_name=None, n_epochs=None,
                    num_layers=2, h_dim=50, activation='relu', init_std=0.1,
                    likelihood_std=0.1, prior_weight_std=1.0, prior_bias_std=1.0):
    # create bnn
    train_loader, test_loader, train, test, noise_std = d.create_regression_dataset()

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
    normal_lik_std = torch.cuda.FloatTensor(d.normalise_data(likelihood_std, 0, train.y_std))
    log_lik_var = torch.ones(size=(), device=device)*torch.log(normal_lik_std**2)  # Gaussian likelihood -4.6 == std 0.1
    print("BNN architecture: \n", model)
    model = load_model(model, exp_name)
    logs = load_logs(exp_name)
    print(logs[-1])
    plot_training_loss_together(logs, exp_name=exp_name)
    d.plot_bnn_pred_post(model, predict, train, test, log_lik_var, None, exp_name, device)
    d.mse_test_step(model, test_loader, train, predict, log_lik_var)
    d.gnll_test_step(model, test_loader, train, predict, log_lik_var)
    d.mse_train(model, train_loader, predict, log_lik_var)
    d.gnll_train(model, train_loader, predict, log_lik_var)


if __name__ == "__main__":
    # v1, e1 = full_training(exp_name='hyper', n_epochs=60000,
    #                        num_layers=4, h_dim=50, activation='relu', init_std=0.05,
    #                        likelihood_std=0.02, prior_weight_std=0.5, prior_bias_std=0.5)
    # load_test_model(exp_name='nl4_hdim50_likstd0.02_pws1.0_pbs1.0_BNN_GPtoyreg', n_epochs=60000,
    #                 num_layers=4, h_dim=50, activation='relu', init_std=0.05,
    #                 likelihood_std=0.02, prior_weight_std=1.0, prior_bias_std=1.0)


    # v2, e2 = full_training(experiment_name='hyper', n_epochs=60000,
    #                        num_layers=2, h_dim=50, activation='relu', init_std=0.05,
    #                        likelihood_std=0.05, prior_weight_std=1.0, prior_bias_std=1.0)
    # v3, e3 = full_training(experiment_name='hyper', n_epochs=60000,
    #                        num_layers=3, h_dim=50, activation='relu', init_std=0.05,
    #                        likelihood_std=0.05, prior_weight_std=1.0, prior_bias_std=1.0)
    # v4, e4 = full_training(experiment_name='hyper', n_epochs=60000,
    #                        num_layers=4, h_dim=50, activation='relu', init_std=0.05,
    #                        likelihood_std=0.05, prior_weight_std=1.0, prior_bias_std=1.0)
    # v5, e5 = full_training(experiment_name='hyper', n_epochs=60000,
    #                        num_layers=5, h_dim=50, activation='relu', init_std=0.05,
    #                        likelihood_std=0.05, prior_weight_std=1.0, prior_bias_std=1.0)
    # print(f'{v1}, {e1}')
    # print(f'{v2}, {e2}')
    # print(f'{v3}, {e3}')
    # print(f'{v4}, {e4}')
    # print(f'{v5}, {e5}')

    # bnn_cross_val()
    # best_nll_model = [0.02, 0.005, 2, 2.0, 5.0]
    # full_training(experiment_name='orgcv', n_epochs=100000,
    #               num_layers=2, h_dim=50, activation='relu', init_std=0.02,
    #               likelihood_std=0.005, prior_weight_std=2.0, prior_bias_std=5.0)
    # best_mse_model = [0.02, 0.02, 2, 2.0, 5.0]
    # full_training(experiment_name='orgcv', n_epochs=100000,
    #               num_layers=2, h_dim=50, activation='relu', init_std=0.02,
    #               likelihood_std=0.02, prior_weight_std=2.0, prior_bias_std=5.0)
    # load_test_model('cv_mse', n_epochs=100000,
    #                 num_layers=2, h_dim=50, activation='relu', init_std=0.02,
    #                 likelihood_std=0.02, prior_weight_std=2.0, prior_bias_std=5.0)
    # ('cv_nll', n_epochs=100000,
    #                 num_layers=2, h_dim=50, activation='relu', init_std=0.02,
    #                 likelihood_std=0.005, prior_weight_std=2.0, prior_bias_std=5.0)

    load_test_model(exp_name='bnn_vague_0.05priorNL1', n_epochs=60000,
                  num_layers=1, h_dim=50, activation='relu', init_std=0.05,
                  likelihood_std=0.05, prior_weight_std=1.0, prior_bias_std=1.0)
    load_test_model(exp_name='bnn_vague_0.05priorNL2', n_epochs=60000,
                  num_layers=2, h_dim=50, activation='relu', init_std=0.05,
                  likelihood_std=0.05, prior_weight_std=1.0, prior_bias_std=1.0)
    load_test_model(exp_name='bnn_vague_0.05priorNL3', n_epochs=60000,
                  num_layers=3, h_dim=50, activation='relu', init_std=0.05,
                  likelihood_std=0.05, prior_weight_std=1.0, prior_bias_std=1.0)
    load_test_model(exp_name='bnn_vague_0.05priorNL4', n_epochs=60000,
                  num_layers=4, h_dim=50, activation='relu', init_std=0.05,
                  likelihood_std=0.05, prior_weight_std=1.0, prior_bias_std=1.0)
    load_test_model(exp_name='bnn_vague_0.05priorNL5', n_epochs=60000,
                  num_layers=5, h_dim=50, activation='relu', init_std=0.05,
                  likelihood_std=0.05, prior_weight_std=1.0, prior_bias_std=1.0)
