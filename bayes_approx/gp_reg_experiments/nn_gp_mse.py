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
from modules.bnn.modules.nnlinear import make_linear_nn
from modules.bnn.utils import *

# OPTIMISE = True


def train_step(model, opt, nll, dataloader, device):
    # global OPTIMISE
    tloss = 0
    for i, (x, y) in enumerate(dataloader):
        opt.zero_grad()
        x = x.to(device); y = y.to(device)
        y_pred = model(x)
        loss = nll(y_pred, y)
        loss.backward()
        opt.step()
        tloss += loss
    return tloss


def full_training(num_layers=2, h_dim=50, weight_decay=0):
    torch.manual_seed(1)

    # import dataset
    train_loader, test_loader, train, test, noise_std = d.create_regression_dataset()

    # create bnn
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    x_dim, y_dim = 1, 1
    layer_sizes = [x_dim] + [h_dim for _ in range(num_layers)] + [y_dim]
    activation = nn.ReLU()
    layer_kwargs = {'prior_weight_std': 1.0,
                    'prior_bias_std': 1.0,
                    'sqrt_width_scaling': False,
                    'init_std': 0,
                    'device': device}
    model = make_linear_nn(layer_sizes, activation=activation, **layer_kwargs)
    print("BNN architecture: \n", model)

    log_noise_var = nn.Parameter(torch.ones(size=(), device=device)*-9999)  # Equivalent to std 0.05
    d.plot_bnn_pred_post(model, predict, train, test, log_noise_var,
                         'FFNN init (before training, 2 hidden layers)', device)

    # training hyperparameters
    learning_rate = 1e-3
    params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    # lr_sch = torch.optim.lr_scheduler.StepLR(opt, 4000, gamma=0.1)

    mse = nn.MSELoss()

    # training loop
    model.train()
    logs = []
    for i in range(60000):
        loss = train_step(
            model, opt, mse, train_loader, device=device
        )
        if (i+1) % 100 == 0:
            with torch.no_grad():
                test_mse = d.test_step(model, test_loader, train, predict)
                logs.append([to_numpy(loss)] + [to_numpy(test_mse)])
                print("Epoch {}, nll={}, test_mse={}".format(i+1, logs[-1][0], logs[-1][1]))

    logs = np.array(logs)

    # plot the training curve
    plt.plot(np.arange(logs.shape[0]), logs[:, 0], label='mse on train')
    plt.plot(np.arange(logs.shape[0]), logs[:, 1], label='mse on test')
    plt.xlabel('epoch')
    plt.title(f'training loss ({num_layers} hidden layers)')
    plt.legend()
    plt.ylim(0, 0.6)
    plt.show()
    # plt.savefig('.png')
    d.plot_bnn_pred_post(model, predict, train, test, log_noise_var,
                         f'FFNN prediction function ({num_layers} hidden layers)')

    d.test_step(model, test_loader, train, predict)


def hyper_training_iter(train_loader, test_loader, train, test, num_layers, height, weight_decay):
    # torch.manual_seed(1)

    # create bnn
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    x_dim, y_dim = 1, 1
    layer_sizes = [x_dim] + [height for _ in range(num_layers)] + [y_dim]
    activation = nn.ReLU()
    layer_kwargs = {'prior_weight_std': 1.0,
                    'prior_bias_std': 1.0,
                    'sqrt_width_scaling': False,
                    'init_std': 0,
                    'device': device}
    model = make_linear_nn(layer_sizes, activation=activation, **layer_kwargs)
    # print("BNN architecture: \n", model)

    # d.plot_bnn_pred_post(model, predict, train, test, log_noise_var, noise_std,
    #                      'FFNN init (before training, 2 hidden layers)', device)

    # training hyperparameters
    learning_rate = 1e-3
    params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    # lr_sch = torch.optim.lr_scheduler.StepLR(opt, 4000, gamma=0.1)

    mse = nn.MSELoss()

    # training loop
    model.train()
    logs = []
    losses = []
    for i in range(30000):
        losses.append(float(train_step(
            model, opt, mse, train_loader, device=device))
        )
        if (i+1) % 5000 == 0:
            with torch.no_grad():
                train_loss_avg = sum(losses[-1000:])/1000
                test_mse = d.test_step(model, test_loader, train, predict)
                logs.append([train_loss_avg] + [to_numpy(test_mse)])
                print("Epoch {}, train_mse={}, test_mse={}".format(i+1, logs[-1][0], logs[-1][1]))

    logs = np.array(logs)

    # # plot the training curve
    # plt.plot(np.arange(logs.shape[0]), logs[:, 0], label='mse on train')
    # plt.plot(np.arange(logs.shape[0]), logs[:, 1], label='mse on test')
    # plt.xlabel('epoch')
    # plt.title(f'training loss ({num_layers} hidden layers)')
    # plt.legend()

    # plt.ylim(0, 0.6)
    # plt.show()
    # # plt.savefig('.png')

    # d.plot_bnn_pred_post(model, predict, train, test, log_noise_var,
    #                      f'FFNN prediction function ({num_layers} hidden layers)')

    return d.test_step(model, test_loader, train, predict)


def nn_cross_val():
    n_splits = 5
    weight_decay_list = [1e-4, 1e-6]
    num_layers_list = [1, 4]
    height_list = [50]
    kf = KFold(n_splits=n_splits, shuffle=True)

    (train_loader_list, val_loader_list, test_loader,
        normalised_train_list, val_list, test, noise_std) = d.create_regression_dataset_kf(kf)

    best_wd_loss = best_nl_loss = best_h_loss = best_loss = float('inf')
    for weight_decay in weight_decay_list:
        for num_layers in num_layers_list:
            for height in height_list:
                t_val_loss = 0
                for i in range(n_splits):
                    t_val_loss += hyper_training_iter(train_loader_list[i], val_loader_list[i],
                                                      normalised_train_list[i], val_list[i],
                                                      num_layers, height, weight_decay)/n_splits
                print(f'Current Model CV Result: wd={weight_decay}, nl={num_layers}, h={height}, cv_loss={t_val_loss}')
                if t_val_loss < best_wd_loss:
                    best_wd = weight_decay
                    best_wd_loss = t_val_loss
                if t_val_loss < best_nl_loss:
                    best_nl = num_layers
                    best_nl_loss = t_val_loss
                if t_val_loss < best_h_loss:
                    best_h = height
                    best_h_loss = t_val_loss
                if t_val_loss < best_loss:
                    best_model = {'weight_decay': weight_decay, 'num_layers': num_layers, 'height': height}
                    best_loss = t_val_loss
                print(f'Best CV Loss:{best_loss}. Best Model:{best_model}')
    print(f'best weight decay: {best_wd}')
    print(f'best num of layers: {best_nl}')
    print(f'best height: {best_h}')


if __name__ == '__main__':
    # full_training(num_layers=1, h_dim=50, weight_decay=1e-6)
    nn_cross_val()
    # nn_cross_val()
