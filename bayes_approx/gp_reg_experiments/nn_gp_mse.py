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


def FULL_TRAINING(N_EPOCHS, NUM_LAYERS, WEIGHT_DECAY):
    torch.manual_seed(1)

    # import dataset
    train_loader, test_loader, train, test, noise_std = d.create_regression_dataset()

    # create bnn
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    x_dim, y_dim = 1, 1
    h_dim = 50
    layer_sizes = [x_dim] + [h_dim for _ in range(NUM_LAYERS)] + [y_dim]
    activation = nn.ReLU()
    layer_kwargs = {'prior_weight_std': 1.0,
                    'prior_bias_std': 1.0,
                    'sqrt_width_scaling': False,
                    'init_std': 0,
                    'device': device}
    model = make_linear_nn(layer_sizes, activation=activation, **layer_kwargs)
    log_noise_var = torch.ones(size=(), device=device)*-999  # Equivalent to std 0.05
    print("BNN architecture: \n", model)

    # d.plot_bnn_pred_post(model, predict, train, test, log_noise_var, noise_std,
    #                      'FFNN init (before training, 2 hidden layers)', device)

    # training hyperparameters
    learning_rate = 1e-3
    params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=learning_rate, weight_decay=WEIGHT_DECAY)
    # lr_sch = torch.optim.lr_scheduler.StepLR(opt, 4000, gamma=0.1)

    mse = nn.MSELoss()

    # training loop
    model.train()
    logs = []
    for i in range(N_EPOCHS):
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
    plt.title('training loss (3 hidden layers)')
    plt.legend()
    plt.ylim(0, 0.6)
    plt.show()
    # plt.savefig('.png')

    d.plot_bnn_pred_post(model, predict, train, test, log_noise_var, 'FFNN prediction function (3 hidden layers)')

    d.test_step(model, test_loader, train, predict)
