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


def train_step(model, opt, nll, dataloader, log_noise_var, device):
    # global OPTIMISE
    tloss = 0
    for i, (x, y) in enumerate(dataloader):
        opt.zero_grad()
        noise_var = torch.exp(log_noise_var)*torch.ones(x.shape[0])
        x = x.to(device); y = y.to(device)
        y_pred = model(x)
        loss = nll(y_pred, y, noise_var)
        loss.backward()
        opt.step()
        tloss += loss
    return tloss


def predict(bnn, x_test, K=1):  # Monte Carlo sampling using K samples
    y_pred = []
    for _ in range(K):
        y_pred.append(bnn(x_test))
    # shape (K, batch_size, y_dim) or (batch_size, y_dim) if K = 1
    y_pred = torch.stack(y_pred, dim=0).squeeze(0)
    return y_pred.mean(0), y_pred.std(0)


if __name__ == "__main__":
    torch.manual_seed(1)

    # import dataset
    train_loader, test_loader, normal_train, test, noise_std = d.create_regression_dataset()

    # create bnn
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    x_dim, y_dim = 1, 1
    h_dim = 50
    layer_sizes = [x_dim, h_dim, h_dim, y_dim]
    activation = nn.ReLU()
    layer_kwargs = {'prior_weight_std': 1.0,
                    'prior_bias_std': 1.0,
                    'sqrt_width_scaling': False,
                    'init_std': 0,
                    'device': device}
    model = make_linear_nn(layer_sizes, activation=activation, **layer_kwargs)
    log_noise_var = torch.ones(size=(), device=device)*-4.6  # Gaussian likelihood
    print("BNN architecture: \n", model)

    d.plot_bnn_pred_post(model, predict, normal_train, test, log_noise_var, noise_std,
                         'BNN init (before training, MFVI)', device)

    # training hyperparameters
    learning_rate = 1e-3
    params = list(model.parameters()) + [log_noise_var]
    opt = torch.optim.Adam(params, lr=learning_rate)
    lr_sch = torch.optim.lr_scheduler.StepLR(opt, 2000, gamma=0.1)
    N_epochs = 1000

    nll = nn.GaussianNLLLoss(full=True, reduction='sum')

    # training loop
    model.train()
    logs = []
    for i in range(N_epochs):
        loss = train_step(
            model, opt, nll, train_loader, log_noise_var, device=device
        )
        logs.append([to_numpy(loss)])
        if (i+1) % 100 == 0:
            print("Epoch {}, nll={}".format(i+1, logs[-1][0]))
    logs = np.array(logs)

    # plot the training curve
    def plot_training_loss(logs):
        plt.plot(np.arange(logs.shape[0]), logs[:, 0], 'r-')
        plt.xlabel('epoch')
        plt.title('nll')
        plt.show()

    plot_training_loss(logs)

    d.plot_bnn_pred_post(model, predict, normal_train, test, log_noise_var, noise_std,
                         'BNN approx. posterior (MFVI)', device)
