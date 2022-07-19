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

from datasets import toy_regression as d
from modules.bnn.modules.linear import make_linear_bnn
from modules.bnn.modules.loss import GaussianKLLoss, nELBO
from modules.bnn.utils import *


def train_step(model, opt, nelbo, dataloader, log_noise_var, device):
    tloss, tnll, tkl = 0,0,0
    for _, (x, y) in enumerate(dataloader):
        minibatch_ratio = x.shape[0] / len(dataloader.dataset)
        noise_var = torch.exp(log_noise_var)*torch.ones(x.shape[0])
        x = x.to(device); y = y.to(device)
        y_pred = model(x)
        loss, nll, kl = nelbo(model, (y_pred, y, noise_var), minibatch_ratio)
        opt.zero_grad()
        loss.backward()
        opt.step()
        tloss += loss; tnll += nll; tkl += kl
    return tloss, tnll, tkl


def predict(bnn, x_test, K=1):  # Monte Carlo sampling using K samples
    y_pred = []
    for _ in range(K):
        y_pred.append(bnn(x_test))
    # shape (K, batch_size, y_dim) or (batch_size, y_dim) if K = 1
    y_pred = torch.stack(y_pred, dim=0).squeeze(0)
    return y_pred.mean(0), y_pred.std(0)


if __name__ == "__main__":
    torch.manual_seed(1)

    # create dataset
    N_data = 100; noise_std = 0.1
    train_loader, test_loader, normal_train, test = d.create_regression_dataset(N_data, noise_std)

    # create bnn
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    x_dim, y_dim = normal_train.x.shape[1], normal_train.y.shape[1]
    h_dim = 50
    layer_sizes = [x_dim, h_dim, h_dim, y_dim]
    activation = nn.GELU()
    layer_kwargs = {'prior_weight_std': 1.0,
                    'prior_bias_std': 1.0,
                    'sqrt_width_scaling': False,
                    'init_std': 0.05,
                    'device': device}
    model = make_linear_bnn(layer_sizes, activation=activation, **layer_kwargs)
    log_noise_var = torch.ones(size=(), device=device)*-3.0  # Gaussian likelihood
    print("BNN architecture: \n", model)

    d.plot_bnn_pred_post(model, predict, normal_train, test, log_noise_var, noise_std,
                         'BNN init (before training, MFVI)', device)

    # training hyperparameters
    learning_rate = 1e-4
    params = list(model.parameters()) + [log_noise_var]
    opt = torch.optim.Adam(params, lr=learning_rate)
    # hyper-parameters of training
    N_epochs = 5000

    gnll_loss = nn.GaussianNLLLoss(full=True, reduction='sum')
    kl_loss = GaussianKLLoss()
    nelbo = nELBO(nll_loss=gnll_loss, kl_loss=kl_loss)

    # training loop
    model.train()
    logs = []
    for i in range(N_epochs):
        loss, nll, kl = train_step(
            model, opt, nelbo, train_loader, log_noise_var, device=device
        )
        logs.append([to_numpy(nll), to_numpy(kl), to_numpy(loss), to_numpy(nll)/to_numpy(kl)])
        if (i+1) % 100 == 0:
            print("Epoch {}, nll={}, kl={}, nelbo={}, ratio={}"
                  .format(i+1, logs[-1][0], logs[-1][1], logs[-1][2], logs[-1][3]))
    logs = np.array(logs)

    d.plot_training_loss(logs)

    d.plot_bnn_pred_post(model, predict, normal_train, test, log_noise_var, noise_std,
                         'BNN approx. posterior (MFVI)', device)