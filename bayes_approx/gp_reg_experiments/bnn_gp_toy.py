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

    # import dataset
    train, test = d.import_train_test()

    X = train[:,0].reshape(-1, 1)
    Y = train[:,1].reshape(-1, 1)
    X_test = test[:,0].reshape(-1, 1)
    Y_test = test[:,1].reshape(-1, 1)

    # create bnn
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    x_dim, y_dim = X.shape[1], Y.shape[1]
    h_dim = 100
    layer_sizes = [x_dim, h_dim, y_dim]
    activation = nn.ReLU()
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
            model, opt, nelbo, dataloader, log_noise_var, device=device
        )
        logs.append([to_numpy(nll), to_numpy(kl), to_numpy(loss), to_numpy(nll)/to_numpy(kl)])
        if (i+1) % 100 == 0:
            print("Epoch {}, nll={}, kl={}, nelbo={}, ratio={}"
                  .format(i+1, logs[-1][0], logs[-1][1], logs[-1][2], logs[-1][3]))
    logs = np.array(logs)

    # plot the training curve
    def plot_training_loss(logs):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.plot(np.arange(logs.shape[0]), logs[:, 0], 'r-')
        ax2.plot(np.arange(logs.shape[0]), logs[:, 1], 'r-')
        ax1.set_xlabel('epoch')
        ax2.set_xlabel('epoch')
        ax1.set_title('nll')
        ax2.set_title('kl')
        plt.show()

    plot_training_loss(logs)

    y_pred_mean, y_pred_std_noiseless = d.get_regression_results(model, x_test_norm, K, predict, dataset)
    model_noise_std = d.unnormalise_data(to_numpy(torch.exp(0.5*log_noise_var)), 0.0, dataset.y_std)
    y_pred_std = np.sqrt(y_pred_std_noiseless ** 2 + model_noise_std ** 2)
    d.plot_regression(x_train, y_train, x_test, y_test, y_pred_mean, y_pred_std_noiseless, y_pred_std,
                      title='BNN approx. posterior (MFVI)')
    print(model_noise_std, noise_std, y_pred_std_noiseless.mean())
