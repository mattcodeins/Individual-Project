# import numpy as np
# from sklearn import datasets

import torch
import torch.nn as nn
import torch.optim as optim
from modules import BayesLinear, GaussianKLLoss, nELBO

import matplotlib.pyplot as plt

model = nn.Sequential(
    BayesLinear(in_features=1, out_features=100),
    nn.ReLU(),
    BayesLinear(in_features=100, out_features=1),
)

gnll_loss = nn.GaussianNLLLoss(full=True, reduction='sum')
kl_loss = GaussianKLLoss()
nelbo = nELBO(nll_loss=gnll_loss, kl_loss=kl_loss)

noise_var = 0.05 ** 2 * torch.ones(x.shape[0])

opt = optim.Adam(model.parameters(), lr=0.01)


def train_step(model, opt, elbo, noise_var, dataloader, N_data, device):
    for _, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad()  # opt is the optimiser
        ### begin of your code ###
        y_pred = model(x)
        loss, nll, kl = nelbo(model, (y, y_pred, noise_var))

        ### end of your code ###
        loss.backward()
        opt.step()
    return nll, kl


for step in range(3000):
    y_pred = model(x)
    loss, nll, kl = nelbo(model, (y, y_pred, noise_var))

    opt.zero_grad()
    loss.backward()
    opt.step()

print('- GNLL : %2.2f, KL : %2.2f' % (nll.item(), kl.item()))
