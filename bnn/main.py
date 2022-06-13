import numpy as np
from sklearn import datasets

import torch
import torch.nn as nn
import torch.optim as optim

import modules as bnn

import matplotlib.pyplot as plt

x = torch.linspace(-2, 2, 500)
y = x.pow(3) - x.pow(2) + 3*torch.rand(x.size())
x = torch.unsqueeze(x, dim=1)
y = torch.unsqueeze(y, dim=1)

plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()

model = nn.Sequential(
    bnn.BayesLinear(in_features=1, out_features=100),
    nn.ReLU(),
    bnn.BayesLinear(in_features=100, out_features=1),
)