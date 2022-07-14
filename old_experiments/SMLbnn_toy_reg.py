import torch
import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt

import datasets.toy_regression as d
from modules.bnn.modules.ext_emp_linear import make_linear_ext_emp_bnn
from modules.bnn.modules.loss import GaussianKLLoss, nELBO
from modules.bnn.utils import *


TRAINING = True


experiment_name = 'SMLbnn_toy_reg-12-07'
if TRAINING:
    experiment_name = uniquify(experiment_name)

torch.manual_seed(1)

# create dataset
N_data = 100; noise_std = 0.1
train_loader, test_loader, train_data, test_data = d.create_regression_dataset(N_data, noise_std)

# create bnn
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
x_dim, y_dim = train_data.x.shape[1], train_data.y.shape[1]
h_dim = 50
layer_sizes = [x_dim, h_dim, h_dim, y_dim]
activation = nn.GELU()
layer_kwargs = {'init_std': 0.05,
                'device': device}
model = make_linear_ext_emp_bnn(layer_sizes, activation=activation, **layer_kwargs)
log_noise_var = torch.ones(size=(), device=device)*-3.0  # Gaussian likelihood
print("BNN architecture: \n", model)

d.plot_bnn_prior(model, predict, train_data, test_data, log_noise_var, noise_std, device)

# training hyperparameters
learning_rate = 1e-4
params = list(model.parameters()) + [log_noise_var]
opt = torch.optim.Adam(params, lr=learning_rate)
# hyper-parameters of training
N_epochs = 2000

gnll_loss = nn.GaussianNLLLoss(full=True, reduction='sum')
kl_loss = GaussianKLLoss()
nelbo = nELBO(nll_loss=gnll_loss, kl_loss=kl_loss)

print(f"kl before training: {kl_loss(model)}")
logs = training_loop(model, N_epochs, opt, nelbo, train_loader, test_loader, log_noise_var, experiment_name)

plot_training_loss(logs)
write_logs_to_file(logs, experiment_name)
