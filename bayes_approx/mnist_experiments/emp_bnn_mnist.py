import torch
import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt

from modules.bnn.modules.emp_linear import make_linear_emp_bnn
from modules.bnn.modules.loss import GaussianKLLoss, nELBO
from modules.bnn.utils import *

import datasets.mnist as d


torch.manual_seed(1)
experiment_name = 'mnist_emp_bnn_singlestdprior_init_as_prior'

# create dataset
batch_size_train = 64
batch_size_test = 1000
train_loader, test_loader = d.import_n_mnist(batch_size_train, batch_size_test)

# create bnn
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
x_dim, y_dim = 784, 10
h1_dim, h2_dim = 128, 64
layer_sizes = [x_dim, h1_dim, h2_dim, y_dim]
activation = nn.GELU()
model = make_linear_emp_bnn(layer_sizes, activation=activation, device=device)
print("BNN architecture: \n", model)

# training hyperparameters
learning_rate = 1e-4
params = list(model.parameters())  # + [log_noise_var]
opt = torch.optim.Adam(params, lr=learning_rate)
N_epochs = 2000

# define loss function (-ELBO)
cross_entropy_loss = nn.CrossEntropyLoss(reduction='sum')
kl_loss = GaussianKLLoss()
nelbo = nELBO(nll_loss=cross_entropy_loss, kl_loss=kl_loss)

print(f"kl before training: {kl_loss(model)}")  # should be approx. 273,465 for MNIST (109,386 params)
logs = training_loop(model, N_epochs, opt, nelbo, train_loader, test_loader, experiment_name)

plot_training_loss(logs)
write_logs_to_file(logs, experiment_name)
