import torch
import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
# import torchvision

from modules.bnn.modules.ext_emp_linear import make_linear_ext_emp_bnn
from modules.bnn.modules.loss import GaussianKLLoss, nELBO
from modules.bnn.utils import *

import datasets.mnist as d


TRAINING = False


torch.manual_seed(1)
experiment_name = uniquify('mnist_emp_bnn_multiprior_initprior')

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
model = make_linear_ext_emp_bnn(layer_sizes, activation=activation, device=device, init_std=1.0)
print("BNN architecture: \n", model)

# training hyperparameters
learning_rate = 1e-4
params = list(model.parameters())
opt = torch.optim.Adam(params, lr=learning_rate)
N_epochs = 2000

# define loss function (-ELBO)
cross_entropy_loss = nn.CrossEntropyLoss(reduction='sum')
kl_loss = GaussianKLLoss()
nelbo = nELBO(nll_loss=cross_entropy_loss, kl_loss=kl_loss)

if TRAINING:
    print(f"kl before training: {kl_loss(model)}")  # should be approx. 273,465 for MNIST (109,386 params)
    logs = training_loop(model, N_epochs, opt, nelbo, train_loader, test_loader, experiment_name)

    plot_training_loss(logs)
    write_logs_to_file(logs, experiment_name)

else:
    model = load_model(model, experiment_name)
    test_step(model, nelbo, test_loader, predict=predict_wo_var, device=device)
