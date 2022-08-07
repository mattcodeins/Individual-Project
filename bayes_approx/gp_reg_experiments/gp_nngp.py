import jax.numpy as np
from jax import random
from jax.example_libraries import optimizers
from jax import jit, grad, vmap
import functools
import neural_tangents as nt
from neural_tangents import stax
import matplotlib.pyplot as plt
import torch

import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import datasets.gp_reg_dataset.gp_regression as d

np.random.seed(1)
torch.manual_seed(1)

train, test, noise_std = d.import_train_test()

X = train[:,0].reshape(-1, 1)
Y = train[:,1].reshape(-1, 1)
X_test = test[:,0].reshape(-1, 1)
Y_test = test[:,1].reshape(-1, 1)

plt.plot(X, Y, "kx", mew=2, label='noisy sample points')
plt.legend()
plt.show()

init_fn, apply_fn = stax.serial(
    stax.Dense(50), stax.Relu,
    stax.Dense(50), stax.Relu,
    stax.Dense(1)
)

key = random.PRNGKey(1)
x = random.normal(key, (10, 100))
_, params = init_fn(key, input_shape=x.shape)

y = apply_fn(params, x)  # (10, 1) np.ndarray outputs of the neural network
