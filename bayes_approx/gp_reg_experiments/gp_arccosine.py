import gpflow
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from gpflow.utilities import print_summary

import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import datasets.gp_reg_dataset.gp_regression as d

np.random.seed(1)
tf.random.set_seed(1)

train, test, noise_std = d.import_train_test()

X = train[:,0].reshape(-1, 1)
Y = train[:,1].reshape(-1, 1)
X_test = test[:,0].reshape(-1, 1)
Y_test = test[:,1].reshape(-1, 1)

plt.plot(X, Y, "kx", mew=2, label='noisy sample points')
plt.legend()
plt.show()

k = gpflow.kernels.ArcCosine(1)
print_summary(k)

m = gpflow.models.GPR(data=(X, Y), kernel=k, mean_function=None)
# m.likelihood.variance.assign(0.05)
# gpflow.set_trainable(m.likelihood, False)
print_summary(m)

opt = gpflow.optimizers.Scipy()

opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=1000))
print_summary(m)

# generate test points for prediction
# xx = np.linspace(-0.1, 1.1, 100).reshape(100, 1)  # test points must be of shape (N, D)

# predict mean and variance of latent GP at test points
mean, var = m.predict_f(X_test)
print(mean.numpy().mean())

# generate 10 samples from posterior
tf.random.set_seed(1)  # for reproducibility
samples = m.predict_f_samples(X_test, 10)  # shape (10, 100, 1)

# plot
plt.figure(figsize=(12, 6))
plt.plot(X, Y, "kx", mew=2, label='noisy sample points')
plt.plot(X_test, mean, "C0", lw=2, label='gp prediction mean')
plt.fill_between(
    X_test[:,0],
    mean[:,0] - 1.96 * np.sqrt(var[:,0]),  # 95% confidence interval
    mean[:,0] + 1.96 * np.sqrt(var[:,0]),
    color="C0",
    alpha=0.2
)
plt.plot(X_test, samples[:,:,0].numpy().T, "C0", linewidth=0.5)


print(Y_test.mean())
plt.plot(X_test, Y_test, color='orange', label='sample function')
plt.legend()
plt.show()
