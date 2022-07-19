import gpflow
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from gpflow.utilities import print_summary

np.random.seed(1)
tf.random.set_seed(1)

data = np.genfromtxt('bayes_approx/datasets/gp_reg_dataset/points.csv', delimiter=',')

X = data[:, 0].reshape(-1, 1)
Y = data[:, 1].reshape(-1, 1)
Y = (Y - Y.mean()) / Y.std()

k = gpflow.kernels.Matern52()
print_summary(k)

m = gpflow.models.GPR(data=(X, Y), kernel=k, mean_function=None)
m.likelihood.variance.assign(0.05)
gpflow.set_trainable(m.likelihood, False)
print_summary(m)

opt = gpflow.optimizers.Scipy()

opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=1000))
print_summary(m)


# generate test points for prediction
xx = np.linspace(0, 1, 1000).reshape(1000, 1)  # test points must be of shape (N, D)

# generate 1 sample from posterior in linspace for testing
sample = m.predict_f_samples(xx, 1)[0,:,0]  # shape (1000,)

print(sample.numpy().mean(0))

plt.scatter(xx, sample, marker='x')

x1 = np.random.randn(20, 1) * 0.04 + 0.2
x2 = np.random.randn(10, 1) * 0.04 + 0.5
x3 = np.random.randn(20, 1) * 0.04 + 0.8
xt = np.sort(np.concatenate([x1, x2, x3], axis=0),0)
sample2 = sample.numpy().take((xt[:,0] * 1000).astype(int))

noise_std = 0.1
if noise_std is not None and noise_std > 1e-6:
    # assume homogeneous noise setting, i.e., "homoscedasticity"
    sample2 += np.random.randn(sample2.shape[0]) * noise_std

print(sample2.mean(0))
plt.scatter(xt, sample2)
plt.show()

path = 'bayes_approx/datasets/gp_reg_dataset/'
np.savetxt(path + 'sample_function.csv', np.dstack((xx[:,0], sample))[0], delimiter=',')
np.savetxt(path + 'training_sample_points.csv', np.dstack((xt[:,0], sample2))[0], delimiter=',')
