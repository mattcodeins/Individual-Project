import gpflow
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from gpflow.utilities import print_summary


# data = np.genfromtxt('datasets/gp_reg_dataset/points.csv', delimiter=',')
data = np.genfromtxt('datasets/gp_reg_dataset/training_sample_points.csv', delimiter=',')

X = data[:, 0].reshape(-1, 1)
Y = data[:, 1].reshape(-1, 1)
# X = X / X[-1]
# Y = (Y - Y.mean()) / Y.std()

plt.plot(X, Y, "kx", mew=2)
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
xx = np.linspace(-0.1, 1.1, 100).reshape(100, 1)  # test points must be of shape (N, D)

# predict mean and variance of latent GP at test points
mean, var = m.predict_f(xx)

# generate 10 samples from posterior
tf.random.set_seed(1)  # for reproducibility
samples = m.predict_f_samples(xx, 10)  # shape (10, 100, 1)

# plot
plt.figure(figsize=(12, 6))
plt.plot(X, Y, "kx", mew=2)
plt.plot(xx, mean, "C0", lw=2)
plt.fill_between(
    xx[:, 0],
    mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),  # 95% confidence interval
    mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
    color="C0",
    alpha=0.2,
)

plt.plot(xx, samples[:, :, 0].numpy().T, "C0", linewidth=0.5)


test_data = np.genfromtxt('datasets/gp_reg_dataset/sample_function.csv', delimiter=',')
print(test_data.shape)
plt.plot(test_data[:,0], test_data[:,1], color='orange')
plt.show()
