import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


ng = np.random.default_rng(1)

data = np.genfromtxt('datasets/gp_reg_dataset/sample_points.csv', delimiter=',')
N_data = 30
noise_std = 0.05

# generate the training dataset, note here we will make data into 2 clusters
x1 = np.random.randn(100, int(N_data/2)) * 0.5 + 0.75
x2 = np.random.choice(100, int(N_data/2)) * 0.5 + 0.25
x = np.concatenate([x1, x2], axis=0)
print(x.shape)
train_data = data[x.astype(int)]
# if noise_std is not None and noise_std > 1e-6:
#     # assume homogeneous noise setting, i.e., "homoscedasticity"
#     y += np.random.randn(y.shape[0], y.shape[1]) * noise_std
plt.plot(train_data)
plt.show()
