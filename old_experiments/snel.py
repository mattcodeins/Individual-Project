from datasets.snelson1d import snelson1d
import numpy as np
import matplotlib.pyplot as plt

(Xd, Yd), _ = snelson1d()
plt.scatter(Xd, Yd)
plt.show()
indices = np.random.permutation(Xd.shape[0])
training_idx, test_idx = indices[:180], indices[180:]
X = Xd[training_idx]; Y = Yd[training_idx]
Xt = Xd[test_idx]; Yt = Yd[test_idx]


# plot the training data and ground truth
plt.scatter(X, Y, label='train')
plt.scatter(Xt, Yt, label='test')
plt.legend()
plt.title('ground-truth function')

plt.show()
