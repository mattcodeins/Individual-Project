import numpy as np
import math
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from modules.bnn.utils import to_numpy


def ground_truth_func(x):
    return np.sin(x * math.pi / 2 + 0.8) * np.exp(-0.1 * np.abs(x)) + 0.1 * x


def gen_data(N_data, ground_truth_func, noise_std=None):
    # generate the training dataset, note here we will make data into 2 clusters
    x1 = np.random.randn(int(N_data/2), 1) * 0.5 + 2.0
    x2 = np.random.randn(int(N_data/2), 1) * 0.5 - 2.0
    x = np.concatenate([x1, x2], axis=0)
    y = ground_truth_func(x)
    if noise_std is not None and noise_std > 1e-6:
        # assume homogeneous noise setting, i.e., "homoscedasticity"
        y += np.random.randn(y.shape[0], y.shape[1]) * noise_std
    return x, y


def normalise_data(x, mean, std):
    return (x - mean) / std


def unnormalise_data(x, mean, std):
    return x * std + mean


class regression_data(Dataset):
    def __init__(self, x, y, normalise=True):
        super(regression_data, self).__init__()
        self.update_data(x, y, normalise)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = torch.tensor(self.x[index]).float()
        y = torch.tensor(self.y[index]).float()
        return x, y

    def update_data(self, x, y, normalise=True, update_stats=True):
        assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y
        # normalise data
        self.normalise = normalise
        if update_stats:
            self.x_mean = self.x.mean(0) if normalise else 0.0
            self.x_std = self.x.std(0) if normalise else 1.0
            self.y_mean = self.y.mean(0) if normalise else 0.0
            self.y_std = self.y.std(0) if normalise else 1.0
        if self.normalise:
            self.x = normalise_data(self.x, self.x_mean, self.x_std)
            self.y = normalise_data(self.y, self.y_mean, self.y_std)


def create_regression_dataset(N_data=100, noise_std=0.1):
    x_train, y_train = gen_data(N_data, ground_truth_func, noise_std)
    dataset = regression_data(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

    # plot the training data and ground truth
    x_test = np.arange(np.min(x_train) - 1.0, np.max(x_train) + 1.0, 0.01)[:, np.newaxis]
    y_test = ground_truth_func(x_test)
    plt.plot(x_train, y_train, 'ro', label='data')
    plt.plot(x_test, y_test, 'k-', label='ground-truth')
    plt.legend()
    plt.title('ground-truth function')
    plt.show()
    return dataloader, dataset, x_train, y_train, x_test, y_test


def get_regression_results(net, x, K, predict, dataset, log_noise_var=None):
    y_pred = predict(net, x, K=K)  # shape (K, N_test, y_dim)
    y_pred_mean = y_pred.mean(0)
    if log_noise_var is not None:
        # total uncertainty: here the preditive std needs to count for output noise variance
        y_pred_std = (y_pred.var(0) + torch.exp(log_noise_var)).sqrt()
    else:
        # model/epistemic uncertainty: only describing the variation of functions sampled from BNN
        y_pred_std = y_pred.std(0)
    # unnormalise
    y_pred_mean = unnormalise_data(to_numpy(y_pred_mean), dataset.y_mean, dataset.y_std)
    y_pred_std = unnormalise_data(to_numpy(y_pred_std), 0.0, dataset.y_std)
    return y_pred_mean, y_pred_std


# plot the BNN prior and ground truth
def plot_regression(x_train, y_train, x_test, y_test, y_pred_mean, y_pred_std_noiseless, y_pred_std, title=''):
    plt.plot(x_train, y_train, 'ro', label='data')
    plt.plot(x_test, y_test, 'k-', label='ground-truth')
    plt.plot(x_test, y_pred_mean, 'b-', label='prediction mean')
    # plot the uncertainty as +- 2 * std
    # first for the total uncertainty (model/epistemic + data/aleatoric)
    plt.fill_between(x_test[:,0], y_pred_mean[:,0] - (2 * y_pred_std[:,0]),
                     y_pred_mean[:,0] + (2 * y_pred_std[:,0]),
                     color='c', alpha=0.3, label='total uncertainty')
    # then for the model/epistemic uncertainty only
    plt.fill_between(x_test[:,0], y_pred_mean[:,0] - (2 * y_pred_std_noiseless[:,0]),
                     y_pred_mean[:,0] + (2 * y_pred_std_noiseless[:,0]),
                     color='b', alpha=0.3, label='model uncertainty')
    plt.legend()
    plt.title(title)
    plt.show()
