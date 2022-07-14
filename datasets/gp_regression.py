import numpy as np
import math
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# from modules.bnn.utils import *


def import_dataset(fname, batch_size=64):
    data = np.genfromtxt(fname, delimiter=',')
    x = data[:,0]
    loader = DataLoader(x, batch_size, shuffle=True)
    return data, loader


def import_train_test():
    train, trainloader = import_dataset('training_sample_points.csv', 128)
    test, testloader = import_dataset('sample_function.csv', 1000)
    return train, test, trainloader, testloader


# plot the BNN prior and ground truth
def plot_regression(train_data, test_data, y_pred_mean, y_pred_std_noiseless, y_pred_std, title=''):
    plt.plot(train_data.x, train_data.y, 'ro', label='data')
    plt.plot(test_data.x, test_data.y, 'k-', label='ground-truth')
    plt.plot(test_data.x, y_pred_mean, 'b-', label='prediction mean')
    # plot the uncertainty as +- 2 * std
    # first for the total uncertainty (model/epistemic + data/aleatoric)
    plt.fill_between(test_data.x[:,0], y_pred_mean[:,0] - (2 * y_pred_std[:,0]),
                     y_pred_mean[:,0] + (2 * y_pred_std[:,0]),
                     color='c', alpha=0.3, label='total uncertainty')
    # then for the model/epistemic uncertainty only
    plt.fill_between(test_data.x[:,0], y_pred_mean[:,0] - (2 * y_pred_std_noiseless[:,0]),
                     y_pred_mean[:,0] + (2 * y_pred_std_noiseless[:,0]),
                     color='b', alpha=0.3, label='model uncertainty')
    plt.legend()
    plt.title(title)
    plt.show()


def plot_bnn_prior(model, predict, train_data, test_data, log_noise_var, noise_std, device):
    # plot the BNN prior in function space
    K = 50  # number of Monte Carlos samples used in test time
    x_test_norm = normalise_data(test_data.x, train_data.x_mean, train_data.x_std)
    x_test_norm = torch.tensor(x_test_norm, ).float().to(device)
    y_pred_mean, y_pred_std_noiseless = get_regression_results(model, x_test_norm, K, predict, train_data)
    model_noise_std = unnormalise_data(to_numpy(torch.exp(0.5*log_noise_var)), 0.0, train_data.y_std)
    y_pred_std = np.sqrt(y_pred_std_noiseless ** 2 + model_noise_std**2)
    plot_regression(train_data, test_data, y_pred_mean, y_pred_std_noiseless, y_pred_std,
                    title='BNN init (before training, MFVI)')
    print(model_noise_std, noise_std, y_pred_std_noiseless.mean())


if __name__ == '__main__':
    import_dataset('datasets/gp_reg_dataset/training_sample_points.csv')