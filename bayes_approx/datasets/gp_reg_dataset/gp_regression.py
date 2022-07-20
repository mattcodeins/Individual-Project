import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


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


def to_numpy(x):
    return x.detach().cpu().numpy()


def normalise_data(x, mean, std):
    return (x - mean) / std


def unnormalise_data(x, mean, std):
    return x * std + mean


def import_dataset(fname):
    data = np.genfromtxt(fname, delimiter=',')
    return data


def import_train_test():
    noise_std = 0.1
    train = import_dataset('bayes_approx/datasets/gp_reg_dataset/training_sample_points.csv')
    test = import_dataset('bayes_approx/datasets/gp_reg_dataset/sample_function.csv')
    return train, test, noise_std


def create_regression_dataset():
    train, test, noise_std = import_train_test()
    train_x = train[:,0].reshape(-1,1); train_y = train[:,1].reshape(-1,1)
    test_x = test[:,0].reshape(-1,1); test_y = test[:,1].reshape(-1,1)
    normalised_train = regression_data(train_x, train_y)
    test = regression_data(test_x, test_y, normalise=False)

    # plt.plot(train_x, train_y, 'ro', label='data')
    # plt.plot(test_x, test_y, 'k-', label='ground-truth')
    # plt.legend()
    # plt.title('ground-truth function')
    # plt.show()

    train_loader = DataLoader(normalised_train, batch_size=50, shuffle=True)
    test_loader = DataLoader(test, batch_size=1000, shuffle=True)
    return train_loader, test_loader, normalised_train, test, noise_std


def get_regression_results(model, x, K, predict, dataset, log_noise_var=None):
    if K > 1:
        y_pred_mean, y_pred_std = predict(model, x, K=K)  # shape (K, N_test, y_dim)
    else:
        return unnormalise_data(to_numpy(model(x)), dataset.y_mean, dataset.y_std)
    if log_noise_var is not None:
        # total uncertainty: here the preditive std needs to count for output noise variance
        y_pred_std = (y_pred_std**2 + torch.exp(log_noise_var)).sqrt()
    # unnormalise
    y_pred_mean = unnormalise_data(to_numpy(y_pred_mean), dataset.y_mean, dataset.y_std)
    y_pred_std = unnormalise_data(to_numpy(y_pred_std), 0.0, dataset.y_std)
    return y_pred_mean, y_pred_std


def plot_regression(normal_train, test, y_pred_mean, y_pred_std_noiseless, y_pred_std, y_pred_mean_samples, title=''):
    x_train = unnormalise_data(normal_train.x, normal_train.x_mean, normal_train.x_std)
    y_train = unnormalise_data(normal_train.y, normal_train.y_mean, normal_train.y_std)

    # first for the total uncertainty (model/epistemic + data/aleatoric)
    plt.figure(figsize=(12, 6))
    plt.plot(x_train, y_train, "kx", mew=2, label='noisy sample points')
    plt.plot(test.x, y_pred_mean, "C0", lw=2, label='prediction mean')
    plt.fill_between(
        test.x[:,0],
        y_pred_mean[:,0] - 1.96 * y_pred_std[:,0],  # 95% confidence interval
        y_pred_mean[:,0] + 1.96 * y_pred_std[:,0],
        color="C0",
        alpha=0.2,
        label='total uncertainity'
    )
    plt.fill_between(
        test.x[:,0],
        y_pred_mean[:,0] - 1.96 * y_pred_std_noiseless[:,0],  # 95% confidence interval
        y_pred_mean[:,0] + 1.96 * y_pred_std_noiseless[:,0],
        color="b",
        alpha=0.2,
        label='model uncertainity'
    )

    plt.plot(test.x, np.array(y_pred_mean_samples)[:,:,0].T, "C0", linewidth=0.5)

    plt.plot(test.x, test.y, color='orange', label='sample function')
    plt.legend()
    plt.title(title)
    plt.show()


def plot_bnn_pred_post(model, predict, normal_train, test, log_noise_var, noise_std, title, device):
    # plot the BNN prior in function space
    x_test_norm = normalise_data(test.x, normal_train.x_mean, normal_train.x_std)
    x_test_norm = torch.tensor(x_test_norm,).float().to(device)

    y_pred_mean, y_pred_std_noiseless = get_regression_results(
        model, x_test_norm, 50, predict, normal_train, log_noise_var
    )
    y_pred_mean_samples = [
        get_regression_results(model, x_test_norm, 1, predict, normal_train, log_noise_var)
        for _ in range(10)]
    model_noise_std = unnormalise_data(to_numpy(torch.exp(0.5*log_noise_var)), 0.0, normal_train.y_std)
    y_pred_std = np.sqrt(y_pred_std_noiseless**2 + model_noise_std**2)
    plot_regression(normal_train, test, y_pred_mean, y_pred_std_noiseless, y_pred_std, y_pred_mean_samples, title)
    print(model_noise_std, noise_std, y_pred_std_noiseless.mean())


def plot_training_loss(logs):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.plot(np.arange(logs.shape[0]), logs[:, 0], 'r-')
    ax2.plot(np.arange(logs.shape[0]), logs[:, 1], 'r-')
    ax1.set_xlabel('epoch')
    ax2.set_xlabel('epoch')
    ax1.set_title('nll')
    ax2.set_title('kl')
    plt.show()


# def plot_regression(normal_train, test, y_pred_mean, y_pred_std_noiseless, y_pred_std, title=''):
#     x_train = unnormalise_data(normal_train.x, normal_train.x_mean, normal_train.x_std)
#     y_train = unnormalise_data(normal_train.y, normal_train.y_mean, normal_train.y_std)
#     plt.plot(x_train, y_train, 'ro', label='data')
#     plt.plot(test.x, test.y, 'k-', label='ground-truth')
#     plt.plot(test.x, y_pred_mean, 'b-', label='prediction mean')
#     # plot the uncertainty as +- 2 * std
#     # first for the total uncertainty (model/epistemic + data/aleatoric)
#     plt.fill_between(test.x[:,0], y_pred_mean[:,0] - 1.96*y_pred_std[:,0],
#                      y_pred_mean[:,0] + 2*y_pred_std[:,0],
#                      color='c', alpha=0.3, label='total uncertainty')
#     # then for the model/epistemic uncertainty only
#     plt.fill_between(test.x[:,0], y_pred_mean[:,0] - 1.96*y_pred_std_noiseless[:,0],
#                      y_pred_mean[:,0] + 2*y_pred_std_noiseless[:,0],
#                      color='b', alpha=0.3, label='model uncertainty')
#     plt.legend()
#     plt.title(title)
#     plt.show()


# def plot_bnn_pred_post(model, predict, normal_train, test, log_noise_var, noise_std, title, device):
#     # plot the BNN prior in function space
#     K = 50  # number of Monte Carlos samples used in test time
#     x_test_norm = normalise_data(test.x, normal_train.x_mean, normal_train.x_std)
#     x_test_norm = torch.tensor(x_test_norm,).float().to(device)

#     y_pred_mean, y_pred_std_noiseless = get_regression_results(model, x_test_norm, K, predict, normal_train)
#     model_noise_std = unnormalise_data(to_numpy(torch.exp(0.5*log_noise_var)), 0.0, normal_train.y_std)
#     y_pred_std = np.sqrt(y_pred_std_noiseless**2 + model_noise_std**2)
#     plot_regression(normal_train, test, y_pred_mean, y_pred_std_noiseless, y_pred_std, title)
#     print(model_noise_std, noise_std, y_pred_std_noiseless.mean())


if __name__ == '__main__':
    train, test, _ = import_train_test()
    print(train.shape)
    print(test.shape)
