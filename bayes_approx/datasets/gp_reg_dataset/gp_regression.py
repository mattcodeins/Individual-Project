import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import tensorflow as tf

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'


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
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    else:
        return x


def normalise_data(x, mean, std):
    return (x - mean) / std


def unnormalise_data(x, mean, std):
    return x * std + mean


def import_dataset(fname):
    data = np.genfromtxt(fname, delimiter=',')
    return data


def import_train_test():
    noise_std = 0.1
    train = import_dataset(f'bayes_approx/datasets/gp_reg_dataset/training_sample_points{noise_std}.csv')
    test = import_dataset(f'bayes_approx/datasets/gp_reg_dataset/sample_function{noise_std}.csv')
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


def create_regression_dataset_kf(kf):
    train, test, noise_std = import_train_test()
    train_x = train[:,0].reshape(-1,1); train_y = train[:,1].reshape(-1,1)
    test_x = test[:,0].reshape(-1,1); test_y = test[:,1].reshape(-1,1)

    # kf split
    normalised_train_list = []
    val_list = []
    train_loader_list = []
    val_loader_list = []
    for train_index, val_index in kf.split(train_x):
        kf_train_x = train_x[train_index]
        kf_train_y = train_y[train_index]
        kf_val_x = train_x[val_index]
        kf_val_y = train_y[val_index]

        # plt.scatter(kf_train_x, kf_train_y)
        # plt.scatter(kf_val_x, kf_val_y)

        normalised_train_list.append(regression_data(kf_train_x, kf_train_y))
        val_list.append(regression_data(kf_val_x, kf_val_y, normalise=False))

        train_loader_list.append(DataLoader(normalised_train_list[-1], batch_size=50, shuffle=True))
        val_loader_list.append(DataLoader(val_list[-1], batch_size=50, shuffle=True))

    test = regression_data(test_x, test_y, normalise=False)

    # plt.plot(test_x, test_y, '-k', label='ground-truth')
    # plt.legend()
    # plt.title('ground-truth function')
    # plt.show()

    test_loader = DataLoader(test, batch_size=1000, shuffle=True)
    return train_loader_list, val_loader_list, test_loader, normalised_train_list, val_list, test, noise_std


def get_regression_results(model, x, predict, dataset, K=50, log_lik_var=None):
    y_pred_mean, y_pred_std = predict(model, x, K=K)  # shape (K, N_test, y_dim)
    if log_lik_var is not None:
        # total uncertainty: here the preditive std needs to count for output noise variance
        y_pred_std = (y_pred_std**2 + torch.exp(log_lik_var)).sqrt()
    # unnormalise
    y_pred_mean = unnormalise_data(to_numpy(y_pred_mean), dataset.y_mean, dataset.y_std)
    # y_pred_mean = unnormalise_data(y_pred_mean, dataset.y_mean, dataset.y_std)
    y_pred_std = unnormalise_data(to_numpy(y_pred_std), 0.0, dataset.y_std)
    # y_pred_std = unnormalise_data(y_pred_std, 0.0, dataset.y_std)
    return y_pred_mean, y_pred_std


def get_unnormal_regression_results(model, x, predict, K=50, log_lik_var=None):
    y_pred_mean, y_pred_std = predict(model, x, K=K)  # shape (K, N_test, y_dim)
    if log_lik_var is not None:
        # total uncertainty: here the preditive std needs to count for output noise variance
        y_pred_std = (y_pred_std**2 + torch.exp(log_lik_var)).sqrt()
    return to_numpy(y_pred_mean), to_numpy(y_pred_std)


def plot_regression(normal_train, test, y_pred_mean, y_pred_std_noiseless,
                    y_pred_std, y_pred_mean_samples, title='', exp_name=None):
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
        # y_pred_mean - 1.96 * y_pred_std,  # 95% confidence interval
        # y_pred_mean + 1.96 * y_pred_std,
        color="C0",
        alpha=0.2,
        label='total uncertainity'
    )
    plt.fill_between(
        test.x[:,0],
        y_pred_mean[:,0] - 1.96 * y_pred_std_noiseless[:,0],  # 95% confidence interval
        y_pred_mean[:,0] + 1.96 * y_pred_std_noiseless[:,0],
        # y_pred_mean - 1.96 * y_pred_std_noiseless,  # 95% confidence interval
        # y_pred_mean + 1.96 * y_pred_std_noiseless,
        color="b",
        alpha=0.2,
        label='model uncertainity'
    )

    plt.plot(test.x, np.array(y_pred_mean_samples)[:,:,0].T, "C0", linewidth=0.5)

    plt.plot(test.x, test.y, color='orange', label='sample function')
    plt.legend()
    plt.title(title)
    if exp_name is not None:
        plt.savefig(f'./figures/{exp_name}_out.png', bbox_inches='tight')
    plt.show()


def plot_bnn_pred_post(model, predict, normal_train, test, log_lik_var,
                       title=None, exp_name=None, device=device):
    if title is None:
        title = 'BNN approximate posterior (MFVI)'
    # plot the BNN prior in function space
    x_test_norm = normalise_data(test.x, normal_train.x_mean, normal_train.x_std)
    x_test_norm = torch.tensor(x_test_norm,).float().to(device)

    y_pred_mean, y_pred_std_noiseless = get_regression_results(
        model, x_test_norm, predict, normal_train
    )
    y_pred_mean_samples = [
        get_regression_results(model, x_test_norm, predict, normal_train, K=1)[0]
        for _ in range(10)
    ]
    model_noise_std = unnormalise_data(to_numpy(torch.exp(0.5*log_lik_var)), 0.0, normal_train.y_std)
    y_pred_std = np.sqrt(y_pred_std_noiseless**2 + model_noise_std**2)
    plot_regression(
        normal_train, test, y_pred_mean, y_pred_std_noiseless, y_pred_std, y_pred_mean_samples, title, exp_name
    )
    print(f'model_std: {model_noise_std}, pred_std: {y_pred_std_noiseless.mean()}')


# def plot_training_loss(logs):
#     _, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
#     ax1.plot(np.arange(logs.shape[0]), logs[:, 0], 'r-')
#     ax2.plot(np.arange(logs.shape[0]), logs[:, 1], 'r-')
#     ax1.set_xlabel('epoch')
#     ax2.set_xlabel('epoch')
#     ax1.set_title('nll')
#     ax2.set_title('kl')
#     plt.show()


def mse_test_step(model, dataloader, normal_train, predict, log_lik_var=None):
    """
    Calculate mean squared error on test set (normalising to train).
    """
    # model.eval()
    tloss = 0
    with torch.no_grad():
        for x_test, y_test in dataloader:
            x_test_norm = normalise_data(x_test, normal_train.x_mean, normal_train.x_std).float()
            y_pred_mean, _ = get_regression_results(
                model, x_test_norm, predict, normal_train, 50, log_lik_var
            )
            tloss += F.mse_loss(torch.from_numpy(y_pred_mean), y_test)
    print('\nTest set: MSE: {}'.format(tloss))
    return tloss/len(dataloader)


def gnll_test_step(model, dataloader, normal_train, predict, log_lik_var=None):
    """
    Calculate mean gaussian negative log likelihood on test set (normalising to train).
    """
    # model.eval()
    tloss = 0
    with torch.no_grad():
        for x_test, y_test in dataloader:
            x_test_norm = normalise_data(x_test, normal_train.x_mean, normal_train.x_std).float()
            y_pred_mean, y_pred_std = get_regression_results(
                model, x_test_norm, predict, normal_train, 50, log_lik_var
            )
            tloss += F.gaussian_nll_loss(
                torch.from_numpy(y_pred_mean), y_test, torch.from_numpy(y_pred_std), full=True
            )
    print('\nTest set: GNLL: {}'.format(tloss))
    return tloss/len(dataloader)


def mse_train(model, dataloader, predict, log_lik_var=None):
    """
    Calculate mean squared error on train set (without normalising to train).
    """
    # model.eval()
    tloss = 0
    with torch.no_grad():
        for x, y in dataloader:
            y_pred_mean, _ = get_unnormal_regression_results(
                model, x, predict, 50, log_lik_var
            )
            tloss += F.mse_loss(torch.from_numpy(y_pred_mean), y)
    print('\nTrain set: MSE: {}'.format(tloss))
    return tloss/len(dataloader)


def gnll_train(model, dataloader, predict, log_lik_var):
    """
    Calculate mean gaussian negative log likelihood on train set (without normalising to train).
    """
    # model.eval()
    tloss = 0
    with torch.no_grad():
        for x, y in dataloader:
            y_pred_mean, y_pred_std = get_unnormal_regression_results(
                model, x, predict, 50, log_lik_var
            )
            tloss += F.gaussian_nll_loss(
                torch.from_numpy(y_pred_mean), y, torch.from_numpy(y_pred_std), full=True
            )/len(dataloader)
    print('\nTrain set: GNLL: {}'.format(tloss))
    return tloss


def gpflow_get_regression_results(m, x, train, lik_var=None, unnormalise=True):
    y_pred_mean, y_pred_var = m.predict_f(x)
    y_pred_std = y_pred_var**0.5
    if lik_var is not None:
        # total uncertainty: here the preditive std needs to count for output noise variance
        y_pred_std = (y_pred_std**2 + lik_var)**0.5
    if unnormalise:
        y_pred_mean = unnormalise_data(y_pred_mean, train.y_mean, train.y_std)
        y_pred_std = unnormalise_data(y_pred_std, 0.0, train.y_std)
    return y_pred_mean, y_pred_std


def gpflow_plot_bnn_pred_post(m, train, test, lik_var, title=None, exp_name=None):
    if title is None:
        title = 'BNN approximate posterior (MFVI)'
    # predict mean and variance of latent GP at test points
    x_test_norm = normalise_data(test.x, train.x_mean, train.x_std)
    y_pred_mean_norm, y_pred_var_noiseless_norm = m.predict_f(x_test_norm)
    y_pred_mean = unnormalise_data(y_pred_mean_norm, train.y_mean, train.y_std)
    y_pred_std_noiseless = unnormalise_data(np.sqrt(y_pred_var_noiseless_norm), 0.0, train.y_std)

    # generate 10 samples from posterior
    y_pred_mean_samples_norm = m.predict_f_samples(x_test_norm, 10)  # shape (10, 100, 1)
    y_pred_mean_samples = unnormalise_data(y_pred_mean_samples_norm, train.y_mean, train.y_std)

    model_noise_std = unnormalise_data(lik_var**0.5, 0.0, train.y_std)
    y_pred_std = np.sqrt(y_pred_std_noiseless**2 + model_noise_std**2)

    plot_regression(
        train, test, y_pred_mean, y_pred_std_noiseless, y_pred_std, y_pred_mean_samples, title, exp_name
    )
    print(f'model_std: {model_noise_std}, pred_std: {y_pred_std_noiseless.mean()}')


def gpflow_test_step(m, test, train, loss_func='mse', lik_var=None):
    x_test_norm = normalise_data(test.x, train.x_mean, train.x_std)
    y_pred_mean, y_pred_std = gpflow_get_regression_results(
        m, x_test_norm, train, lik_var
    )
    if loss_func == 'mse':
        mse = tf.keras.losses.MeanSquaredError()
        loss = mse(y_pred_mean, test.y).numpy()
        print('\nTest set: MSE: {}'.format(loss))
    elif loss_func == 'gnll':
        loss = np.mean(0.5 * (2*np.log(y_pred_std) + (y_pred_mean - test.y)**2 / y_pred_std**2 + np.log(2*np.pi)))
        print('\nTest set: GNLL: {}'.format(loss))
    return loss


def gpflow_mse_train(m, train, lik_var=None):
    y_pred_mean, _ = gpflow_get_regression_results(
        m, train.x, None, lik_var, False
    )
    mse = tf.keras.losses.MeanSquaredError()
    loss = mse(y_pred_mean, train.y).numpy()
    print('\nTrain set: MSE: {}'.format(loss))
    return loss


if __name__ == '__main__':
    train_loader, test_loader, train, test, noise_std = create_regression_dataset()

    x_train = unnormalise_data(train.x, train.x_mean, train.x_std)
    y_train = unnormalise_data(train.y, train.y_mean, train.y_std)
    # first for the total uncertainty (model/epistemic + data/aleatoric)
    plt.figure(figsize=(12, 6))
    plt.plot(test.x, test.y)
    plt.plot(x_train, y_train, "kx", mew=2, label='noisy sample points')
    plt.fill_between(
        test.x[:,0],
        # y_pred_mean[:,0] - 1.96 * y_pred_std[:,0],  # 95% confidence interval
        # y_pred_mean[:,0] + 1.96 * y_pred_std[:,0],
        test.y[:,0] - 1.96 * 0.1,  # 95% confidence interval
        test.y[:,0] + 1.96 * 0.1,
        color="C0",
        alpha=0.2,
        label='noise uncertainity (95% ci)'
    )
    plt.plot(test.x, test.y, color='orange', label='sample function')
    plt.legend()
    plt.title('GP Sample Function Regression Dataset')
    plt.show()
