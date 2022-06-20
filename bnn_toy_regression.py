import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import datasets.toy_regression as d
from modules.bnn.modules.linear import make_linear_bnn
from modules.bnn.modules.loss import GaussianKLLoss, nELBO
from modules.utils import to_numpy


def train_step(model, opt, nelbo, log_noise_var, dataloader, N_data, device):
    for _, (x, y) in enumerate(dataloader):
        x = x.to(device); y = y.to(device)
        y_pred = model(x)
        noise_var = torch.exp(log_noise_var)*torch.ones(N_data)
        loss, nll, kl = nelbo(model, (y, y_pred, noise_var))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss, nll, kl


def predict(bnn, x_test, K=1):  # Monte Carlo sampling using K samples
    y_pred = []
    for _ in range(K):
        y_pred.append(bnn(x_test))
    # shape (K, batch_size, y_dim) or (batch_size, y_dim) if K = 1
    return torch.stack(y_pred, dim=0).squeeze(0)


if __name__ == "__main__":

    # create dataset
    N_data = 100; noise_std = 0.1
    dataloader, dataset, x_train, y_train, x_test, y_test = d.create_regression_dataset(N_data, noise_std)

    # create bnn
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    x_dim, y_dim = x_train.shape[1], y_train.shape[1]
    h_dim = 50
    layer_sizes = [x_dim, h_dim, h_dim, y_dim]
    activation = nn.GELU()
    layer_kwargs = {'prior_weight_std': 1,
                    'prior_bias_std': 1,
                    'sqrt_width_scaling': False,
                    'init_std': 0.05,
                    'device': device}
    model = make_linear_bnn(layer_sizes, activation=activation, **layer_kwargs)
    log_noise_var = torch.ones(size=(), device=device)*-3.0  # Gaussian likelihood
    print("BNN architecture: \n", model)

    # plot the BNN prior in function space
    K = 50  # number of Monte Carlos samples used in test time
    x_test_norm = d.normalise_data(x_test, dataset.x_mean, dataset.x_std)
    x_test_norm = torch.tensor(x_test_norm, ).float().to(device)

    y_pred_mean, y_pred_std_noiseless = d.get_regression_results(model, x_test_norm, K, predict, dataset)
    model_noise_std = d.unnormalise_data(to_numpy(torch.exp(0.5*log_noise_var)), 0.0, dataset.y_std)
    y_pred_std = np.sqrt(y_pred_std_noiseless ** 2 + model_noise_std ** 2)
    d.plot_regression(x_train, y_train, x_test, y_test, y_pred_mean, y_pred_std_noiseless, y_pred_std,
                      title='BNN init (before training, MFVI)')
    print(model_noise_std, noise_std, y_pred_std_noiseless.mean())

    # training hyperparameters
    learning_rate = 1e-4
    params = list(model.parameters()) + [log_noise_var]
    opt = torch.optim.Adam(params, lr=learning_rate)
    # hyper-parameters of training
    N_epochs = 50000

    gnll_loss = nn.GaussianNLLLoss(full=True, reduction='sum')
    kl_loss = GaussianKLLoss()
    nelbo = nELBO(nll_loss=gnll_loss, kl_loss=kl_loss)

    # training loop
    model.train()
    logs = []
    for i in range(N_epochs):
        loss, nll, kl = train_step(
            model, opt, nelbo, log_noise_var, dataloader, N_data=len(dataloader.dataset), device=device
        )
        logs.append([to_numpy(nll), to_numpy(kl), to_numpy(loss), to_numpy(nll)/to_numpy(kl)])
        if (i+1) % 100 == 0:
            print("Epoch {}, nll={}, kl={}, nelbo={}, ratio={}"
                  .format(i+1, logs[-1][0], logs[-1][1], logs[-1][2], logs[-1][3]))
    logs = np.array(logs)

    # plot the training curve
    def plot_training_loss(logs):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.plot(np.arange(logs.shape[0]), logs[:, 0], 'r-')
        ax2.plot(np.arange(logs.shape[0]), logs[:, 1], 'r-')
        ax1.set_xlabel('epoch')
        ax2.set_xlabel('epoch')
        ax1.set_title('nll')
        ax2.set_title('kl')
        plt.show()

    plot_training_loss(logs)

    y_pred_mean, y_pred_std_noiseless = d.get_regression_results(model, x_test_norm, K, predict, dataset)
    model_noise_std = d.unnormalise_data(to_numpy(torch.exp(0.5*log_noise_var)), 0.0, dataset.y_std)
    y_pred_std = np.sqrt(y_pred_std_noiseless ** 2 + model_noise_std ** 2)
    d.plot_regression(x_train, y_train, x_test, y_test, y_pred_mean, y_pred_std_noiseless, y_pred_std,
                      title='BNN approx. posterior (MFVI)')
    print(model_noise_std, noise_std, y_pred_std_noiseless.mean())
