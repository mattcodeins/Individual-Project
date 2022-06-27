import torch
import torch.nn as nn
import numpy as np
# import matplotlib.pyplot as plt
# import torchvision

from modules.bnn.modules.emp_linear import make_linear_emp_bnn
from modules.bnn.modules.loss import GaussianKLLoss, nELBO
from modules.utils import to_numpy

import datasets.mnist as d


def train_step(model, opt, nelbo, dataloader, device):
    tloss, tnll, tkl = 0,0,0
    for _, (x, y) in enumerate(dataloader):
        opt.zero_grad()
        batch_size = x.shape[0]
        minibatch_ratio = batch_size / len(dataloader.dataset)
        x = x.to(device).reshape((batch_size, 784)); y = y.to(device)
        y_pred = model(x)
        loss, nll, kl = nelbo(model, (y_pred, y), minibatch_ratio)
        loss.backward(retain_graph=True)
        opt.step()
        tloss += loss; tnll += nll; tkl += kl
    return tloss, tnll, tkl


def predict(model, x_test, K=1):  # Monte Carlo sampling using K samples
    y_pred = []
    for _ in range(K):
        y_pred.append(model(x_test.to(device)))
    # shape (K, batch_size, y_dim) or (batch_size, y_dim) if K = 1
    y_pred = torch.stack(y_pred, dim=0).squeeze(0)
    return y_pred.mean(0), y_pred.std(0)


if __name__ == "__main__":
    torch.manual_seed(1)

    # create dataset
    batch_size_train = 64
    batch_size_test = 1000
    train_loader, test_loader = d.import_n_mnist(batch_size_train, batch_size_test)

    # create bnn
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    x_dim, y_dim = 784, 10
    h1_dim, h2_dim = 128, 64
    layer_sizes = [x_dim, h1_dim, h2_dim, y_dim]
    activation = nn.GELU()
    model = make_linear_emp_bnn(layer_sizes, activation=activation, device=device)
    log_noise_var = torch.ones(size=(), device=device)*-3.0  # Gaussian likelihood
    print("BNN architecture: \n", model)

    # training hyperparameters
    learning_rate = 1e-4
    params = list(model.parameters())  # + [log_noise_var]
    opt = torch.optim.Adam(params, lr=learning_rate)
    # hyper-parameters of training
    N_epochs = 5000

    cross_entropy_loss = nn.CrossEntropyLoss(reduction='sum')
    kl_loss = GaussianKLLoss()
    nelbo = nELBO(nll_loss=cross_entropy_loss, kl_loss=kl_loss)

    # training loop
    model.train()
    logs = []
    for i in range(N_epochs):
        # train step is whole training dataset (minibatched inside function)
        loss, nll, kl = train_step(model, opt, nelbo, train_loader, device=device)
        logs.append([to_numpy(nll), to_numpy(kl), to_numpy(loss), to_numpy(nll)/to_numpy(kl)])

        if (i+1) % 1 == 0:
            model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for x_test, y_test in test_loader:
                    m, v = predict(model, x_test.reshape((-1,784)), K=50)
                    pred = m.max(1, keepdim=True)[1]
                    correct += pred.eq(y_test.to(device).view_as(pred)).sum()
            print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
                correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))

            logs.append([to_numpy(nll), to_numpy(kl), to_numpy(loss),
                         to_numpy(torch.log(1 + torch.exp(model._prior_std)))])

            print("Epoch {}, nll={}, kl={}, nelbo={}, prior_std={}"
                  .format(i+1, logs[-1][0], logs[-1][1], logs[-1][2], logs[-1][3]))
    logs = np.array(logs)
