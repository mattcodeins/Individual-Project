import numpy as np
import torch
import matplotlib.pyplot as plt
import csv
import os
# import torch.nn as nn
# import torchviz as tv

from modules.bnn.modules.linear import BayesLinear
from modules.bnn.modules.emp_linear import EmpBayesLinear
from modules.bnn.modules.ext_emp_linear import ExtEmpBayesLinear


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def to_numpy(x):
    return x.detach().cpu().numpy()  # convert a torch tensor to a numpy array


def training_loop(model, N_epochs, opt, nelbo, train_loader, test_loader, filename, device=device):
    model.train()
    logs = []
    for i in range(N_epochs):
        # train step is whole training dataset (minibatched inside function)
        loss, nll, kl = train_step(model, opt, nelbo, train_loader, device)
        # logging depends on learnable parameter (this is not an eloquent solution)
        first_layer = list(model.modules())[1]
        if isinstance(first_layer, (BayesLinear)):
            pass
        elif isinstance(first_layer, (EmpBayesLinear)):
            prior_std = np.log(1 + np.exp(to_numpy(model._prior_std_param)))
            logs.append([i+1, to_numpy(loss), to_numpy(nll), to_numpy(kl), prior_std])
            print("Epoch {}, nelbo={}, nll={}, kl={}, prior_std={}".format(
                logs[-1][0], logs[-1][1], logs[-1][2], logs[-1][3], logs[-1][4]
            ))
        elif isinstance(first_layer, (ExtEmpBayesLinear)):
            fl_prior_avg_std = (to_numpy(first_layer.prior_weight_std) + to_numpy(first_layer.prior_bias_std))/2
            logs.append([i+1, to_numpy(loss), to_numpy(nll),
                         to_numpy(kl), to_numpy(model.prior_mean), fl_prior_avg_std])
            print("Epoch {}, nelbo={}, nll={}, kl={}, prior_mean={}, avg prior_std (1st layer)={}".format(
                logs[-1][0], logs[-1][1], logs[-1][2], logs[-1][3], logs[-1][4], logs[-1][5]
            ))

        if (i+1) % 1 == 0:
            torch.save(model.state_dict(), f'./saved_models/{filename}')
            write_logs_to_file(logs, filename)
            model.eval()
            correct = 0
            with torch.no_grad():
                for x_test, y_test in test_loader:
                    m, _ = predict(model, x_test.reshape((x_test.shape[0],-1)), K=50)
                    pred = m.max(1, keepdim=True)[1]
                    correct += pred.eq(y_test.to(device).view_as(pred)).sum()

            test_acc = 100. * correct / len(test_loader.dataset)
            print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
                correct, len(test_loader.dataset), test_acc
            ))

    logs = np.array(logs)
    return logs


def train_step(model, opt, nelbo, dataloader, device=device):
    tloss, tnll, tkl = 0,0,0
    for _, (x, y) in enumerate(dataloader):
        opt.zero_grad()
        batch_size = x.shape[0]
        minibatch_ratio = batch_size / len(dataloader.dataset)
        x = x.to(device).reshape((batch_size, -1))
        y = y.to(device)
        y_pred = model(x)
        loss, nll, kl = nelbo(model, (y_pred, y), minibatch_ratio)
        loss.backward(retain_graph=True)
        opt.step()
        tloss += loss; tnll += nll; tkl += kl
        # g = tv.make_dot(loss, params=dict(model.named_parameters()))
        # g.filename = 'network.dot'
        # g.render()
    return tloss, tnll, tkl


def predict(model, x_test, K=1, device=device):  # Monte Carlo sampling using K samples
    y_pred = []
    for _ in range(K):
        y_pred.append(model(x_test.to(device)))
    # shape (K, batch_size, y_dim) or (batch_size, y_dim) if K = 1
    y_pred = torch.stack(y_pred, dim=0).squeeze(0)
    return y_pred.mean(0), y_pred.std(0)


def plot_training_loss(logs):
    fig, axs = plt.subplots(2, 2)
    axs[0,0].plot(np.arange(logs.shape[0]), -logs[:,1], 'r-')
    axs[0,1].plot(np.arange(logs.shape[0]), logs[:,2], 'r-')
    axs[1,0].plot(np.arange(logs.shape[0]), logs[:,3], 'r-')
    axs[1,1].plot(np.arange(logs.shape[0]), logs[:,4], 'r-')

    axs[1,0].set_xlabel('epoch')
    axs[1,1].set_xlabel('epoch')

    axs[0,0].set_title('elbo')
    axs[0,1].set_title('nll')
    axs[1,0].set_title('kl')
    axs[1,1].set_title('prior std')

    plt.show()


def write_logs_to_file(logs, name):
    with open(f"./results/{name}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'elbo', 'nll', 'kl', 'prior_std'])
        writer.writerows(logs)
        f.close()


def uniquify(name):
    counter = 1
    while os.path.exists("results/" + name + ".csv"):
        name = name + "_" + str(counter)
        counter += 1
    return name
