import numpy as np
import torch
import matplotlib.pyplot as plt
import csv
import os
import torch.nn.functional as F
# import torchviz as tv

from modules.bnn.modules.linear import BayesLinear
from modules.bnn.modules.emp_linear import EmpBayesLinear
from modules.bnn.modules.ext_emp_linear import ExtEmpBayesLinear
from modules.bnn.modules.cm_linear import CMBayesLinear
from modules.bnn.modules.cmv_linear import CMVBayesLinear
from modules.bnn.modules.mlg_linear import MLGBayesLinear


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'


def to_numpy(x):
    return x.detach().cpu().numpy()  # convert a torch tensor to a numpy array


# Methods below are for single ELBO objectives
def training_loop(model, N_epochs, opt, lr_sch, nelbo, train_loader, test_loader, beta,
                  test_step=None, train=None, filename=None, device=device):
    model.train()
    logs = []
    losses = []; nlls = []; kls = []
    for i in range(N_epochs):
        # train step is whole training dataset (minibatched inside function)
        loss, nll, kl = train_step(model, opt, nelbo, train_loader, beta, device)
        losses.append(float(loss))
        nlls.append(float(nll))
        kls.append(float(kl))
        lr_sch.step()
        if (i+1) % 1000 == 0:
            # print(beta)
            avgloss = sum(losses[-1000:])/1000
            avgnll = sum(nlls[-1000:])/1000
            avgkl = sum(kls[-1000:])/1000
            logs = logging(model, logs, i, avgloss, avgnll, avgkl, beta)
            logs[-1].append(to_numpy(test_step(model, test_loader, train, predict, log_lik_var=beta)))
            if filename is not None:
                torch.save(model.state_dict(), f'bayes_approx/saved_models/{filename}.pt')
                write_logs_to_file(logs, filename)

    logs = np.array(logs)
    return logs


def train_step(model, opt, nelbo, dataloader, log_noise_var, device=device):
    """
    Minibatch gradient descent through entire training batch.
    """
    tloss, tnll, tkl = 0,0,0
    for _, (x, y) in enumerate(dataloader):
        opt.zero_grad()
        batch_size = x.shape[0]
        minibatch_ratio = batch_size / len(dataloader.dataset)
        x = x.to(device).reshape((batch_size, -1))
        y = y.to(device)
        y_pred = model(x)
        if log_noise_var is None:
            loss_args = (y_pred, y)
        else:
            beta = torch.exp(log_noise_var)*torch.ones(x.shape[0], device=device)
            loss_args = (y_pred, y, beta)
        loss, nll, kl = nelbo(model, loss_args, minibatch_ratio)
        loss.backward()
        opt.step()
        tloss += loss; tnll += nll; tkl += kl
        # g = tv.make_dot(loss, params=dict(model.named_parameters()))
        # g.filename = 'network.dot'
        # g.render()
    return tloss, tnll, tkl


# def classif_test_step(model, nelbo, dataloader, device=device):
#     """
#     Calculate accuracy on test set.
#     """
#     model.eval()
#     N_data = len(dataloader.dataset)
#     tloss, tnll, tkl = 0,0,0
#     correct = 0
#     with torch.no_grad():
#         for x_test, y_test in dataloader:
#             m, _ = predict(model, x_test.reshape((x_test.shape[0],-1)), K=50)
#             loss, nll, kl = nelbo(model, (m, y_test), 1/10)
#             tloss += loss; tnll += nll; tkl += kl
#             pred = m.max(1, keepdim=True)[1]
#             correct += pred.eq(y_test.to(device).view_as(pred)).sum()
#     print(tloss)
#     print(tnll)
#     print(tkl)
#     test_acc = 100. * correct / N_data
#     print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
#         correct, N_data, test_acc
#     ))


def predict(model, x, K=50, device=device):
    """
    Monte Carlo sampling of BNN using K samples.
    """
    if K == 1:
        return model(x.to(device)), torch.tensor([0])

    y_pred = []
    for _ in range(K):
        y_pred.append(model(x.to(device)))
    # shape (K, batch_size, y_dim) or (batch_size, y_dim) if K = 1
    y_pred = torch.stack(y_pred, dim=0).squeeze(0)
    return y_pred.mean(0), y_pred.std(0)


# # Methods below are for seperate training and model selection objectives
# def ml_sep_training_loop(model, N_epochs, nn_opt, ml_opt, map_loss, ml_loss,
#                          train_loader, test_loader, filename, device=device):
#     model.train()
#     logs = []
#     for i in range(N_epochs):
#         # train step is whole training dataset (minibatched inside function)
#         loss, nll, prior_reg = train_step(model, nn_opt, map_loss, train_loader, device)
#         ml_loss = ml_train_step(model, ml_opt, ml_loss, train_loader, device)

#         logs = logging(model, logs, i, loss, nll, prior_reg, beta, ml_loss)

#         if (i+1) % 10 == 0:
#             # torch.save(model.state_dict(), f'saved_models/{filename}.pt')
#             # write_logs_to_file(logs, filename)
#             test_step(model, map_loss, test_loader, device=device)

#     logs = np.array(logs)
#     return logs


# def ml_train_step(model, opt, ml_approx, dataloader, device=device):
#     """
#     Minibatch gradient descent through entire training batch.
#     """
#     log_q = ml_approx(model)


#     for _, (x, y) in enumerate(dataloader):
#         opt.zero_grad()
#         batch_size = x.shape[0]
#         minibatch_ratio = batch_size / len(dataloader.dataset)
#         x = x.to(device).reshape((batch_size, -1))
#         y = y.to(device)
#         y_pred = model(x)
#         loss, nll, kl = nelbo(model, (y_pred, y), minibatch_ratio)
#         loss.backward()
#         opt.step()
#         tloss += loss; tnll += nll; tkl += kl
#         # g = tv.make_dot(loss, params=dict(model.named_parameters()))
#         # g.filename = 'network.dot'
#         # g.render()
#     return tloss, tnll, tkl


# General functions below
def logging(model, logs, i, loss, nll, prior_reg, beta=None, ml_loss=None):
    """
    What we want to log depends on the learnable parameters.
    """
    loss_logs = [i+1, loss, nll, prior_reg]
    if ml_loss is not None:
        loss_logs.append(to_numpy(ml_loss))
    first_layer = list(model.modules())[1]
    if isinstance(first_layer, (BayesLinear)):
        logs.append(loss_logs)
        print("Epoch {}, nelbo={}, nll={}, kl={}".format(
            logs[-1][0], logs[-1][1], logs[-1][2], logs[-1][3]
        ))
    elif isinstance(first_layer, (EmpBayesLinear)):
        prior_std = np.log(1 + np.exp(to_numpy(model._prior_std_param)))
        if beta.requires_grad:
            logs.append(loss_logs + [prior_std] + [*to_numpy(beta)])
            print("Epoch {}, nelbo={}, nll={}, kl={}, prior_std={}, beta={}".format(
                logs[-1][0], logs[-1][1], logs[-1][2], logs[-1][3], logs[-1][4], logs[-1][5]
            ))
        else:
            logs.append(loss_logs + [prior_std])
            print("Epoch {}, nelbo={}, nll={}, kl={}, prior_std={}".format(
                logs[-1][0], logs[-1][1], logs[-1][2], logs[-1][3], logs[-1][4]
            ))
    elif isinstance(first_layer, (ExtEmpBayesLinear)):
        fl_prior_avg_std = (to_numpy(first_layer.prior_weight_std) + to_numpy(first_layer.prior_bias_std))/2
        if ml_loss is None:
            logs.append(loss_logs + [to_numpy(model.prior_mean)] + [fl_prior_avg_std])
            print("Epoch {}, nelbo={}, nll={}, kl={}, prior_mean={}, avg prior_std (1st layer)={}".format(
                logs[-1][0], logs[-1][1], logs[-1][2], logs[-1][3], logs[-1][4], logs[-1][5]
            ))
        else:
            logs.append([i+1, to_numpy(loss), to_numpy(nll), to_numpy(prior_reg), to_numpy(ml_loss),
                        to_numpy(model.prior_mean), fl_prior_avg_std])
            print("Epoch {}, nelbo={}, nll={}, prior_reg={}, prior_mean={}, avg prior_std (1st layer)={}".format(
                logs[-1][0], logs[-1][1], logs[-1][2], logs[-1][3], logs[-1][4], logs[-1][5], logs[-1][6]
            ))
    elif isinstance(first_layer, (CMBayesLinear)):
        # prior_mean_hyperstd = np.log(1 + np.exp(to_numpy(model._prior_mean_hyperstd_param)))
        # logs.append(loss_logs + [prior_mean_hyperstd])
        logs.append(loss_logs + [np.log(1 + np.exp(to_numpy(model._prior_mean_hyperstd_param)))])
        print("Epoch {}, nelbo={}, nll={}, kl={}, prior_mean_hyperstd={}".format(
            logs[-1][0], logs[-1][1], logs[-1][2], logs[-1][3], logs[-1][4]
        ))
    elif isinstance(first_layer, (CMVBayesLinear)):
        # prior_mean_hyperstd = np.log(1 + np.exp(to_numpy(model._prior_mean_hyperstd_param)))
        # logs.append(loss_logs + [prior_mean_hyperstd])
        logs.append(loss_logs + [np.log(1 + np.exp(to_numpy(model._hyperprior_alpha_param)))]
                              + [np.log(1 + np.exp(to_numpy(model._hyperprior_beta_param)))]
                              + [np.log(1 + np.exp(to_numpy(model._hyperprior_delta_param)))])
        print("Epoch {}, nelbo={}, nll={}, kl={}, alpha={}, beta={}, delta={}".format(
            logs[-1][0], logs[-1][1], logs[-1][2], logs[-1][3], logs[-1][4], logs[-1][5], logs[-1][6]
        ))
    if isinstance(first_layer, (MLGBayesLinear)):
        logs.append(loss_logs)
        print("Epoch {}, nelbo={}, nll={}, kl={}".format(
            logs[-1][0], logs[-1][1], logs[-1][2], logs[-1][3]
        ))

    return logs


def plot_training_loss(logs, exp_name=None):
    fig, axs = plt.subplots(2, 2)
    axs[0,0].plot(np.arange(logs.shape[0]), logs[:,1], 'r-')
    axs[0,1].plot(np.arange(logs.shape[0]), logs[:,2], 'r-')
    axs[1,0].plot(np.arange(logs.shape[0]), logs[:,3], 'r-')
    # axs[1,1].plot(np.arange(logs.shape[0]), logs[:,4], 'r-')

    axs[1,0].set_xlabel('epoch')
    axs[1,1].set_xlabel('epoch')

    axs[0,0].set_title('elbo')
    axs[0,1].set_title('nll')
    axs[1,0].set_title('kl')
    # axs[1,1].set_title('prior std')

    plt.show()
    if exp_name is not None:
        plt.savefig(f'./figures/{exp_name}train.png', bbox_inches='tight')


def plot_training_loss_together(logs, title='training curve', exp_name=None):
    x = np.arange(logs.shape[0])*1000
    plt.plot(x, logs[:,1], 'r-', label='nelbo')
    plt.fill_between(x, logs[:,2], label='nll')
    plt.fill_between(x, logs[:,1], logs[:,2], label='kl')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.ylim(0, 3000)
    # plt.xlim(0, 60000)
    plt.title(title)
    plt.legend()
    if exp_name is not None:
        plt.savefig(f'./figures/{exp_name}train.png', bbox_inches='tight')
    plt.show()


def write_logs_to_file(logs, name):
    with open(f"./bayes_approx/results/{name}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'elbo', 'nll', 'kl', 'prior_std'])
        writer.writerows(logs)
        f.close()


def uniquify(name):
    if name is None:
        return None
    counter = 1
    while os.path.exists("bayes_approx/results/" + name + ".csv"):
        name = name + "_" + str(counter)
        counter += 1
    return name


def predict_wo_var(model, x_test, device=device, **kwargs):
    return model(x_test.to(device), variance=False)


def load_model(model, model_name):
    path = f"bayes_approx/saved_models/{model_name}.pt"
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    model.eval()
    return model


def load_logs(model_name):
    path = f"bayes_approx/results/{model_name}.csv"
    logs = np.genfromtxt(path, delimiter=',', skip_header=1)
    return logs
