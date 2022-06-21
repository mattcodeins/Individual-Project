import torch

from modules.bnn.modules.emp_linear import EmpBayesLinear

from .modules.linear import BayesLinear


def _gaussian_kl(mean_1, std_1, mean_2, std_2):
    """
    KL divergence between Gaussian distribtuions.

    Arguments:
        mu_0 (tensor) : mean of normal distribution.
        log_sigma_0 (tensor): log(standard deviation of normal distribution).
        mu_1 (tensor): mean of normal distribution.
        log_sigma_1 (tensor): log(standard deviation of normal distribution).
    """
    kl = (-0.5 + torch.log(std_2) - torch.log(std_1) +
          (torch.pow(std_1, 2) + torch.pow(mean_1-mean_2, 2)) /
          (2*torch.pow(std_2, 2)))
    return kl.sum()


def gaussian_kl_loss(model):
    """
    KL divergence between approximate posterior and prior, assuming both are Gaussian, of all layers in the model.
    This is the closed form complexity cost in Weight Uncertainty in Neural Networks.
    """
    device = torch.device("cuda" if next(model.parameters()).is_cuda else "cpu")
    kl = torch.Tensor([0]).to(device)
    kl_sum = torch.Tensor([0]).to(device)

    for m in model.modules():
        if isinstance(m, (BayesLinear)) or isinstance(m, (EmpBayesLinear)):
            kl = _gaussian_kl(m.weight_mean, m.weight_std, m.prior_weight_mean, m.prior_weight_std)
            kl_sum += kl

            if m.bias:
                kl = _gaussian_kl(m.bias_mean, m.bias_std, m.prior_bias_mean, m.prior_bias_std)
                kl_sum += kl

    return kl_sum


def nelbo(model, loss_args, minibatch_ratio, nll_loss, kl_loss):
    """
    kl divided by number of minibatches
    """
    device = torch.device("cuda" if next(model.parameters()).is_cuda else "cpu")
    nelbo = torch.Tensor([0]).to(device)
    # N_data = loss_args[0].shape[0]
    nll = nll_loss(*loss_args)
    kl = kl_loss(model) * minibatch_ratio
    nelbo = nll + kl

    return nelbo, nll, kl
