import torch

from modules import BayesLinear


def _gaussian_kl_loss(mean_1, std_1, mean_2, std_2):
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
    This is the complexity cost in Weight Uncertainty in Neural Networks.
    """
    device = torch.device("cuda" if next(model.parameters()).is_cuda else "cpu")
    kl = torch.Tensor([0]).to(device)
    kl_sum = torch.Tensor([0]).to(device)

    for m in model.modules():
        if isinstance(m, (BayesLinear)):
            kl = _gaussian_kl_loss(m.weight_mean, m.weight_std, m.prior_weight_mean, m.prior_weight_std)
            kl_sum += kl

            if m.bias:
                kl = _gaussian_kl_loss(m.bias_mean, m.bias_std, m.prior_bias_mean, m.prior_bias_std)
                kl_sum += kl

    return kl_sum


def elbo(model, loss_args, nll_loss, kl_loss):
    """
    An method for calculating KL divergence of whole layers in the model.

    nll_loss is reduce by sum, but then we divide by N, so we find mean over data, but true log likelihood over
    multivariate distribution
    """
    device = torch.device("cuda" if next(model.parameters()).is_cuda else "cpu")
    elbo = torch.Tensor([0]).to(device)
    N_data = loss_args[0].shape[0]
    nll = nll_loss(*loss_args) / N_data
    kl = kl_loss(model)
    elbo = nll + kl

    return elbo, nll, kl
