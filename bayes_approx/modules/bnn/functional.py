import torch

from modules.bnn.modules.linear import BayesLinear
from modules.bnn.modules.emp_linear import EmpBayesLinear
from modules.bnn.modules.ext_emp_linear import ExtEmpBayesLinear
from modules.bnn.modules.mlg_linear import MLGBayesLinear
from modules.bnn.modules.cm_linear import CMBayesLinear
from modules.bnn.modules.cmv_linear import CMVBayesLinear


def _gaussian_kl(mean_q, std_q, mean_p, std_p):
    """
    KL divergence between diagonal Gaussian distribtuions.
    (We can simply calculate the sum of the kl between independent single variate gaussians)
    """
    kl = (-0.5 + torch.log(std_p) - torch.log(std_q) +
          (torch.pow(std_q, 2) + torch.pow(mean_q-mean_p, 2)) /
          (2*torch.pow(std_p, 2)))
    return kl.sum()


def gaussian_kl_loss(model):
    """
    KL divergence between approximate posterior and prior, assuming both are diagonal Gaussian.
    This is the closed form complexity cost in Weight Uncertainty in Neural Networks.
    """
    device = torch.device("cuda" if next(model.parameters()).is_cuda else "cpu")
    kl = torch.Tensor([0]).to(device)
    kl_sum = torch.Tensor([0]).to(device)
    for m in model.modules():
        if (isinstance(m, (BayesLinear))
                or isinstance(m, (EmpBayesLinear))
                or isinstance(m, (ExtEmpBayesLinear))
                or isinstance(m, (MLGBayesLinear))):
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
    nll = nll_loss(*loss_args)
    kl = kl_loss(model) * minibatch_ratio
    nelbo = nll + kl

    return nelbo, nll, kl


def _cml(post_mean, post_std, alpha, prior_std):
    gamma = prior_std**2
    alpha_reg = gamma / (gamma + alpha)
    cml = ((post_std**2 + alpha_reg*post_mean**2) / gamma + torch.log(gamma)
           - 2*torch.log(post_std) - torch.log(alpha_reg) - 1)
    return cml.sum()/2


def cm_loss(model):
    device = torch.device("cuda" if next(model.parameters()).is_cuda else "cpu")
    cml = torch.Tensor([0]).to(device)
    cml_sum = torch.Tensor([0]).to(device)
    for m in model.modules():
        if (isinstance(m, (CMBayesLinear))):
            cml = _cml(m.weight_mean, m.weight_std, m.prior_mean_hyperstd, m.prior_weight_std)
            cml_sum += cml
            if m.bias:
                cml = _cml(m.bias_mean, m.bias_std, m.prior_mean_hyperstd, m.prior_bias_std)
                cml_sum += cml
    return cml_sum


def _cmvl(post_mean, post_std, alpha, beta, delta):
    cmvl = ((alpha + 1/2)*torch.log(beta + delta*post_mean**2/2 + post_std**2/2)
            - torch.log(post_std) - torch.lgamma(alpha + 1/2) + torch.lgamma(alpha)
            - alpha*torch.log(beta) - torch.log(delta))
    return cmvl.sum()


def cmv_loss(model):
    device = torch.device("cuda" if next(model.parameters()).is_cuda else "cpu")
    cmvl = torch.Tensor([0]).to(device)
    cmvl_sum = torch.Tensor([0]).to(device)
    for m in model.modules():
        if (isinstance(m, (CMVBayesLinear))):
            cmvl = _cmvl(m.weight_mean, m.weight_std, m.hyperprior_alpha, m.hyperprior_beta, m.hyperprior_delta)
            cmvl_sum += cmvl
            if m.bias:
                cmvl = _cmvl(m.bias_mean, m.bias_std, m.hyperprior_alpha, m.hyperprior_beta, m.hyperprior_delta)
                cmvl_sum += cmvl
    return cmvl_sum


def prior_regularisation(model):
    device = torch.device("cuda" if next(model.parameters()).is_cuda else "cpu")
    prior_dif = torch.Tensor([0]).to(device)
    prior_dif_sum = torch.Tensor([0]).to(device)
    for m in model.modules():
        if (isinstance(m, (BayesLinear))
                or isinstance(m, (EmpBayesLinear))
                or isinstance(m, (ExtEmpBayesLinear))
                or isinstance(m, (MLGBayesLinear))):
            prior_dif = _gaussian_kl(m.weight_mean, m.weight_std, m.prior_weight_mean, m.prior_weight_std)
            prior_dif_sum += kl
            if m.bias:
                kl = _gaussian_kl(m.bias_mean, m.bias_std, m.prior_bias_mean, m.prior_bias_std)
                kl_sum += kl
    return kl_sum


def maximum_a_posteriori(model, loss_args, minibatch_ratio, nll_loss):
    device = torch.device("cuda" if next(model.parameters()).is_cuda else "cpu")
    map_loss = torch.Tensor([0]).to(device)
    nll = nll_loss(*loss_args)
    prior_loss = prior_regularisation(model)
    map_loss = nll + prior_loss

    return map_loss, nll, prior_loss


def MLG_gaussian_kl(model):
    """
    KL divergence is between the epsilon posterior and prior (standard normal)
    """
    device = torch.device("cuda" if next(model.parameters()).is_cuda else "cpu")
    kl = torch.Tensor([0]).to(device)
    kl_sum = torch.Tensor([0]).to(device)
    for m in model.modules():
        if isinstance(m, (MLGBayesLinear)):
            kl = _gaussian_kl(m.weight_mean, m.weight_std, torch.tensor(0), torch.tensor(1))
            kl_sum += kl
            if m.bias:
                kl = _gaussian_kl(m.bias_mean, m.bias_std, torch.tensor(0), torch.tensor(1))
                kl_sum += kl
    return kl_sum


def MLG_approximate_scheme(model, loss_args, minibatch_ratio, nll_loss, kl_loss):
    device = torch.device("cuda" if next(model.parameters()).is_cuda else "cpu")
    nelbo = torch.Tensor([0]).to(device)
    nll = nll_loss(*loss_args)
    kl = kl_loss(model) * minibatch_ratio
    nelbo = nll + kl

    return nelbo, nll, kl