import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


class EmpBayesLinear(nn.Module):
    """Applies a linear transformation to the incoming data: y = xW^T + b, where
       the weight W and bias b are sampled from the approximate q distribSution.
    """
    def __init__(self, in_features, out_features, _prior_std, bias=True,
                 init_std=0.05, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype, 'requires_grad': True}
        super(EmpBayesLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # approximate posterior parameters (Gaussian)
        self.weight_mean = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self._weight_std_param = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        self.bias = bias
        if self.bias:
            self.bias_mean = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            self._bias_std_param = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias_mean', None)
            self.register_parameter('_bias_std_param', None)
        self.reset_parameters(init_std)

        # prior parameters (Gaussian)
        prior_mean = 0.0
        # prior parameters are registered as constants
        self.register_buffer('prior_weight_mean', torch.full_like(self.weight_mean, prior_mean))
        # self.register_buffer('prior_weight_std', torch.full_like(self._weight_std_param, prior_weight_std))
        self.prior_weight_std = torch.log(1 + torch.exp(_prior_std))
        if self.bias:
            self.register_buffer('prior_bias_mean', torch.full_like(self.bias_mean, prior_mean))
            # self.register_buffer('prior_bias_std', torch.full_like(self._bias_std_param, prior_bias_std))
            self.prior_bias_std = torch.log(1 + torch.exp(_prior_std))

        else:
            self.register_buffer('prior_bias_mean', None)
            self.register_buffer('prior_bias_std', None)

    def extra_repr(self):
        repr = "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )
        weight_std = self.prior_weight_std.data.flatten()[0]
        if torch.allclose(weight_std, self.prior_weight_std):
            repr += f", weight prior std={weight_std.item():.2f}"
        bias_std = self.prior_bias_std.flatten()[0]
        if torch.allclose(bias_std, self.prior_bias_std):
            repr += f", bias prior std={bias_std.item():.2f}"
        return repr

    def reset_parameters(self, init_std=0.05):
        # nn.init.kaiming_uniform_(self.weight_mean, a=math.sqrt(5))
        bound = 1. / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mean, -bound, bound)
        nn.init.constant_(self._weight_std_param, np.log(np.exp(init_std) - 1))
        if self.bias:
            nn.init.uniform_(self.bias_mean, -bound, bound)
            nn.init.constant_(self._bias_std_param, np.log(np.exp(init_std) - 1))

    # define the q distribution standard deviations with property decorator
    @property
    def weight_std(self):
        return torch.log(1 + torch.exp(self._weight_std_param))

    @property
    def bias_std(self):
        return torch.log(1 + torch.exp(self._bias_std_param))

    # KL divergence KL[q||p] between approximate Gaussian posterior and Gaussian prior
    def kl_divergence(self):
        """
        Alternative to benchmark later:
        kl = (log_sigma_1 - log_sigma_0 + \
        (torch.exp(log_sigma_0)**2 + (mu_0-mu_1)**2)/(2*math.exp(log_sigma_1)**2) - 0.5).sum()
        Could also test using sampled weights as in paper (not closed form complexity cost).
        """
        q_weight = dist.Normal(self.weight_mean, self.weight_std)
        p_weight = dist.Normal(self.prior_weight_mean, self.prior_weight_std)
        kl = dist.kl_divergence(q_weight, p_weight).sum()
        if self.bias:
            q_bias = dist.Normal(self.bias_mean, self.bias_std)
            p_bias = dist.Normal(self.prior_bias_mean, self.prior_bias_std)
            kl += dist.kl_divergence(q_bias, p_bias).sum()
        return kl

    # forward pass using reparam trick
    def forward(self, input):
        weight = self.weight_mean + self.weight_std * torch.randn_like(self.weight_std)
        if self.bias:
            bias = self.bias_mean + self.bias_std * torch.randn_like(self.bias_std)
        else:
            bias = None
        return F.linear(input, weight, bias)


# construct a BNN with learnable prior (std)
def make_linear_emp_bnn(layer_sizes, device, activation='LeakyReLU'):
    nonlinearity = getattr(nn, activation)() if isinstance(activation, str) else activation
    net = nn.Sequential()
    net.register_parameter(name='_prior_std', param=nn.Parameter(torch.tensor(0.5413, device=device)))
    for i, (dim_in, dim_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        net.add_module(f'EmpBayesLinear{i}', EmpBayesLinear(dim_in, dim_out, net._prior_std, device=device))
        if i < len(layer_sizes) - 2:
            net.add_module(f'Nonlinearity{i}', nonlinearity)
    return net
