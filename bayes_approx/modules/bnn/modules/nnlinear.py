import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


class nnLinear(nn.Module):
    """Applies a linear transformation to the incoming data: y = xW^T + b, where
       the weight W and bias b are sampled from the approximate q distribSution.
    """
    def __init__(self, in_features, out_features, prior_weight_std=1.0, prior_bias_std=1.0,
                 bias=True, init_std=0.05, sqrt_width_scaling=False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype, 'requires_grad': True}
        super(nnLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # approximate posterior parameters (Gaussian)
        self.weight_mean = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        self.bias = bias
        if self.bias:
            self.bias_mean = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias_mean', None)
        self.reset_parameters(init_std)

        # prior parameters (Gaussian)
        prior_mean = 0.0
        if sqrt_width_scaling:  # prior variance scales as 1/in_features
            prior_weight_std /= self.in_features ** 0.5
        # prior parameters are registered as constants
        self.register_buffer('prior_weight_mean', torch.full_like(self.weight_mean, prior_mean))
        self.register_buffer('prior_weight_std', torch.full_like(self.weight_mean, prior_weight_std))
        if self.bias:
            self.register_buffer('prior_bias_mean', torch.full_like(self.bias_mean, prior_mean))
            self.register_buffer('prior_bias_std', torch.full_like(self.bias_mean, prior_bias_std))
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
        if self.bias:
            nn.init.uniform_(self.bias_mean, -bound, bound)

    # KL divergence KL[q||p] between approximate Gaussian posterior and Gaussian prior
    def kl_divergence(self):
        """
        Alternative to benchmark later:
        kl = (log_sigma_1 - log_sigma_0 + \
        (torch.exp(log_sigma_0)**2 + (mu_0-mu_1)**2)/(2*math.exp(log_sigma_1)**2) - 0.5).sum()
        Could also test using sampled weights as in paper (not closed form complexity cost).
        """
        q_weight = dist.Normal(self.weight_mean, 0)
        p_weight = dist.Normal(self.prior_weight_mean, self.prior_weight_std)
        kl = dist.kl_divergence(q_weight, p_weight).sum()
        if self.bias:
            q_bias = dist.Normal(self.bias_mean, 0)
            p_bias = dist.Normal(self.prior_bias_mean, self.prior_bias_std)
            kl += dist.kl_divergence(q_bias, p_bias).sum()
        return kl

    # forward pass using reparam trick
    def forward(self, input):
        weight = self.weight_mean
        if self.bias:
            bias = self.bias_mean
        else:
            bias = None
        return F.linear(input, weight, bias)


# construct a BNN
def make_linear_nn(layer_sizes, activation='LeakyReLU', **layer_kwargs):
    nonlinearity = getattr(nn, activation)() if isinstance(activation, str) else activation
    net = nn.Sequential()
    for i, (dim_in, dim_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        net.add_module(f'nnLinear{i}', nnLinear(dim_in, dim_out, **layer_kwargs))
        if i < len(layer_sizes) - 2:
            net.add_module(f'Nonlinearity{i}', nonlinearity)
    return net
