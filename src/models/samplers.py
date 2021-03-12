# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021. by Daniel Barrejon, UC3M.                                         +
#  All rights reserved. This file is part of the Shi-VAE, and is released under the      +
#  "MIT License Agreement". Please see the LICENSE file that should have been included   +
#   as part of this package.                                                             +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import torch.nn as nn
from torch.autograd import Variable


class GumbelSoftmaxSampler:
    """
    Gumbel Softmax Sampler.

    Reference:
        https://arxiv.org/abs/1611.01144
    """
    def __init__(self):
        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def sample_gumbel(shape, eps=1e-20):
        u = torch.rand(shape)
        return -Variable(torch.log(-torch.log(u + eps) + eps))

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size()).to(logits.device)
        return self.softmax(y / temperature)

    def gumbel_softmax(self, logits, temperature, one_hot=False):
        y_soft = self.gumbel_softmax_sample(logits, temperature)
        shape = y_soft.size()
        if one_hot:
            _, ind = y_soft.max(dim=-1)
            y_hard = torch.zeros_like(y_soft).view(-1, shape[-1])
            y_hard.scatter_(1, ind.view(-1, 1), 1)
            y_hard = y_hard.view(*shape)
            y_hard = (y_hard - y_soft).detach() + y_soft
            y = y_hard.view(-1, shape[-1])
        else:
            y = y_soft
        return y


class Sampler:
    """
    Generic Sampler
    - Gaussian.
    - Bernoulli.
    - Categorical.
    """
    def __init__(self):
        return

    @staticmethod
    def reparameterized_sample(mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_().to(mean.device)
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    # def sample_gaussian(self, mean, std):
    #     # Reparametrization trick samples from a multivariate Gaussian.
    #     return self.reparameterized_sample(mean, std)
    #
    # @staticmethod
    # def sample_bernoulli(logits):
    #     # ber_sampler = torch.distributions.Bernoulli(logits=logits)
    #     probs = torch.torch.sigmoid(logits)
    #     ber_sampler = torch.distributions.Bernoulli(probs)
    #     return ber_sampler.sample()
    #
    # @staticmethod
    # def sample_categorical(logits):
    #     # probs = torch.torch.sigmoid(logits)
    #     cat_sampler = torch.distributions.one_hot_categorical.OneHotCategorical(logits=logits)
    #     return cat_sampler.sample()
