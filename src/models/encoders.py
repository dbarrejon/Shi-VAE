# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021. by Daniel Barrejon, UC3M.                                         +
#  All rights reserved. This file is part of the Shi-VAE, and is released under the      +
#  "MIT License Agreement". Please see the LICENSE file that should have been included   +
#   as part of this package.                                                             +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from abc import ABC

import torch
import torch.nn as nn

from lib import utils
from lib.aux import set_device
from models import samplers


class GMEncoder(nn.Module, ABC):
    r"""
    Encoder for the Shi-VAE, two latent codes: :math:`z`, real  and :math:`s, discrete.
    """
    def __init__(self, x_dim, z_dim, s_dim, h_dim, activation_layer='ReLU'):
        r"""

        Args:
            x_dim (int): Dimensionality of x.
            z_dim (int): Dimensionality of :math:`z`.
            s_dim (int): Dimensionality of :math:`s`.
            h_dim (int): Dimensionality of the embedding space for the LSTM.
            activation_layer (string): Choose "relu", "tanh" or "sigmoid".
        """
        super(GMEncoder, self).__init__()
        self.h_dim = h_dim
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.s_dim = s_dim

        self.device = set_device()
        self.activation = utils.set_activation_layer(activation_layer)

        # Sampler
        self.sampler = samplers.Sampler()
        # Gumbel Softmax for discrete latent variable: s
        self.gs_sampler = samplers.GumbelSoftmaxSampler()

        # Feature extraction
        self.phi_x = nn.Sequential(
            nn.Linear(self.x_dim, self.h_dim),
            self.activation,
            nn.Linear(self.h_dim, self.h_dim),
            self.activation)

        self.phi_x_s = nn.Sequential(
            nn.Linear(self.x_dim + self.h_dim, self.h_dim),
            self.activation,
            nn.Linear(self.h_dim, self.s_dim),
            nn.LeakyReLU(negative_slope=0.01))

        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(self.h_dim + self.h_dim + self.s_dim, self.h_dim),  # add s_dim
            self.activation,
            nn.Linear(self.h_dim, self.h_dim),
            self.activation)
        self.enc_mean = nn.Linear(self.h_dim, self.z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(self.h_dim, self.z_dim),
            nn.Softplus())

    def forward(self, x_t, h_past, temp=3):
        r"""
        Forward pass.
        Args:
            x_t (Tensor): Shape (BxD)
            h_past (Tensor): Shape (Bxh_dim)
            temp (int): Temperature value for the Gumbel Softmax sampler.

        Returns:
            z_pack: Tuple with sample z, z mean and z std.
            s_pack: Tuple with sample s and s probs.
        """
        phi_x_t = self.phi_x(x_t)
        phi_x_t_s = self.phi_x_s(torch.cat([x_t, h_past], 1))   # v2: z_t prior for s_t+1

        # s_t
        s_t_logits = phi_x_t_s
        s_t_probs = torch.nn.functional.softmax(s_t_logits, dim=1)  # probs Categorical
        s_t = self.gs_sampler.gumbel_softmax(s_t_logits, temp, one_hot=False)
        s_pack = (s_t, s_t_probs)

        enc_t = self.enc(torch.cat([phi_x_t, s_t, h_past], 1))
        # moments
        z_t_mean = self.enc_mean(enc_t)
        z_t_std = self.enc_std(enc_t)
        # sample
        z_t = self.sampler.reparameterized_sample(z_t_mean, z_t_std)
        z_pack = (z_t, z_t_mean, z_t_std)
        return z_pack, s_pack
