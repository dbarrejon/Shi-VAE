# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021. by Daniel Barrejon, UC3M.                                         +
#  All rights reserved. This file is part of the Shi-VAE, and is released under the      +
#  "MIT License Agreement". Please see the LICENSE file that should have been included   +
#   as part of this package.                                                             +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from abc import ABC

import torch
import torch.nn as nn
from torch.autograd import Variable

from lib import utils
from models import samplers
from models.base_missvae import BaseMissVAE
from models.decoders import HeterDecoder
from models.encoders import GMEncoder


class ShiVAE(BaseMissVAE, ABC):
    r"""
    Shi-VAE model.
    """
    def __init__(self, h_dim, z_dim, s_dim, n_layers, types_list=None, learn_std=False, activation_layer='ReLU',
                 gain_init=1):
        r"""
        Args:
            h_dim (int): Dimensionality of the embedding space for the LSTM.
            z_dim (int): Dimensionality of :math:`z`.
            s_dim (int): Dimensionality of :math:`s`.
            n_layers (int): Number of layers for the LSTM. Normally, use just one.
            types_list (list of dictionaries): Each dictionary contains: name, type, dim, nclass, index; for every
            attribute.
            learn_std (boolean): If true, learn the :math:`\sigma` for the real and positive distributions.
            activation_layer (string): Choose "relu", "tanh" or "sigmoid".
            gain_init (int): Gain value for the initializers.
        """
        super(ShiVAE, self).__init__(types_list, learn_std, activation_layer)

        # Dimensions
        self.x_dim = utils.get_x_dim(self.types_list)
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.s_dim = s_dim
        self.n_layers = n_layers

        # Encoder
        self.encoder = GMEncoder(self.x_dim, self.z_dim, self.s_dim, self.h_dim, activation_layer=activation_layer)

        # Feature z
        self.phi_z = nn.Sequential(nn.Linear(self.z_dim, self.h_dim),
                                   self.activation)

        # Prior
        self.prior = nn.Sequential(
            nn.Linear(self.h_dim + self.s_dim, self.h_dim),
            self.activation)
        self.prior_mean = nn.Linear(self.h_dim, self.z_dim)
        self.prior_std = nn.Sequential(nn.Linear(self.h_dim, self.z_dim),
                                       nn.Softplus())

        # Decoder: Decode for every dimension
        self.dec = HeterDecoder(self.h_dim, self.x_dim, self.s_dim, types_list=self.types_list,
                                learn_std=self.learn_std)

        # Recurrence
        self.rnn = nn.LSTM(self.h_dim, self.h_dim, self.n_layers)

        self.init_weights(gain=gain_init)

    def feed2model(self, x, temp=3):
        r"""
        Feed data to the model. It is used as forward function.
        Args:
            x (Tensor): Shape (TxBxD)
            temp (int): Temperature value for Shi-VAE.

        Returns:
            z_tup: Tuple with sample z, z mean and z std.
            z_prior_tup: Tuple with z_prior sample and z_prior mean.
            s_tup: Tuple with s sample and s probs.
            likes: Dictionary with the parameters.
        """
        T = x.size(0)
        batch_t = x.size(1)

        # ======= Truncate long sequences ======= #
        t_range = utils.chop_sequence(T)

        # ======= Forward ======= #
        # init h
        h = Variable(torch.zeros(self.n_layers, batch_t, self.h_dim)).to(self.device)
        c = Variable(torch.zeros(self.n_layers, batch_t, self.h_dim)).to(self.device)
        t_init = 0

        z_list = []
        z_prior_list = []
        s_list = []
        dec_params_list = []

        for t_end in t_range:

            # Zero-filling
            x_minibatch = x[t_init:t_end, :batch_t, :]
            t_init = t_end
            T = x_minibatch.shape[0]

            for t in range(T):

                # ======= Fetch Data ======= #
                x_t = x_minibatch[t]

                # ======= Update RNN states ======= #
                h_t = h
                c_t = c

                # ======= Encoder ======= #
                z_pack, s_pack = self.encoder(x_t, h_t[-1], temp=temp)
                z_t, z_t_mean, z_t_std = z_pack
                s_t, s_t_probs = s_pack

                # ======= Prior ======= #
                z_prior_t = self.prior(torch.cat([h_t[-1], s_t], 1))
                # moments
                z_prior_t_mean = self.prior_mean(z_prior_t)
                z_prior_t_std = self.prior_std(z_prior_t)
                z_prior_pack = (z_prior_t_mean, z_prior_t_std)

                # ======= Decoder ======= #
                # Decoder: Decode dimensional wise.
                phi_z_t = self.phi_z(z_t)
                dec_params = self.dec(phi_z_t, s_t, h_t[-1])

                # ======= Recurrence ======= #
                out_t_rec, h_c = self.rnn(phi_z_t.unsqueeze(0), (h_t, c_t))
                h, c = h_c

                # Append
                z_list.append(z_pack)
                z_prior_list.append(z_prior_pack)
                s_list.append(s_pack)
                dec_params_list.append(dec_params)

        # Convert into a list of tuples
        z_tup = utils.list2tupandstack(z_list)
        z_prior_tup = utils.list2tupandstack(z_prior_list)
        s_tup = utils.list2tupandstack(s_list)

        # Convert params into likelihood objects.
        dec_params = self.list2dict_decparams(dec_params_list)
        likes = self.decparams2likes(dec_params)

        return z_tup, z_prior_tup, s_tup, likes

    def sample_from_prior(self, batch_size, ts, temp=0.02):
        r"""
        Sample from prior. Use to generate samples.
        Args:
            batch_size (int): Batch size.
            ts (int): Length of time series.
            temp (int): emperature value for Shi-VAE.

        Returns:
            z_prior_tup: Tuple with z_prior sample and z_prior mean.
            s_tup: Tuple with s sample and s probs.
            likes: Dictionary with the parameters.
        """
        # init h
        h = Variable(torch.zeros(self.n_layers, batch_size, self.h_dim)).to(self.device)
        c = Variable(torch.zeros(self.n_layers, batch_size, self.h_dim)).to(self.device)
        gs_sampler = samplers.GumbelSoftmaxSampler()
        z_prior_list = []
        s_list = []
        dec_params_list = []

        for t in range(ts):
            # s
            s_prior_t_probs = 1 / self.s_dim * torch.ones([batch_size, self.s_dim]).to(self.device)
            s_t = gs_sampler.gumbel_softmax(torch.log(s_prior_t_probs), temp, one_hot=False)
            s_pack = (s_t, s_prior_t_probs)

            # Prior
            prior_t = self.prior(torch.cat([h[-1], s_t], 1))
            z_prior_t_mean = self.prior_mean(prior_t)
            z_prior_t_std = self.prior_std(prior_t)
            # Sample z_t_prior
            z_prior_t = self.sampler.reparameterized_sample(z_prior_t_mean, z_prior_t_std)

            z_prior_pack = (z_prior_t, z_prior_t_mean, z_prior_t_std)

            # ======= Decoder ======= #
            # Decoder: Decode dimensional wise.
            phi_z_t = self.phi_z(z_prior_t)
            dec_params = self.dec(phi_z_t, s_t, h[-1])

            # Recurrence update
            phi_z_t_rec, h_c = self.rnn(phi_z_t.unsqueeze(0), (h, c))
            h, c = h_c

            # Append
            z_prior_list.append(z_prior_pack)
            s_list.append(s_pack)
            dec_params_list.append(dec_params)

        # Convert into a list of tuples
        z_prior_tup = utils.list2tupandstack(z_prior_list)
        s_tup = utils.list2tupandstack(s_list)

        # Convert params into likelihood objects.
        dec_params = self.list2dict_decparams(dec_params_list)
        likes = self.decparams2likes(dec_params)

        return z_prior_tup, s_tup, likes
