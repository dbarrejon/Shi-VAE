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


# ====== Shi-VAE Heterogeneous Decoder ====== #
class HeterDecoder(nn.Module, ABC):
    r"""
    Heterogeneous decoder for the Shi-VAE.
    """
    def __init__(self, h_dim, x_dim, s_dim, types_list=None, learn_std=False, fixed_std=0.1, activation_layer='ReLU'):
        r"""
        Init
        Args:
            h_dim (int): Dimensionality of the embedding space for the LSTM.
            x_dim (int): Dimensionality of x.
            s_dim (int): Dimensionality of :math:`s`.
            types_list (list of dictionaries): Each dictionary contains: name, type, dim, nclass, index; for every
            attribute.
            learn_std (boolean): If true, learn the :math:`\sigma` for the real and positive distributions.
            fixed_std (float): If :attr:learn_std is False, then use this as :math:`\sigma`.
            activation_layer (string): Choose "relu", "tanh" or "sigmoid".
        """
        super(HeterDecoder, self).__init__()
        # dimensions
        self.h_dim = h_dim
        self.x_dim = x_dim
        self.s_dim = s_dim

        # std options
        self.learn_std = learn_std
        self.fixed_std = fixed_std
        self.device = set_device()

        self.decoder_dict = nn.ModuleDict()  # need for parameters in state_dict()

        self.activation = utils.set_activation_layer(activation_layer)

        assert types_list is not None
        self.types_list = types_list

        # Init Dimensional Decoder:
        for type_dict in self.types_list:
            var = type_dict['name']
            type = type_dict['type']

            d_dict = nn.ModuleDict()
            # Common to any type of data
            d_dict["dec"] = nn.Sequential(
                nn.Linear(h_dim + h_dim + s_dim, h_dim + h_dim),
                self.activation,
                nn.Linear(h_dim + h_dim, h_dim),
                self.activation,
                nn.Linear(h_dim, h_dim),
                self.activation)

            # Real
            if type == 'real':
                d_dict["dec_mean"] = nn.Linear(h_dim, 1)
                if self.learn_std:
                    d_dict["dec_std"] = nn.Sequential(
                        nn.Linear(h_dim, 1),
                        nn.Softplus())
                else:
                    pass
            # Positive
            elif type == 'pos':
                d_dict["dec_mean"] = nn.Linear(h_dim, 1)

                if self.learn_std:
                    d_dict["dec_std"] = nn.Sequential(
                        nn.Linear(h_dim, 1),
                        nn.Softplus())
                else:
                    pass
            # Bernoulli
            elif type == 'bin':
                d_dict["dec_p"] = nn.Sequential(
                    nn.Linear(h_dim, h_dim),
                    self.activation,
                    nn.Linear(h_dim, 1))

            # Categorical
            elif type == 'cat':
                # Categorical parameters are the output of 1 DNN.
                C = int(type_dict['nclass'])  # number of categories
                # First parameter is set to 0 for identifiability
                d_dict["dec_p_cat"] = nn.Sequential(
                    nn.Linear(h_dim, h_dim),
                    self.activation,
                    nn.Linear(h_dim, C - 1))

            else:
                pass

            self.decoder_dict[var] = d_dict

    def forward(self, phi_z_t, s_t, h_past):
        r"""
        Forward pass.
        Args:
            phi_z_t (Tensor): Shape (BxD)
            s_t (Tensor): Shape (BxD)
            h_past (Tensor): Shape (Bxh_dim)

        Returns:
            params (dict): Dictionary with the parameters of the distributions for every attribute.

        """
        batch_size = phi_z_t.size(0)

        # Params dictionary
        var_names = [type_dict['name'] for type_dict in self.types_list]
        params = dict.fromkeys(var_names)

        embed_z_h = torch.cat([phi_z_t, s_t, h_past], 1)

        # Dimensional decoder
        for type_dict in self.types_list:
            var = type_dict['name']
            type = type_dict['type']

            # Common Layer
            dec_t = self.decoder_dict[var]["dec"](embed_z_h)

            # Gaussian
            if type == 'real':
                dec_mean_t_d = self.decoder_dict[var]["dec_mean"](dec_t)
                if self.learn_std:
                    dec_std_t_d = self.decoder_dict[var]["dec_std"](dec_t)
                else:
                    dec_std_t_d = torch.ones(batch_size, 1).to(self.device) * self.fixed_std  # default 0.1

                params[var] = [dec_mean_t_d, dec_std_t_d]

            # Positive
            if type == 'pos':
                dec_log_mean_t_d = self.decoder_dict[var]["dec_mean"](dec_t)
                if self.learn_std:
                    dec_log_std_t_d = self.decoder_dict[var]["dec_std"](dec_t)
                else:
                    dec_log_std_t_d = torch.ones(batch_size, 1).to(self.device) * self.fixed_std  # default 0.1

                params[var] = [dec_log_mean_t_d, dec_log_std_t_d]

            # Bernoulli
            elif type == 'bin':
                dec_p_t_d = self.decoder_dict[var]["dec_p"](dec_t)
                params[var] = dec_p_t_d

            # Categorical
            elif type == 'cat':

                # Categorical parameters are the output of 1 DNN.
                dec_p_t = self.decoder_dict[var]["dec_p_cat"](dec_t)

                # insert 0 for the first value
                zeros_d = torch.zeros(batch_size, 1).to(self.device)
                dec_p_t_cat = torch.cat([zeros_d, dec_p_t], dim=1)
                params[var] = dec_p_t_cat

            else:
                pass

        return params
