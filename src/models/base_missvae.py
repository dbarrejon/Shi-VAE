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
from lib.loss import Loss
from models import samplers

print_shivae = "ELBO:{n_elbo:.3f} | NLL:{nll_loss:.3f}  KL_Q_Z:{kld_q_z:.3f}  KL_Q_S:{kld_q_s:.3f}|  " \
                "Real:{nll_real:.3f}  Pos:{nll_pos:.3f}  Bin:{nll_bin:.3f}  Cat:{nll_cat:.3f}"
LOSSES = ["real", "pos", "bin", "cat"]


class BaseMissVAE(nn.Module, ABC):
    """
    Base class for MissVAE model.
    """
    def __init__(self, types_list=None, learn_std=False, activation_layer='ReLU', K=1, M=1):
        """
        Initialize BaseMissVAE.
        Args:
            types_list (list of dictionaries): Each dictionary contains: name, type, dim, nclass, index; for every
            attribute.
            learn_std (boolean): Learn the :math:`\sigma` for the real and positive distributions.
            activation_layer (string): Choose "relu", "tanh" or "sigmoid".
            K: number of importance weights for IWAE model (see: https://arxiv.org/abs/1509.00519)
            M: number of Monte Carlo samples for ELBO estimation
        """
        super(BaseMissVAE, self).__init__()

        # Heterogeneous vars
        assert types_list is not None
        self.types_list = utils.reindex_types_list(types_list)
        self.transform_idx = utils.get_idx_transform(self.types_list)

        self.device = set_device()
        self.learn_std = learn_std
        self.activation = utils.set_activation_layer(activation_layer)

        # Sampler
        self.sampler = samplers.Sampler()

        # Loss
        self.loss = Loss()
        self.K = K
        self.M = M

    def get_name(self):
        r"""
        Obtain the name of the specific model.
        """
        return self.__class__.__name__

    def init_weights(self, gain=1):
        r"""
        Initialize model weights. LSTM weights are initialized with orthogonal initialization and the rest with xavier
        normalization.
        Orthogonal init.: https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.orthogonal_
        Xavier normal init.: https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_normal_
        Args:
            gain (int): Gain parameter for the initializers.
        """

        for m in self.children():
            if type(m) in [nn.LSTM]:
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0.0)
                    elif 'weight' in name:
                        nn.init.orthogonal_(param, gain=gain)
            else:
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0.0)
                    elif 'weight' in name:
                        nn.init.xavier_normal_(param, gain=gain)

    @staticmethod
    def list2dict_decparams(dec_params_list):
        r"""
        Convert a list of dictionaries containing the parameters of the distributions into a single dictionary
        with the parameters for each distribution stored as tensors, or tuples of tensors (normal, real).
        Args:
            dec_params_list (list of dicts): Each dictionary in the list contains the parameters for every attribute,
            e.g., for a real variable it will contain the mean and the sigma parameter. The length of the list is equal
            to the length of the sequences (or if the sequences are of different length, the maximum length in the
            batch)

        Returns:
            A dictionary with every attribute and the corresponding parameters with shape (TxBxL), where L depends on
            the type of parameter.
        """
        dec_params = {

            k: [d.get(k) for d in dec_params_list]
            for k in set().union(*dec_params_list)
        }
        for key in dec_params.keys():
            dec_params[key] = utils.list2tupandstack(dec_params[key])
        return dec_params

    def decparams2likes(self, dec_params):
        r"""
        Generate the different likelihoods depending on the data type with the corresponding parameters of the
        distribution.
        Args:
            dec_params (dictionary): Each key is an attribute with the corresponding parameters as values.

        Returns:
            Dictionary with the attributes as keys and the corresponding likelihood objects as values.
        """
        likes = {}
        for i, type_dict in enumerate(self.types_list):
            # Fetch type data
            type = type_dict['type']
            var = type_dict['name']

            # Real and positive distributions
            if type in ["real", "pos"]:
                mu, std = dec_params[var]
                likes[var] = torch.distributions.Normal(mu, std)

            # Bernoulli Distributions
            elif type == "bin":
                theta = dec_params[var]
                likes[var] = torch.distributions.Bernoulli(logits=theta)

            # Categorical Distributions
            elif type == "cat":
                theta = dec_params[var]
                likes[var] = torch.distributions.one_hot_categorical.OneHotCategorical(logits=theta)

        return likes

    def forward(self, x, mask, beta=1., temp=3):
        r"""
        Forward the data through the model and compute the losses.

        Args:
            x (Tensor): Input data of shape :math:`(TxNxD)`
            mask (Tensor): Mask data of shape :math:`(TxNxD)`
            beta (int, optional): Annealing parameter :math:`\beta`
            temp (int, optional): Temperature value for the Gumbel Softmax Sampler.

        Returns:
            Dictionary with the losses, e.g., elbo, KL loss, etc.
        """
        # Replicate for IWAE and MonteCarlo
        x = x.repeat((1, self.K*self.M, 1))  # shape=(T, M*K*BS, D)
        mask = mask.repeat((1, self.K*self.M, 1))  # shape=(T, M*K*BS, D)

        z_tup, z_prior_tup, s_tup, likes = self.feed2model(x, temp)
        loss_dict = self.compute_loss(x, mask, z_tup, z_prior_tup, likes, beta=beta, s_tup=s_tup)
        print(print_shivae.format(**loss_dict))
        return loss_dict

    def compute_loss(self, x, mask, z_tup, z_prior_tup, likes, beta=1., s_tup=None):
        r"""
        Computes the global loss for the model, calculating the heterogeneous nll, the individual :math:`KL` terms for
        :math:`z` and :math:`s` and applying the annealing on the KL. If IWAE or Monte Carlo are available, the loss
        is calculated accordingly.
        Args:
            x (Tensor): Input data of shape :math:`(TxNxD)`
            mask (Tensor): Mask data of shape :math:`(TxNxD)`
            z_tup (Tensor tuple): Tensor with dimension 3.
                - Firs element: z sample, with shape :math:`(TxNxZ_dim)`.
                - Second element: z mean, with shape :math:`(TxNxZ_dim)`.
                - Third element: z std, with shape :math:`(TxNxZ_dim)`.

            z_prior_tup (Tensor tuple): Tensor with dimension 2.
                - Firs element: z prior mean, with shape :math:`(TxNxZ_dim)`.
                - Second element: z prior std, with shape :math:`(TxNxZ_dim)`.

            likes (Dictionary of distributions): Dictionary with the attributes as keys and the corresponding
                likelihood objects as values.
            beta (int, optional): Annealing parameter :math:`\beta`
            s_tup (Tensor tuple): Tensor with dimension 2.
                - Firs element: s sample, with shape :math:`(TxNxK)`.
                - Second element: s probs prior std, with shape :math:`(TxNxK)`.

        Returns:
            Dictionary with the losses, e.g., elbo, KL loss, etc.
        """
        # NLL
        nll_dict = self.heter_nll(x, mask, likes, types_list=self.types_list, eval_obs=False)
        nll_loss = sum(nll_dict.values())  # shape=(M*K*BS)

        # KL Divergence
        kld_q_z_loss = self.kl_q_z(z_tup, z_prior_tup)  # shape=(M*K*BS)
        kld_q_s_loss = self.kl_q_s(s_tup)  # shape=(M*K*BS)

        # IWAE
        if self.K > 1:
            weights = -nll_loss - kld_q_z_loss - kld_q_s_loss  # shape=(M*K*BS)
            weights = weights.reshape((10, 10, -1))  # shape=(M, K, BS)

            elbo = torch.logsumexp(weights, axis=1)  # shape=(M, 1, BS)
            elbo = torch.mean(elbo)  # scalar
            n_elbo = -elbo
        # Monte Carlo, K=1
        else:
            n_elbo = nll_loss + beta * kld_q_z_loss + beta * kld_q_s_loss
            n_elbo = n_elbo.mean()

        nll_dict = {k: v.mean() for k, v in nll_dict.items()}  # average over M,K
        loss_dict = {'n_elbo': n_elbo,
                     'nll_loss': nll_loss.mean(),
                     'kld': kld_q_z_loss.mean() + kld_q_s_loss.mean(),
                     'kld_q_z': kld_q_z_loss.mean(),
                     'kld_q_s': kld_q_s_loss.mean()
                     }

        # ======= Check nan ======= #
        if torch.isnan(kld_q_z_loss.sum()):
            print("KL Loss NAN")
            print(kld_q_z_loss)
            exit()
            # Heter NLL
        utils.check_nan_losses(nll_dict)

        loss_dict = {**loss_dict, **nll_dict}
        return loss_dict

    def heter_nll(self, x, mask, likes, types_list=None, eval_obs=False):
        r"""
        Evaluate heterogeneous nll on the different types.
        If eval_obs is set to True, nll is calculated on the missing points given by :attr:mask.
        Args:
            x (Tensor): Input data of shape :math:`(TxNxD)`
            mask (Tensor): Mask data of shape :math:`(TxNxD)`
            likes (Dictionary of distributions): Dictionary with the attributes as keys and the corresponding
                likelihood objects as values.
            types_list (list of dictionaries): Each dictionary contains: name, type, dim, nclass, index; for every
            attribute.
            eval_obs (boolean): True, evaluate on observations. False, evaluate on missing mask (designed for evaluating
            at syntehitc missing).

        Returns:
            Dictionary with the nll for every attribute.
        """

        if types_list is None:
            types_list = self.types_list

        # heterogeneous nll dict
        loss_dict = utils.create_lossdict(LOSSES, self.device)

        for i, type_dict in enumerate(types_list):
            # Fetch type data
            type = type_dict['type']
            nll_type = "nll_" + type
            var = type_dict['name']
            dim = int(type_dict['dim'])

            # Get transformed idx
            transform_idx = self.transform_idx[i]
            low_idx = transform_idx
            up_idx = transform_idx + dim

            # Filter by idx
            x_type = x[:, :, low_idx:up_idx]  # filter by variables
            mask_type = mask[:, :, low_idx:up_idx].squeeze(dim=-1)
            # Collide into one dimension for categorical.
            if type == "cat":
                mask_type = mask_type[..., -1]

            # ======= log p(x_t^0 | z<=t, s<=t = k) HETEROGENEOUS NLL ======= #
            nll = -likes[var].log_prob(x_type).squeeze(-1)  # shape=(T,M*K*BS)
            if eval_obs:
                nll = torch.where(mask_type, nll, torch.zeros_like(nll))
            else:
                nll = torch.where(mask_type, torch.zeros_like(nll), nll)

            # loss_dict[nll_type] = loss_dict[nll_type] + nll.sum()  # total loss
            loss_dict[nll_type] = loss_dict[nll_type] + nll.sum(dim=0)  # Sum over time!: nll shape = (K*BS*M)
        return loss_dict

    @staticmethod
    def kl_q_z(z_tup, z_prior_tup):
        r"""
        Compute the KL for :math:`z`.
        Args:
            z_tup (Tensor tuple): Tensor with dimension 3.
                - Firs element: z sample, with shape :math:`(TxNxZ_dim)`.
                - Second element: z mean, with shape :math:`(TxNxZ_dim)`.
                - Third element: z std, with shape :math:`(TxNxZ_dim)`.

            z_prior_tup (Tensor tuple): Tensor with dimension 2.
                - Firs element: z prior mean, with shape :math:`(TxNxZ_dim)`.
                - Second element: z prior std, with shape :math:`(TxNxZ_dim)`.

        Returns:
            The KL loss for :math:`z`, with shape=N.
        """
        z, z_mean, z_std = z_tup
        z_prior_mean, z_prior_std = z_prior_tup

        # stability correction
        std_q = torch.clamp(z_std, 1e-6, 1e20)
        std_p = torch.clamp(z_prior_std, 1e-6, 1e20)

        z_mean = torch.clamp(z_mean, -1e20, 1e20)
        z_prior_mean = torch.clamp(z_prior_mean, -1e20, 1e20)

        q = torch.distributions.Normal(z_mean, std_q)
        p = torch.distributions.Normal(z_prior_mean, std_p)
        kl = torch.distributions.kl_divergence(q, p)
        kl[torch.isnan(kl)] = 0
        return kl.sum(axis=(0, 2))

    def kl_q_s(self, s_tup):
        r"""
        Computes the KL for :math:`s`. The prior is assumed uniform.
        Args:
            s_tup (Tensor tuple): Tensor with dimension 2.
                - Firs element: s sample, with shape :math:`(TxNxK)`.
                - Second element: s probs prior std, with shape :math:`(TxNxK)`.s

        Returns:
            The KL loss for :math:`s`, with shape=N.

        """
        s, s_probs = s_tup

        # Uniform Prior
        s_prior = (1 / self.s_dim * torch.ones_like(s_probs)).to(self.device)

        q = torch.distributions.Categorical(s_probs)
        p = torch.distributions.Categorical(s_prior)
        kl = torch.distributions.kl_divergence(q, p)
        if torch.isnan(kl).sum() > 0:
            # 0 * log (0) ~ 0 to avoid nan
            kl[torch.isnan(kl)] = 0
        return kl.sum(axis=0)

    def reconstruction(self, x, temp=0.01):
        r"""
        Generate the reconstruction for some input data.
        Args:
            x (Tensor): Input data of shape :math:`(TxNxD)`
            temp (int, optional): Temperature value for the Gumbel Softmax Sampler.

        Returns:
            A tuple of:
            - (likes, dec): Likelihood parameters and reconstructed samples.
            - (z_tup, s_tup): The tuples for :math:`z` and :math:`s`.
        """
        z_tup, z_prior_tup, s_tup, likes = self.feed2model(x, temp)
        dec_x = self.generate_heter_samples(likes)
        return (likes, dec_x), (z_tup, s_tup)

    def sample(self, batch_size, ts, temp=0.01):
        r"""
        Sample with shape :attr:batch_size of lenth :attr:ts.
        Args:
            batch_size (int): Batch size.
            ts (int): Length for every sequence.
            temp (int, optional): Temperature value for Gumbel Softmax.

        Returns:
            A tuple of:
                - (likes, dec): Likelihood parameters and reconstructed samples.
                - (z_tup, s_tup): The tuples for :math:`z` and :math:`s`.
        """
        z_prior_tup, s_tup, likes = self.sample_from_prior(batch_size, ts, temp)
        dec_x = self.generate_heter_samples(likes)
        return (likes, dec_x), z_prior_tup, s_tup

    def generate_heter_samples(self, likes, n_samples=100):
        r"""
        Generates the samples from the likelihood distributions.
        Args:
            likes: Dictionary with the likelihood distributions for every attribute.
            n_samples: Number of samples to generate.

        Returns:
            The reconstructed sample.
        """
        dec_x = torch.zeros([0, ]).to(self.device)
        # Heterogeneous Sampling
        for (var, lik), type_dict in zip(likes.items(), self.types_list):
            type = type_dict["type"]
            if type in ["bin", "cat"]:
                # x_type = torch.round(lik.probs).detach()  # Not sample, but take mean
                x_type = torch.round(torch.mean(lik.sample((n_samples,)), axis=0))

            elif type in ["real", "pos"]:
                # x_type = lik.mean.detach()  # Not sample, but take mean
                x_type = torch.mean(lik.sample((n_samples,)), axis=0)

            else:
                x_type = lik.sample()

            dec_x = torch.cat([dec_x, x_type], dim=-1)
        return dec_x
