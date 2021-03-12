# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021. by Daniel Barrejon, UC3M.                                         +
#  All rights reserved. This file is part of the Shi-VAE, and is released under the      +
#  "MIT License Agreement". Please see the LICENSE file that should have been included   +
#   as part of this package.                                                             +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import mean_squared_error, confusion_matrix

from lib import utils
from lib.aux import set_device


def burst_cross_correlation(x, x_hat, mask):
    r"""
    Calculates the cross correlation for :attr:x and :attr:x_hat given by mask. Mask must present missing following
    sequential patterns (bursts).
    Args:
        x (Tensor): True input data.
        x_hat (Tensor): Reconstructed signal.
        mask (Tensor): Mask for the artificial missing.

    Returns:
        Total normalized cross correlation for all the bursts.
    """
    cross_corr = []
    for i in range(x.shape[0]):

        x_i = x[i, ...].flatten()
        x_hat_i = x_hat[i, ...].flatten()
        mask_i = mask[i, ...].flatten()

        # Create array with number of burst
        mask_num = utils.artmask2burst(mask_i)

        # Filter by burst
        num_burst = np.unique(mask_num)[1:]
        for burst in num_burst:
            index_burst = np.where(mask_num == burst)[0]

            # Cross Correlation
            corr_burst, max_corr_burst = norm_cross_corr(x_i, x_hat_i, index_burst)
            cross_corr.append(max_corr_burst)

    return np.sum(cross_corr)


def norm_cross_corr(x, x_hat, mask):
    r"""
    Calculate the normalized cross-correlation between two signals.
    Args:
        x (Tensor): True input data.
        x_hat (Tensor): Reconstructed signal.
        mask (Tensor): Mask for the artificial missing.

    Returns:
        Tuple:
            - Normalized cross correlation.
            - Maximum value from the normalized cross correlation.
    """
    x = x[mask] - x[mask].mean()
    x_hat = x_hat[mask] - x_hat[mask].mean()
    cross_corr_signal = np.correlate(x, x_hat, mode="same")
    return cross_corr_signal, cross_corr_signal.max()


def compute_avg_error(x, x_hat, mask, types_list, img_path=".", plot_imp=True):
    r"""
    Computes the avg error and the normalized cross correlation.
    Args:
        x (Tensor): True input data.
        x_hat (Tensor): Reconstructed signal.
        mask (Tensor): Mask for the artificial missing.
        types_list (list of dictionaries): Each dictionary contains: name, type, dim, nclass, index; for every
        attribute.
        img_path (string): Path where the images will be saved.
        plot_imp (boolean): If true, the signals are plotted and saved. Useful for debugging.

    Returns:
        error_obs (dict): Error on observations for every attribute.
        error_miss (dict): Error on missing for every attribute.
        corr_obs (dict): Cross correlation on observations for every attribute.
        corr_miss (dict): Cross correlation on missing for every attribute.
    """

    names = utils.get_var_names(types_list)
    error_obs = dict.fromkeys(names, 0)
    error_miss = dict.fromkeys(names, 0)
    corr_obs = dict.fromkeys(names, 0)
    corr_miss = dict.fromkeys(names, 0)

    for i, type_dict in enumerate(types_list):
        # Fetch type data
        type = type_dict['type']
        var_name = type_dict["name"]
        idx = int(type_dict["index"])

        low_idx = idx
        up_idx = low_idx + 1

        # Filter by idx
        x_type = x[:, :, low_idx:up_idx]  # filter by variables
        x_hat_type = x_hat[:, :, low_idx:up_idx]
        mask_type = mask[:, :, low_idx:up_idx]

        # For GP-VAE and baselines
        if type == "bin":
            x_hat_type = np.round(x_hat_type)

        # Observed
        # if (mask_type == 0).sum() == 0:
        #     error_obs[var_name] += 0
        #     corr_obs[var_name] += 0
        # else:
        #     # Error
        #     if type == 'real' or type == 'pos':  # NRMSE
        #         norm = (x_type.max() - x_type.min()) if x_type.max() != x_type.min() else 1
        #         rmse = mean_squared_error(x_type[mask_type == 0], x_hat_type[mask_type == 0], squared=False)
        #         error_obs_type = rmse / norm
        #
        #     elif type == 'bin':
        #         error_obs_type = np.mean((x_type[mask_type == 0] != x_hat_type[mask_type == 0]))
        #
        #     elif type == 'cat':
        #         conf = confusion_matrix(x_type[mask_type == 0], x_hat_type[mask_type == 0], normalize="true")
        #         error_obs_type = (1 - np.diag(conf)).sum()
        #
        #     error_obs[var_name] += error_obs_type
        #
        #     # Correlation
        #     corr_obs[var_name] += np.abs(np.cov(x_type[mask_type == 1], x_hat_type[mask_type == 1])[0, 1])

        # Missing
        if (mask_type == 1).sum() == 0:
            error_miss[var_name] += 0
            corr_miss[var_name] += 0

        else:
            if type == 'real':
                norm = (x_type.max() - x_type.min()) if x_type.max() != x_type.min() else 1
                rmse = mean_squared_error(x_type[mask_type == 1], x_hat_type[mask_type == 1], squared=False)
                error_miss_type = rmse / norm

            elif type == 'pos':  # NRMSE
                # Negative values, set to 0
                x_type[x_type < 0] = 0
                x_hat_type[x_hat_type < 0] = 0
                if x_type.max() > 1000:   # Human Monitoring. Avoids exploding!
                    x_type = np.log(x_type + 1)
                    x_hat_type = np.log(x_hat_type + 1)
                norm = (x_type.max() - x_type.min()) if x_type.max() != x_type.min() else 1
                rmse = mean_squared_error(x_type[mask_type == 1], x_hat_type[mask_type == 1], squared=False)
                error_miss_type = rmse / norm

            elif type == 'bin':
                error_miss_type = np.mean((x_type[mask_type == 1] != x_hat_type[mask_type == 1]))

            elif type == 'cat':
                conf = confusion_matrix(x_type[mask_type == 1], x_hat_type[mask_type == 1], normalize="true")
                error_miss_type = (1 - np.diag(conf)).sum()
            else:
                error_miss_type = 0

            error_miss[var_name] += error_miss_type

            # Correlation
            corr = burst_cross_correlation(x_type, x_hat_type, mask_type)
            corr = corr / mask_type.sum()

            corr_miss[var_name] += corr

            if plot_imp:
                if type in ["real", "pos"]:
                    # plt.plot(np.log(x_type[mask_type == 1] + 1), "b-", label="real", alpha=0.5)
                    # plt.plot(np.log(x_hat_type[mask_type == 1] + 1), "r--", label="dec", alpha=0.5)
                    plt.plot(x_type[mask_type == 1], "b-", label="real", alpha=0.5)
                    plt.plot(x_hat_type[mask_type == 1], "r--", label="dec", alpha=0.5)
                else:
                    plt.plot(x_type[mask_type == 1], "b-", label="real", alpha=0.5)
                    plt.plot(x_hat_type[mask_type == 1], "r--", label="dec", alpha=0.5)
                plt.legend()
                plt.title(r"${}$ | error: {:.3f} | corr: {:.4f}".format(type_dict["name"],
                                                                        error_miss_type,
                                                                        corr_miss[var_name]))
                plt.savefig(img_path + "/" + type_dict["name"] + ".pdf")
                plt.close()

                if type == "cat":
                    conf = confusion_matrix(x_type[mask_type == 1], x_hat_type[mask_type == 1], normalize="true")
                    sns.heatmap(conf, annot=True)
                    plt.title("{} | error: {:.3f} | corr: {:.4f} ".format(type_dict["name"],
                                                                          error_miss_type,
                                                                          corr_miss[var_name]))
                    plt.savefig(img_path + "/" + type_dict["name"] + "_conf.pdf")
                    plt.close()

    return error_obs, error_miss, corr_obs, corr_miss


def get_type_error(error_miss, types_list):
    r"""
    Obtain the global error for every data type.
    Args:
        error_miss (dict): Error on missing for every attribute.
        types_list (list of dictionaries): Each dictionary contains: name, type, dim, nclass, index; for every
        attribute.

    Returns:

        error_miss_types (dict.): Dictionary with the error values for every data type.

    """
    types = list(set([type_dict['type'] for type_dict in types_list]))
    error_miss_types = dict.fromkeys(types, 0)
    for var, err in error_miss.items():
        type = [type_dict["type"] for type_dict in types_list if type_dict["name"] == var].pop()
        error_miss_types[type] += err
    return error_miss_types


class Loss:
    r"""
    General class for losses.
    """
    def __init__(self):
        self.device = set_device()
        self.eps = 1e-6

    def kld_gauss(self, mean_q, std_q, mean_p, std_p, reduction='sum'):
        r"""
        Calculates the KL for a Gaussian distributions.
        Args:
            mean_q (Tensor): mean of distribution q.
            std_q (Tensor): std of distribution q.
            mean_p (Tensor): mean of distribution p.
            std_p (Tensor): std of distribution p.
            reduction (string): "mean", "sum" can be chosen.

        Returns:
            kl (Tensor): KL term.
        """

        # stability correction
        std_q = torch.clamp(std_q, self.eps, 1e20)
        std_p = torch.clamp(std_p, self.eps, 1e20)

        q = torch.distributions.Normal(mean_q, std_q)
        p = torch.distributions.Normal(mean_p, std_p)
        kl = torch.distributions.kl_divergence(q, p)

        if reduction == 'mean':
            kl = kl.mean()
        elif reduction == 'sum':
            kl = kl.sum()
        else:
            kl = kl  # return vector instead of scalar

        return kl

    @staticmethod
    def kld_cat(prob_hat, prob, reduction='sum'):
        r"""
        Calculates the KL for a Categorical distributions.

        Args:
            prob_hat (Tensor): Probabilities for distribution q.
            prob (Tensor): Probabilities for distribution p.
            reduction (string): "mean", "sum" can be chosen.

        Returns:
            kl (Tensor): KL term.
        """
        q = torch.distributions.Categorical(prob_hat)
        p = torch.distributions.Categorical(prob)
        kl = torch.distributions.kl_divergence(q, p)

        if torch.isnan(kl).sum() > 0:
            # 0 * log (0) ~ 0 to avoid nan
            kl[torch.isnan(kl)] = 0

        if reduction == 'mean':
            kl = kl.mean()
        elif reduction == 'sum':
            kl = kl.sum()
        else:
            kl = kl  # return vector instead of scalar

        return kl

    def nll_gauss(self, mean, std, x, reduction='sum'):
        r"""
        Calculates the nll for a Gaussian distribution.
        Args:
            mean (Tensor): Mean for the Gaussian distribution.
            std (Tensor): Std for the Gaussian distribution.
            x (Tensor): Input data.
            reduction (string): "mean", "sum" can be chosen.

        Returns:
            nll (Tensor): NLL for Gaussian.
        """
        # stability correction
        std = torch.clamp(std, self.eps, 1e20)

        p_x = torch.distributions.Normal(mean, std)
        log_p_x = p_x.log_prob(x)
        nll = -log_p_x

        if reduction == 'mean':
            nll = nll.sum(dim=1).mean()
        elif reduction == 'sum':
            nll = nll.sum()
        else:
            nll = nll  # return vector instead of scalar

        return nll

    @staticmethod
    def nll_bernoulli(theta, x, reduction='sum'):
        r"""
        Calculates the nll for a Bernoulli distribution.

        Args:
            theta (Tensor): Probabilities for the Bernoulli distribution.
            x (Tensor): Input data.
            reduction (string): "mean", "sum" can be chosen.

        Returns:
            nll (Tensor): NLL for Bernoulli.

        """
        p_x = torch.distributions.Bernoulli(logits=theta)
        log_p_x = p_x.log_prob(x)
        nll = -log_p_x

        if reduction == 'mean':
            nll = nll.sum(dim=1).mean()
        elif reduction == 'sum':
            nll = nll.sum()
        else:
            nll = nll  # return vector instead of scalar

        return nll

    @staticmethod
    def nll_categorical(theta, x, reduction='sum'):
        r"""
        Calculates the nll for a Categorical distribution.
        Args:
            theta (Tensor): Probabilities for the Categorical distribution.
            x (Tensor): Input data.
            reduction (string): "mean", "sum" can be chosen.

        Returns:
            nll (Tensor): NLL for Categorical.

        """
        p_x = torch.distributions.one_hot_categorical.OneHotCategorical(logits=theta)
        log_p_x = p_x.log_prob(x)
        nll = -log_p_x

        if reduction == 'mean':
            nll = nll.sum(dim=1).mean()
        elif reduction == 'sum':
            nll = nll.sum()
        else:
            nll = nll  # return vector instead of scalar

        return nll

    @staticmethod
    def MSE(x, x_hat):
        r"""
        Calculate the MSE.
        Args:
            x (Tensor): Input data.
            x_hat (Tensor): Reconstruction data.

        Returns:
            MSE
        """
        return (x - x_hat) ** 2

    def RMSE(self, x, x_hat):
        r"""
        Calculate the RMSE.
        Args:
            x (Tensor): Input data.
            x_hat (Tensor): Reconstruction data.

        Returns:
            RMSE
        """
        return torch.sqrt(self.MSE(x, x_hat))

    @staticmethod
    def NRMSE(x, x_hat):
        r"""
        Calculate th NRMSE.
        Args:
            x (Tensor): Input data.
            x_hat (Tensor): Reconstruction data.

        Returns:
            NRMSE
        """
        norm = (x.max() - x.min()) if x.max() != x.min() else 1
        return np.sqrt(mean_squared_error(x, x_hat)) / norm
