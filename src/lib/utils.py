# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021. by Daniel Barrejon, UC3M.                                         +
#  All rights reserved. This file is part of the Shi-VAE, and is released under the      +
#  "MIT License Agreement". Please see the LICENSE file that should have been included   +
#   as part of this package.                                                             +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import copy
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn

latex_losses = {"n_elbo": "-ELBO",
                "nll_loss": "$\sum$ NLL$ _\ell$",
                "nll_real": "NLL$_{real}$",
                "nll_pos": "NLL$_{pos}$",
                "nll_bin": "NLL$_{bin}$",
                "nll_cat": "NLL$_{cat}$",
                "kld": "$\sum$KL",
                'kld_q_z': "KL$[q(Z)]$",
                'kld_q_s': "KL$[q(S)]$"
                }


# =========== #
# MODEL UTILS
# =========== #
def save_theta_decoder(likes, types_list, path):
    r"""
    Save the parameters of the decoder as plots.
    Function for debugging. Mostly for discrete (categorical and binary) data.
    Args:
        likes (dict.): Dictionary with the decoder likelihoods.
        types_list (list of dictionaries): Each dictionary contains: name, type, dim, nclass, index; for every
            attribute.
        path (string): Path to save plots.
    """
    # Create checkpoint directory
    theta_path = os.path.join(path, "theta_decoder")
    if not os.path.exists(theta_path):
        os.makedirs(theta_path)

    for type_dict, (var, lik) in zip(types_list, likes.items()):
        type = type_dict["type"]
        if type in ["real", "pos"]:
            mean, std = np.array(lik.loc.data), np.array(lik.scale.data)
            mean_reshape = mean.reshape(mean.shape[0] * mean.shape[1], -1)
            std_reshape = std.reshape(mean.shape[0] * std.shape[1], -1)
            sns.heatmap(mean_reshape[:50, :])
            plt.savefig(os.path.join(theta_path, var + "_" + type + "_mean.pdf"))
            plt.close()
            sns.heatmap(std_reshape[:50, :])
            plt.savefig(os.path.join(theta_path, var + "_" + type + "_std.pdf"))
            plt.close()
        elif type in ["bin", "cat"]:
            p = np.array(lik.probs.data)
            p_reshape = p.reshape(p.shape[0] * p.shape[1], -1)
            sns.heatmap(p_reshape[:50, :])
            plt.savefig(os.path.join(theta_path, var + "_" + type + "_prob.pdf"))
            plt.close()


def set_activation_layer(activation_layer):
    r"""
    Choose which type of activation layer used in the model.

    Args:
        activation_layer (string): "relu", "tanh" or "sigmoid".

    Returns:
        Activation layer module.
    """
    if activation_layer.lower() == 'relu':
        return nn.ReLU()
    elif activation_layer.lower() == 'tanh':
        return nn.Tanh()
    elif activation_layer.lower() == 'sigmoid':
        return nn.Sigmoid()
    else:
        print("Activation Layer not found. Default: ReLU")
        return nn.ReLU()


def zero_fill(x, mask):
    r"""
    Fill positions with missing data with 0s.
    Args:
        x (Tensor): Data
        mask (Tensor): Mask where to put zeros.

    Returns:
        Data filled with zeros at mask.
    """
    x_filled = torch.where(mask, torch.zeros_like(x), x)
    return x_filled


def list2tupandstack(list_tups):
    r"""
    Convert a list of tuples, i.e, [('a','b'), ..., ('c','d')] into a tuple of list, i.e., ([a,...,c], [b,...,d]).
    Then, the lists are stacked.
    Args:
        list_tups (list of tuples):
    Returns:
        Stacked tuple of lists.
    """
    if type(list_tups[0]) is not torch.Tensor:

        tup = zip(*list_tups)
        l_tup = []
        for t in tup:
            l_tup.append(torch.stack(t))
        return l_tup
    else:
        return torch.stack(list_tups)


def chop_sequence(T, max_length=50):
    r"""
    Divide a signal into subsequences, subsampled by length given by :attr:max_length.
    Args:
        T (int): Length of the signal.
        max_length (int, optional): Max. length.

    Returns:
        Range of times.
    """
    # ======= Truncate long sequences ======= #
    if T > max_length:
        # Create sequence lengths range
        t_range = np.arange(max_length, T, max_length)
        if (T not in t_range) == True:
            t_range = np.append(t_range, T)
    else:
        t_range = [T]  # do not need minibatch
    return t_range


def create_lossdict(types, device):
    r"""
    Create a dictionary for the losses, with types given by :attr:types.
    Also initialize with 0 and send to device.
    Args:
        types (list): List of types.
        device : CPU or GPU.

    Returns:
        Dictionary witgh losses.
    """
    keys = ["nll_" + t for t in types]
    loss_dict = dict.fromkeys(keys)
    for key in loss_dict.keys():
        loss_dict[key] = torch.tensor(0, dtype=torch.float).to(device)
    return loss_dict


def check_nan_losses(nll_heter_dict):
    r"""
    Checks if a loss inside the dictionary :attr:nll_heter_dict is NaN.
    Args:
        nll_heter_dict (dict): Dictionary with nll.

    Returns:
        If there is any NaN, program stops and prints the value.
    """
    for key, nll in nll_heter_dict.items():
        if torch.isnan(nll):
            print("{} Loss NAN".format(key))
            print(nll)
            exit()
        else:
            pass


def write_csv_types(row_list, name):
    r"""
    Write data_types.csv.
    Args:
        row_list (list): List with values.
        name (string): Name of the file.

    Example:
        row_list = [["name", "type", "dim", "nclass", "index"],
                    ["var0", "real", 1, 1, 0],
                    ["var1", "pos", 1, 1, 1],
                    ["var2", "real", 1, 1, 2]]

    """

    with open(name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(row_list)


def read_csv_types(csv_name):
    r"""
    Read data_types.csv.
    Args:
        csv_name (string): Name of the data_types.csv

    Returns:
        List of dictionaries for each type.
    """
    with open(csv_name) as f:
        types_list = [{k: v for k, v in row.items()}
                      for row in csv.DictReader(f, skipinitialspace=True)]
    return types_list


def get_var_names(types_list):
    r"""
    Obtain the name of the attributes from :attr:types_list.
    Args:
        types_list (list of dictionaries): Each dictionary contains: name, type, dim, nclass, index; for every
            attribute.

    Returns:
        list of attribute's names.
    """
    var_names = [type_dict['name'] for type_dict in types_list]
    return var_names


def get_x_dim(types_list):
    r"""
    Return dimensionality of data. The original dimension of the data is expanded if categorical data is available
    + number of classes.
    Args:
        types_list (list of dictionaries): Each dictionary contains: name, type, dim, nclass, index; for every
            attribute.

    Returns:
        Total number of dimensions.
    """
    dims = [int(type_dict['dim']) for type_dict in types_list]
    return np.sum(dims)


def get_idx(types_list):
    r"""
    Obtain indexes of every attribute.
    Args:
        types_list (list of dictionaries): Each dictionary contains: name, type, dim, nclass, index; for every
            attribute.

    Returns:
        list of indexes.
    """
    idx = [int(type_dict['index']) for type_dict in types_list]
    return idx


def filter_csv_types(types_list, filter_names):
    r"""
    Auxiliary function useful for performing experiments on subsets of variables.
    Args:
        types_list (list of dictionaries): Each dictionary contains: name, type, dim, nclass, index; for every
            attribute.
        filter_names (list): List of the variables to be used.

    Returns:
        filtered types_list
    """
    types_list = [type_dict for
                  type_dict in types_list if type_dict['name'] in filter_names]

    return types_list


def reindex_types_list(types_list):
    r"""
    Reindex types_list indexes in increasing order. Useful when performing experiments on desired variables.
    Args:
        types_list (list of dictionaries): Each dictionary contains: name, type, dim, nclass, index; for every
        attribute.
    Returns:
        reindexed types_list
    """
    # Re-index in increasing order.
    types_list_copy = copy.deepcopy(types_list)
    for i, type_dict in enumerate(types_list_copy):
        type_dict.update({'index': i})
    return types_list_copy


def get_idx_transform(types_list):
    r"""
    Get the idx with for the transformed data. Once the transformation is applied with the scaler, the data expands
    in dimensionality if discrete data is present. Therefore, we need the new indexes.
    Args:
        types_list (list of dictionaries): Each dictionary contains: name, type, dim, nclass, index; for every
        attribute.

    Returns:
        list of indexes.
    """
    transform_idx = []
    dim = 0
    for type_dict in types_list:
        if dim == 0:  # initial index
            transform_idx.append(0)
        elif dim > 1:  # for datatypes with more than 1 classes
            last_idx = transform_idx[-1]
            transform_idx.append(last_idx + dim)
        else:  # datatypes with one class
            last_idx = transform_idx[-1]
            transform_idx.append(last_idx + 1)
        dim = int(type_dict['dim'])

    return transform_idx


#  ===== OTHERS UTILS ===== #
def dist_moments(data):
    r"""
    Calculate mean and standard deviation, avoiding the missing data.
    Args:
        data (Tensor): Data

    Returns:
        mean and std
    """
    mean = np.nanmean(data)
    std = np.nanstd(data)
    std = np.clip(std, a_min=1e-6, a_max=1e20)  # clip values
    return mean, std


def losses2latexformat(losses):
    r"""
    Return a list of string losses in proper latex format.
    Args:
        losses (list): List of string losses.

    Returns:
        list of strings in latex format.
    """
    return [latex_losses[loss] for loss in losses]


def artmask2burst(mask):
    r"""
    Returns an array with the burst of artificial missing encoded with the number of the burst.

    Args:
        mask (Tensor): Missing mask.

    Example:
         [0,1,1,1,0,0,0,1,1,1] = [0,1,1,1,0,0,0,2,2,2]
    Returns:
        mask encoded with bursts of missing.
    """
    mask = mask.flatten()

    # Create array with number of burst
    # Index missing
    index_missing = np.where(mask)[0]

    # Create array with number of burst
    ones_mask = []
    j = 1

    # Only 1 missing value
    if len(index_missing) == 1:
        ones_mask.append(j)

    else:
        for i in range(len(index_missing)):
            if (i + 1) == len(index_missing):  # last index check
                if index_missing[i] - index_missing[i - 1] == 1:  # same burst
                    ones_mask.append(ones_mask[i - 1])
                else:
                    ones_mask.append(ones_mask[i - 1] + 1)  # different burst
                continue

            if index_missing[i + 1] - index_missing[i] == 1:  # same burst
                ones_mask.append(j)
            else:
                ones_mask.append(j)  # next, different burst
                j += 1

    # Create burst array with each burst number
    mask_num = np.zeros_like(mask).astype(int)
    mask_num[index_missing] = ones_mask
    return mask_num
