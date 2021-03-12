# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021. by Daniel Barrejon, UC3M.                                         +
#  All rights reserved. This file is part of the Shi-VAE, and is released under the      +
#  "MIT License Agreement". Please see the LICENSE file that should have been included   +
#   as part of this package.                                                             +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import pickle

import numpy as np
from torch.utils import data

from lib import utils


class HeterDataset(data.Dataset):
    r"""
    Base dataset class for heterogeneous data.
    """
    def __init__(self, data, mask, data_full=None, mask_artificial=None, types_list=None):
        r"""

        Args:
            data (Tensor): Input data of shape (NxTxD).
            mask (Tensor): Input data of shape (NxTxD).
            data_full (Tensor): Real data of shape (NxTxD).
            mask_artificial (Tensor): Artificial missing (points observed values), with shape (NxTxD).
            types_list (list of dictionaries): Each dictionary contains: name, type, dim, nclass, index; for every
            attribute.
        """
        assert types_list is not None
        self.data = data  # TxBxD
        self.mask = mask
        self.mask_artificial = mask_artificial if mask_artificial is not None else mask
        self.data_full = data_full if data_full is not None else data
        self.eval_opt = False
        self.set_eval()

        # Data type handling.
        self.types_list = types_list
        self.idx_vars = utils.get_idx(self.types_list)
        self.idx_vars.sort()  # sort in ascending order

    def set_eval(self, eval_opt=False):
        r"""
        For validation purposes, such as computing nll or MSE on artificial missing mask, set eval_opt to True.
        Args:
            eval_opt (boolean): Evaluation option inside dataset.
        """
        self.eval_opt = eval_opt

    def __len__(self):
        r"""
        Denotes the total number of samples
        Returns:
            Length of the dataset, N.
        """
        return len(self.data)

    def __getitem__(self, index):
        r"""
        Return the sample. Used together with DataLoader from Pytorch.
        Args:
            index: Index to get the sample.

        Returns:
            sequence, mask, index. If :attr:eval_opt is True, returns also the complete data and the artificial mask.
        """
        # Select sample (NxTxD)
        sequence = self.data[index, :, self.idx_vars].transpose()
        mask = self.mask[index, :, self.idx_vars].transpose()
        if self.eval_opt:
            data_full = self.data_full[index, :, self.idx_vars].transpose()
            mask_artificial = self.mask_artificial[index, :, self.idx_vars].transpose()
            return sequence, mask, data_full, mask_artificial, index
        else:  # return same mask
            return sequence, mask, index


class HMMDatset(HeterDataset):
    """
    Synthetic Dataset for Heterogeneous HMM.
    """
    def __init__(self, data, mask, data_full=None, mask_artificial=None, types_list=None,
                 index=None, Z=None, theta=None):
        r"""
        Initialize HMM Dataset.
        Args:
            data (Tensor): Input data of shape (NxTxD).
            mask (Tensor): Input data of shape (NxTxD).
            data_full (Tensor): Real data of shape (NxTxD).
            mask_artificial (Tensor): Artificial missing (points observed values), with shape (NxTxD).
            types_list (list of dictionaries): Each dictionary contains: name, type, dim, nclass, index; for every
            attribute.
            index (list of int): List of indexes to filter the complete dataset.
            Z (matrix): States for the HMM, of shape (NxT)
            theta (dict.): Dictionary with the parameters of the emission parameters for the Heterogeneous HMM.

        """
        super(HMMDatset, self).__init__(data, mask, data_full=data_full, mask_artificial=mask_artificial,
                                        types_list=types_list)

        # HMM Parameters
        self.Z = Z
        self.theta = theta
        self.index = index

        # Filter by index
        self.data = self.data[self.index]
        self.data_full = self.data_full[self.index]
        self.mask = self.mask[self.index]
        self.mask_artificial = self.mask_artificial[self.index]
        self.Z = self.Z[index]

    def __len__(self):
        r"""
        Denotes the total number of samples
        Returns:
            Length of the dataset, N.
        """
        return len(self.index)

    def __getitem__(self, index):
        r"""
        Return the sample. Used together with DataLoader from Pytorch.
        Args:
            index: Index to get the sample.

        Returns:
            sequence, mask, index. If :attr:eval_opt is True, returns also the complete data and the artificial mask.
        """
        # Select sample (BxTxD)
        sequence = self.data[index, :, self.idx_vars].transpose()
        mask = self.mask[index, :, self.idx_vars].transpose()
        if self.eval_opt:
            data_full = self.data_full[index, :, self.idx_vars].transpose()
            mask_artificial = self.mask_artificial[index, :, self.idx_vars].transpose()
            return sequence, mask, data_full, mask_artificial, index
        else:  # return same mask
            return sequence, mask, index


def standard_collate(batch):
    r"""
    Custom collate_fn for the Dataloader. This function returns the data and the missing mask, with the complete data
    and the artificial missing if :attr:eval_opt was set to True in the HeterDataset.
    It also return some additional information in the batch_attributes.
    Args:
        batch (list): List containing the data returned from _get_item() in the dataset.

    Returns:
        The data, the missing mask (and the complete data and the artificial mask if :attr:eval_opt was True), together
        with some information in the batch_attributes.
    """
    if len(batch[0]) == 3:  # not eval_opt
        data, mask, index = zip(*batch)
        # 1. Stack: BxTxD
        data = np.stack(data)
        mask = np.stack(mask).astype(bool)
        index = np.stack(index)

    elif len(batch[0]) > 3:  # eval_opt
        data, mask, data_full, mask_artificial, index = zip(*batch)

        # 1. Stack: BxTxD
        data = np.stack(data)
        data_full = np.stack(data_full)
        mask = np.stack(mask).astype(bool)
        mask_artificial = np.stack(mask_artificial).astype(bool)
        index = np.stack(index)
    else:
        data, mask, index = zip(*batch)
        # 1. Stack: BxTxD
        data = np.stack(data)
        mask = np.stack(mask).astype(bool)
        index = np.stack(index)

    B = data.shape[0]
    T = data.shape[1]
    dates = [d for d in range(T)]
    seq_lengths = B * [T]
    batch_lengths = B * [B]

    batch_attributes = {'dates': dates,
                        'sequence_lengths': seq_lengths,
                        'batch_lengths': batch_lengths,
                        'index': index}

    if len(batch[0]) == 3:  # not eval_opt
        return data, mask, batch_attributes
    elif len(batch[0]) > 3:  # eval_opt
        return data, mask, data_full, mask_artificial, batch_attributes
    else:
        return data, mask, batch_attributes


def load_pickle(data_path, percent_miss=0, mask=1):
    r"""
    Load data from HMM saved in pickle format.
    Args:
        data_path (string): Path for the pickle file.
        percent_miss (int): Missing rate : 0, 10, 30, 50, 70.
        mask (int): Index for the partition of the mask [1-10]

    Returns:
        X (Tensor): Data NxTxD.
        Z (Tensor): States for HMM NxT.
        theta (dict): Dictionary with the parameters of the emission distributions.
        mask (Tensor): Mask NxTxD.
        index (tuple of list): Tuple of dimension 3 with the indexes for train, val and test.
    """
    with open(data_path, 'rb') as handle:
        data = pickle.load(handle)

    X = data["X"]
    Z = data['Z']  # states
    mask = data["mask_" + str(percent_miss) + "_" + str(mask)]
    theta = data["theta"]
    index = (data["train_index"], data["val_index"], data["test_index"])
    # shape: N x T x D
    return X, Z, theta, mask, index


def load_data_pickle(dataset, data_path, percent_miss=0, mask_i=1):
    r"""
    Function for loading the data from a pickle
    Args:
        dataset (string): Name of the dataset.
        data_path (string): Folder containing the data.
        percent_miss (int): Missing percentage.
        mask_i (int): Partition for the mask [1-10]

    Returns:
        X (Tensor): Data NxTxD.
        Z (Tensor): States for HMM NxT.
        theta (dict): Dictionary with the parameters of the emission distributions.
        mask (Tensor): Mask NxTxD.
        index (tuple of list): Tuple of dimension 3 with the indexes for train, val and test.
    """

    # data, mask = load_mat(os.path.join(data_path, dataset), percent_miss=percent_miss)
    X, Z, theta, mask, index = load_pickle(os.path.join(data_path, dataset, dataset+".pickle"),
                                           percent_miss=percent_miss, mask=mask_i)

    # DONE IN DATASET CLASS
    # Shape: TxBxD
    # x_train = np.transpose(x_train, (1, 0, 2))
    # mask_train = np.transpose(mask_train, (1, 0, 2))
    # x_valid = np.transpose(x_valid, (1, 0, 2))
    # mask_valid = np.transpose(mask_valid, (1, 0, 2))
    # x_test = np.transpose(x_test, (1, 0, 2))
    # mask_test = np.transpose(mask_test, (1, 0, 2))

    # print('x_train: {} {} MB'.format(x_train.shape, x_train.nbytes / 1e6))
    # print('x_valid: {} {} MB'.format(x_valid.shape, x_valid.nbytes / 1e6))
    # print('x_test: {} {} MB'.format(x_test.shape, x_test.nbytes / 1e6))

    # DONE IN STANDARD COLLATE FN
    # To Tensor
    # x_train = torch.tensor(x_train)
    # mask_train = torch.tensor(mask_train)
    # x_valid = torch.tensor(x_valid)
    # mask_valid = torch.tensor(mask_valid)
    # x_test = torch.tensor(x_test)
    # mask_test = torch.tensor(mask_test)

    return X, Z, theta, mask, index


# NOT USED
def split_train_test(data, mask, perc):
    r"""
    Splits the data into two sets: train and test. The :attr:perc value is assigned to train.
    Args:
        data (Tensor): Shape (NxTxD)
        mask (Tensor): Shape (NxTxD)
        perc (float): Percentage for training set.

    Returns:
        data_train
        mask_train
        data_test
        mask_test

    """
    train_size = int(data.shape[0] * perc)
    # test_size = data.shape[0] - train_size
    data_train = data[:train_size]
    mask_train = mask[:train_size]
    data_test = data[train_size:]
    mask_test = mask[train_size:]
    return data_train, mask_train, data_test, mask_test
