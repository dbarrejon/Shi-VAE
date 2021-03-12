"""
Scaler

This module contains the implementation of an Heterogeneous scaler, plus the individual scalers for
different distributions.

Available Distribution:
    - Real distribution: normalize + standarize.
    - Positive distribution: normalize + standarize applied to the log(data)
    - Categorical Distribution
"""


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021. by Daniel Barrejon, UC3M.                                         +
#  All rights reserved. This file is part of the Shi-VAE, and is released under the      +
#  "MIT License Agreement". Please see the LICENSE file that should have been included   +
#   as part of this package.                                                             +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
from sklearn.preprocessing import OneHotEncoder

from lib import utils


class HeterogeneousScaler:
    r"""
    Heterogeneous scaler
    """

    def __init__(self, types_list=None):
        r"""
        Initialize heterogeneous scaler.
        Args:
            types_list (list of dicts): dictionary with the information for every attribute.
        """
        assert types_list is not None
        self.types_list = utils.reindex_types_list(types_list)
        self.scalers = []
        # Transformed index to handle variables with more classes.
        self.transform_idx = utils.get_idx_transform(self.types_list)

        for type_dict in self.types_list:
            type = type_dict['type']
            if type == 'real':
                scaler = StandardScaler()
            elif type == 'pos':
                scaler = PositiveScaler()
            elif type == 'bin':
                scaler = None
            elif type == 'cat':
                R = int(type_dict['nclass'])
                scaler = CategoricalScaler(R=R)
            else:
                scaler = StandardScaler()

            self.scalers.append(scaler)

    def fit(self, data, mask=None):
        r"""
        Fits every scaler defined in the Heterogeneous scaler.
        Args:
            data (Tensor): Data of shape (NxTxD).
            mask (Tensor, optional): Mask of shape (NxTxD).

        Returns:

        """
        if mask is None:  # assume all observed
            mask = np.isnan(data)

        n_samples = np.prod(data.shape[:-1])
        data = np.reshape(data, (n_samples, data.shape[-1]))
        mask = np.reshape(mask, (n_samples, mask.shape[-1]))

        # Fill with NaN, because scalers take into account NaN values.
        data_nan = np.where(mask, np.NaN, data)
        for i, (type_dict, scaler) in enumerate(zip(self.types_list, self.scalers)):
            type = type_dict['type']
            idx = int(type_dict['index'])
            if type != "bin":
                scaler.fit(data_nan[:, idx])

    def transform(self, data, mask=None):
        r"""
        Transform the heterogeneous data with the corresponding scalers.
        Args:
            data (Tensor): Data of shape (NxTxD).
            mask (Tensor, optional): Mask of shape (NxTxD).

        Returns:
            transformed_data (Tensor): This data has more dimension, depending on the discrete data which can be encoded
            as one-hot. Shape(NxTxnew_D)
            transformed_mask (Tensor): This mask has more dimension, depending on the discrete data which can be encoded
            as one-hot. Shape(NxTxnew_D)
        """
        if mask is None:
            mask = np.isnan(data)

        transform_x_dim = utils.get_x_dim(self.types_list)
        transform_shape = [data.shape[0], data.shape[1], transform_x_dim]
        transform_data = np.zeros(transform_shape)
        transform_mask = np.zeros(transform_shape, dtype=bool)

        for i, type_dict in enumerate(self.types_list):
            scaler = self.scalers[i]
            idx = int(type_dict['index'])
            dim = int(type_dict['dim'])

            # Filter
            data_type = data[..., idx]
            data_type = data_type[..., np.newaxis]
            mask_type = mask[..., idx]
            mask_type = mask_type[..., np.newaxis]

            # Categorical mask
            if dim > 1:
                mask_type = np.repeat(mask_type, dim, axis=-1)

            # Real, Pos, Cat
            if scaler is not None:
                data_type = scaler.transform(data_type)

            transform_idx = self.transform_idx[i]
            low_idx = transform_idx
            up_idx = transform_idx + dim

            transform_data[..., low_idx:up_idx] = data_type
            transform_mask[..., low_idx:up_idx] = mask_type
        return transform_data, transform_mask

    def inverse_transform(self, data, mask=None, transpose=True):
        r"""

        Heterogeneous inverse transformation. For samples with heterogeneous data.

        Note:
            Assumes data and mask are numpy. ASSUMES DATA AND MASK ARE NUMPY

        Args:
            Shape(NxTxnew_D)
            data (Tensor): Data of shape (NxTxnew_D).
            mask (Tensor, optional): Mask of shape (NxTxnew_D).
            transpose (boolean, optional): If True, transpose the data to have (TxNxD)

        Returns:
            original_data (Tensor): Data of shape (NxTxD).
            original_mask (Tensor, optional): Mask of shape (NxTxD).
        """
        if mask is None:
            mask = np.isnan(data)
        if not isinstance(data, np.ndarray):
            data = np.array(data.data)

        original_x_dim = len(self.types_list)
        original_shape = [data.shape[0], data.shape[1], original_x_dim]
        original_data = np.zeros(original_shape)
        original_mask = np.zeros(original_shape, dtype=bool)

        for i, type_dict in enumerate(self.types_list):
            scaler = self.scalers[i]
            idx = int(type_dict['index'])
            dim = int(type_dict['dim'])

            transform_idx = self.transform_idx[i]
            low_idx = transform_idx
            up_idx = transform_idx + dim

            data_type = data[..., low_idx:up_idx]
            mask_type = mask[..., low_idx]  # filter mask

            if scaler is not None:
                data_type = scaler.inverse_transform(data_type)

            original_data[..., idx] = data_type.squeeze(axis=-1)
            original_mask[..., idx] = mask_type

        if transpose:
            original_data = original_data.transpose(1, 0, 2)
            original_mask = original_mask.transpose(1, 0, 2)

        return original_data, original_mask

    def param_inverse_transform(self, likes, transpose=True):
        r"""
        Transform the parameters estimated by the model.
        Inverse transform, permute to BxTxD and to numpy.
        Args:
            likes (dict): Dictionary with the likelihoods of the model.
            transpose (boolean, optional): If true, tranpose data.

        Returns:
            dec_params(dict): Dictionary with the likelihoods of the model, with the inverse transformed.
        """
        dec_params = {}
        for i, type_dict in enumerate(self.types_list):
            scaler = self.scalers[i]
            var = type_dict['name']
            type = type_dict['type']
            lik = likes[var]
            if type in ['real', 'pos']:
                dec_mean, dec_std = np.array(lik.loc.data), np.array(lik.scale.data)
                dec_trans_mean = scaler.inverse_transform(dec_mean)
                dec_trans_std = scaler.std_inverse_transform(dec_std)
                if transpose:
                    dec_trans_mean = dec_trans_mean.transpose(1, 0, 2)
                    dec_trans_std = dec_trans_std.transpose(1, 0, 2)
                dec_params[var] = [dec_trans_mean, dec_trans_std]

            elif type in ['bin', 'cat']:
                # No inverse transformation is applied on dec_p
                dec_p = np.array(lik.probs.data)
                if transpose:
                    dec_p = dec_p.transpose(1, 0, 2)
                dec_params[var] = dec_p
            else:
                pass

        return dec_params


class StandardScaler:
    r"""
    Scaler for real values.

    References:
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    """
    def __init__(self):
        self.mean = 0
        self.std = 0

    def fit(self, data):
        self.mean, self.std = utils.dist_moments(data)
        return

    def transform(self, data):
        data = (data - self.mean) / self.std
        return data

    def inverse_transform(self, data):
        # for data and mean
        data = data * self.std + self.mean
        return data

    def std_inverse_transform(self, std_hat):
        # for std
        std_hat = std_hat * self.std  # std
        return std_hat


class PositiveScaler:
    """
    Scaler for positive variables.
    Similar to standard scaler, but applying to the data the following transformation.
    .. math::
        \tilde{x} = \log(x + 1)
    """
    def __init__(self):
        self.mean = 0
        self.std = 0

    def fit(self, data):
        # apply log to the data.
        data_log = data.copy()
        data_log[~np.isnan(data)] = np.log(data[~np.isnan(data)] + 1)
        self.mean, self.std = utils.dist_moments(data_log)
        return

    def transform(self, data):
        data_log = data.copy()
        data_log[~np.isnan(data)] = np.log(data[~np.isnan(data)] + 1)
        data_log = (data_log - self.mean) / self.std
        return data_log

    def inverse_transform(self, data_log):
        # 1. de-standarize
        data_log = data_log * self.std + self.mean

        # 2. Undo transformation
        data = np.exp(data_log) - 1
        return data

    def std_inverse_transform(self, std_hat):
        # for std
        std_hat = std_hat * self.std
        std_hat = np.exp(std_hat) - 1
        return std_hat


class CategoricalScaler:
    """
    Scaler for Categorical data.
    Transforms into one-hot encoded.
    """
    def __init__(self, R=None):
        assert R is not None
        self.R = R  # num. categories
        self.enc = OneHotEncoder(sparse=False)

    def fit(self, data):
        data_copy = data.copy()
        data_copy[np.isnan(data_copy)] = 0  # Fill nan with.
        self.enc.fit(data_copy[..., np.newaxis])
        return

    def transform(self, data):
        mask = np.repeat(np.isnan(data), self.R, axis=-1)
        data_join = data.reshape(data.shape[0] * data.shape[1], -1).copy()  # B*T x 1
        data_join[np.isnan(data_join)] = 0  # Fill nan with.
        onehot_join = self.enc.transform(data_join)
        onehot_data = onehot_join.reshape(data.shape[0], data.shape[1], -1)  # B*T x R
        onehot_data[mask != 0] = np.NaN  # Set back to NaN
        return onehot_data

    @staticmethod
    def inverse_transform(onehot_data):
        # one_hot_data = softmax(onehot_data_logit, axis=-1)
        mask = np.isnan(onehot_data)[..., -1]
        data = np.argmax(onehot_data, axis=-1).astype(float)
        data[mask != 0] = np.NaN
        return data[..., np.newaxis]
