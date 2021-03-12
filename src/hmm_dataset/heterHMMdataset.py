"""
Generate a synthetic dataset with heterogeneous streams of data: real, positive, binary and categorical.
"""
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021. by Daniel Barrejon, UC3M.                                         +
#  All rights reserved. This file is part of the Shi-VAE, and is released under the      +
#  "MIT License Agreement". Please see the LICENSE file that should have been included   +
#   as part of this package.                                                             +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
# Add sources path to run properly.
module_path = os.path.abspath(os.path.join('../../src'))
if module_path not in os.sys.path:
    os.sys.path.insert(1, module_path)

import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn
import seaborn as sns
from hmmlearn import hmm
from tqdm import tqdm

from lib.utils import write_csv_types

# Seaborn Options
seaborn.set()
seaborn.set_context("paper")

prng = np.random.RandomState(10)

def plot_sequence(X, name):
    f = plt.figure()
    plt.plot(X)
    f.savefig(name)
    plt.close()

def permute_mask(mask):
    bs = mask.shape[0]
    mask_old = mask.copy()
    for batch in range(bs):
        np.random.shuffle(mask[batch])  # in_place operation!
    return mask, mask_old


class ToyDatasetGenerator():
    def __init__(self, x_real_dim=2, x_pos_dim=2, x_bin_dim=1, x_cat_dim=1, x_cat_class=[3], n_components=4, T=48,
                 N=1000):
        # Heterogeneous dimensions.
        self.x_real_dim = x_real_dim
        self.x_pos_dim = x_pos_dim
        self.x_bin_dim = x_bin_dim
        self.x_cat_dim = x_cat_dim
        self.x_cat_class = x_cat_class
        self.X_dim = self.x_real_dim + self.x_pos_dim + self.x_bin_dim + self.x_cat_dim
        self.n_components = n_components
        self.T = T
        self.N = N

        # Parameters for every distribution
        self.theta_list = []
        self.theta = {}

        # Folders
        self.toy_name = "hmm_heter_{}_{}real_{}pos_{}bin_{}cat_{}_{}".format(self.N, self.x_real_dim, self.x_pos_dim,
                                                                             self.x_bin_dim, self.x_cat_dim,
                                                                             self.n_components, self.T)
        self. save_dir = os.path.join("../../data", self.toy_name)
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)

        # ======== HMM Parameters ======== #
        if self.n_components == 3:
            self.transmat = np.array([[0.9, 0.1, 0],
                                      [0., 0.9, 0.1],
                                      [0.1, 0., 0.9]])
            self.startprob = np.array([0.7, 0.2, 0.1])

        elif self.n_components == 4:
            self.transmat = np.array([[0.7, 0.2, 0.0, 0.1],
                                      [0.3, 0.5, 0.2, 0.0],
                                      [0.0, 0.3, 0.5, 0.2],
                                      [0.2, 0.0, 0.2, 0.6]])
            self.startprob = np.array([0.6, 0.3, 0.1, 0.0])

        # For greater dimensions, create random.
        else:
            self.transmat = np.random.rand(self.n_components, self.n_components)
            self.transmat = self.transmat / self.transmat.sum(axis=1)[:, None]
            self.startprob = np.random.rand(1, self.n_components)
            self.startprob = self.startprob / self.startprob.sum(axis=1)[:, None]

        # ======== HMM ======== #
        self.gaussian_hmm = self.init_gaussian_hmm(dim=self.x_real_dim, type='gauss')
        self.positive_hmm = self.init_gaussian_hmm(dim=self.x_pos_dim, type='positive')
        self.binary_hmm_list = self.init_binary_hmm(dim=self.x_bin_dim)
        self.categorical_hmm_list = self.init_cat_hmm(dim=self.x_cat_dim)

    def init_gaussian_hmm(self, dim=3, std=5, type='gauss'):
        means, covars = self.generate_emission_gauss(dim=dim)

        # Build an HMM instance and set parameters
        model = hmm.GaussianHMM(n_components=self.n_components, covariance_type="full")
        model.startprob_ = self.startprob
        model.transmat_ = self.transmat
        model.means_ = means
        model.covars_ = covars

        if type == 'gauss':
            self.means_gauss = means
            self.covars_gauss = covars
        elif type == 'positive':
            self.means_pos = means
            self.covars_pos = covars

        # Append to parameters
        means_list = [mu for mu in means.transpose()]
        covars_list = [covars for i in range(dim)]
        theta = [{"mean": mu, "covar": var} for (mu, var) in zip(means_list, covars_list)]
        self.theta_list = self.theta_list + theta

        return model

    def generate_emission_gauss(self, dim):
        """
        Generate emission probabilities for Guassian HMM. This way the means for each state are rather appart and we
        avoid overlap.
        :param dim:
        :return:
        """
        num_steps = 20
        limit = 20
        means = np.linspace(-limit, limit, num_steps)
        steps = np.round(np.linspace(0, limit, self.n_components+1))
        means_list = []
        for d in range(dim):
            for state in range(self.n_components):
                idx = np.random.randint(steps[state], steps[state+1])
                means_list.append(means[idx])
        means = np.array(means_list)
        means = means.reshape([self.n_components, -1])
        np.random.shuffle(means)
        covars = .5 * np.tile(np.identity(dim), (self.n_components, 1, 1))

        return means, covars

    def init_binary_hmm(self, dim=1):
        """
        Create a list of binary HMM depending on the dimension. All share the start and transition probabilities, but each
        binary HMM has a different emission probability.
        :param dim: number of binary HMM.
        :return: list of binary HMM.
        """
        list_models = []
        # self.emission_prob_bin = np.empty((self.x_bin_dim,), dtype=object)
        emission_prob_bin = []
        for i in range(dim):
            emission_prob = self.generate_emission_bin()
            model = hmm.MultinomialHMM(n_components=self.n_components)
            model.startprob_ = self.startprob
            model.transmat_ = self.transmat
            model.emissionprob_ = emission_prob

            emission_prob_bin.append(emission_prob)
            list_models.append(model)

        # Assign to parameters
        theta = [{"prob": prob} for prob in emission_prob_bin]
        self.theta_list = self.theta_list + theta
        return list_models

    def generate_emission_bin(self):
        p = np.concatenate([np.ones((self.n_components, 1)),
                            np.zeros((self.n_components, 1))], axis=1)
        emission_prob = self.crazyshuffle(p)
        # p = np.random.randint(0, 10, size=(3, 1)) / 10
        # q = 1 - p
        # mat_prob = np.concatenate([p, q], axis=1)
        # emission_prob = self.crazyshuffle(mat_prob)
        #
        # # Make sure each state is assigned a more probable category
        # while len(np.unique(np.argmax(emission_prob, axis=1))) != 2:
        #     emission_prob = self.crazyshuffle(emission_prob)
        return emission_prob

    def init_cat_hmm(self, dim=1):
        """
        Create a list of categorical HMM depending on the dimension. All share the start and transition probabilities,
        but each cateogorical HMM has a different emission probability.
        :param dim: number of categorical HMM.
        :return: list of categorical HMM.
        """
        list_models = []
        # self.emission_prob_cat = np.empty((self.x_cat_dim,), dtype=object)
        emission_prob_cat = []
        for i in range(dim):
            emission_prob = self.generate_emission_cat(self.x_cat_class[i])

            model = hmm.MultinomialHMM(n_components=self.n_components)
            model.startprob_ = self.startprob
            model.transmat_ = self.transmat
            model.emissionprob_ = emission_prob

            emission_prob_cat.append(emission_prob)
            list_models.append(model)

        # Assign to parameters
        theta = [{"prob": prob} for prob in emission_prob_cat]
        self.theta_list = self.theta_list + theta

        return list_models

    def crazyshuffle(self, arr):
        x, y = arr.shape
        rows = np.indices((x, y))[0]
        cols = [np.random.permutation(y) for _ in range(x)]
        return arr[rows, cols]

    def generate_emission_cat(self, cat_dim):
        p = np.random.randint(8, 11, (self.n_components, 1)) / 10
        q = 1-p
        mat = np.concatenate([p, np.repeat(np.round((q / (cat_dim-1)), 2), cat_dim-1, 1)], axis=1)
        emission_prob = mat

        # Make sure each state is assigned a more probable category
        while len(np.unique(np.argmax(emission_prob, axis=1))) != cat_dim:
            emission_prob = self.crazyshuffle(mat)
        return emission_prob

    def generate_samples(self):
        """
        Sample from a HMM to generate data.
        :return:
        """
        # Real
        X_gauss, Z = self.gaussian_hmm.sample(self.T)

        # Sample positive hmm from gaussian states
        X_log_gauss = []
        self.positive_hmm.sample()  # init sampler
        for t in range(self.T):
            x_t = self.positive_hmm._generate_sample_from_state(Z[t])
            x_log_t = np.log(1 + np.exp(x_t))
            # x_log_t = soft_plus(self.positive_hmm._generate_sample_from_state(Z[t]))
            X_log_gauss.append(x_log_t)
        X_log_gauss = np.asarray(X_log_gauss)

        # Sample binary hmm from gaussian states
        X_bin = np.zeros((self.T, 0))
        for binary_hmm in self.binary_hmm_list:
            X_bin_i = []
            for t in range(self.T):
                x_bin_t = binary_hmm._generate_sample_from_state(Z[t])
                X_bin_i.append(x_bin_t)
            X_bin_i = np.asarray(X_bin_i)
            X_bin = np.concatenate((X_bin, X_bin_i), axis=1)

        # Sample categorical hmm from gaussian states
        X_cat = np.zeros((self.T, 0))
        for categorical_hmm in self.categorical_hmm_list:
            X_cat_i = []
            for t in range(self.T):
                x_cat_t = categorical_hmm._generate_sample_from_state(Z[t])
                X_cat_i.append(x_cat_t)
            X_cat_i = np.asarray(X_cat_i)
            X_cat = np.concatenate((X_cat, X_cat_i), axis=1)

        # Concat
        X = np.concatenate((X_gauss, X_log_gauss, X_bin, X_cat), axis=1)
        return X, Z

    def create_param_dict(self):
        self.theta = {"var"+str(i): theta for i, theta in enumerate(self.theta_list)}

    def miss_mask(self, percent):
        """
        Create a missing mask with a certain percentage of missing data per dimension.
        :param percent: missing data percentage
        :return: missing_mask
        """
        num_miss = round(percent * self.T)
        mask = np.zeros((self.T, self.X_dim), dtype=bool)

        for d in range(self.X_dim):
            miss_idx = random.sample(range(self.T), num_miss)
            mask[miss_idx, d] = True
        return mask

    def burst_missmask(self, percent):
        """
        Generate bursts of missing data in the generated samples.
        Args:
            percent:

        Returns:

        """
        # Create burst of missing
        num_miss = np.round(T * percent).astype(int)
        burst_mask = np.zeros((self.T, self.X_dim), dtype=bool)

        for d in range(self.X_dim):

            index_list = list(range(self.T))
            burst = np.zeros(0,)
            while index_list:
                # Random length burst
                t_art = int(np.random.uniform(3, 10))

                # Generate proper index for burst
                miss_index = -1
                while (miss_index + t_art / 2 > T) or (miss_index - t_art / 2 < 0):
                    miss_index = random.sample(index_list, 1)[0]

                # Generte burst
                new_burst = np.arange(miss_index - t_art / 2, miss_index + t_art / 2).astype(int)

                # Concatenate and check there is no overlapping
                burst = np.unique(np.concatenate([burst, new_burst])).astype(int)

                # Delete indexes from burst
                index_list = list(set(index_list) - set(burst))

                if len(burst)>=num_miss:
                    burst = burst[:num_miss]
                    break

            # if len(np.where(burst >= T)[0]) or len(np.where(burst < 0)[0]):
            #     burst = np.delete(burst, burst > T)
            #     burst = np.delete(burst, burst < 0)
            #     sample_idx = list(set(list(range(self.T))) - set(burst))
            #
            #     miss_index = random.sample(sample_idx, 1)[0]
            #     new_len = num_miss - len(burst)
            #     append_burst = np.arange(miss_index - new_len / 2, miss_index + new_len / 2).astype(int)
            #     burst = np.concatenate([append_burst, burst])
            #
            # # Element bigger than T bound are replace for inbound values
            # if len(np.where(burst >= T)[0]):
            #     # Delete out-of-bound index
            #     out_bound_miss = len(np.where(burst >= T)[0])
            #     burst = np.delete(burst, np.where(burst >= T))
            #
            #     # Append new burst
            #     append_burst = np.arange(burst[0] - out_bound_miss, burst[0])
            #     burst = np.concatenate([append_burst, burst])
            #
            # # Element smaller than T bound are replace for inbound values
            # elif len(np.where(burst < 0)[0]):
            #     # Delete out-of-bound index
            #     out_bound_miss = len(np.where(burst < 0)[0])
            #     burst = np.delete(burst, np.where(burst < 0))
            #
            #     # Append new burst
            #     append_burst = np.arange(burst[-1] + 1, burst[-1] + out_bound_miss + 1)
            #     burst = np.concatenate([burst, append_burst])

            burst_mask[burst, d] = True

        return burst_mask

    def generate_toydataset(self, percent_miss=[30, 50, 70], num_masks=10):
        """
        Generate a toy dataset from a HMM. Each sequence of T instants and dimensions x_real_dim is saved into a mat file.
        Each mat file consists on the X, Z, the parameteres of the HMM and the different misssing masks.
        :param percent_miss: list with the considered missing data percentages.
        :return:
        """

        # Data
        X_list= []
        Z_list = []
        mask_dict = {}
        for percent in percent_miss:
            for i in range(num_masks):  # 10 different masks.
                mask_dict["mask_" + str(int(percent * 100)) + "_" + str(i+1)] = []

        # ==== Generate Samples ==== #
        for n in tqdm(range(self.N)):
            X, Z = self.generate_samples()
            X_list.append(X)
            Z_list.append(Z)

            # # RANDOM MISSING MASK
            # for percent in percent_miss:
            #     mask = self.miss_mask(percent)
            #     for i in range(num_masks):
            #         key = "mask_" + str(int(percent * 100)) + "_" + str(i+1)
            #         mask_perm, mask = permute_mask(mask)
            #         mask_dict[key].append(mask_perm)

            # BURST MISSING MASK
            for percent in percent_miss:
                for i in range(num_masks):
                    key = "mask_" + str(int(percent * 100)) + "_" + str(i + 1)
                    mask = self.burst_missmask(percent)
                    mask_dict[key].append(mask)

            # Plot one sample data
            if n == 1:
                f, axs = self.save_sample_data(X, Z)
                sample_path = os.path.join(self.save_dir, 'sample.pdf')
                f.savefig(sample_path)

        # ===== STACK DATA ===== #
        X = np.stack(X_list)
        Z = np.stack(Z_list)
        for key in mask_dict.keys():
            mask_dict[key] = np.stack(mask_dict[key])

        # ===== TRAIN/VAL/TEST INDEX ===== #
        train_percent = 0.8
        train_i = int(train_percent * self.N)
        val_percent = 0.1
        val_i = int(val_percent * self.N + train_i)
        test_percent = 0.1
        test_i = int(test_percent * self.N  + val_i)

        index = np.arange(0, self.N)
        np.random.shuffle(index)

        train_index = index[:train_i]
        val_index = index[train_i:val_i]
        test_index = index[val_i:test_i]


        # ===== SAVE TO PICKLE ===== #
        data_dict = {"X": X,
                     "Z": Z,
                     "trans_mat": self.transmat,
                     "start_prob": self.startprob,
                     "train_index": train_index,
                     "val_index": val_index,
                     "test_index": test_index,
                     }
        theta_dict = {"theta": self.theta}

        # Join dicts
        data_dict = {**data_dict, **theta_dict, **mask_dict}
        pickle_path = os.path.join(self.save_dir, self.toy_name + '.pickle')
        with open(pickle_path, 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


        # ===== CSV DATA TYPES ===== #
        types_data = self.x_real_dim * ['real'] + self.x_pos_dim * ['pos'] + \
                     self.x_bin_dim * ['bin'] + self.x_cat_dim * ['cat']

        num_classes = self.x_real_dim * [1] + self.x_pos_dim * [1] \
                      + self.x_bin_dim * [2] + self.x_cat_class

        num_dim = self.x_real_dim * [1] + self.x_pos_dim * [1] \
                  + self.x_bin_dim * [1] + self.x_cat_class

        rows = [["var" + str(i), type, dim, nclass, i] for i, (type, nclass, dim) in
                enumerate(zip(types_data, num_classes, num_dim))]
        rows.insert(0, ["name", "type", "dim", "nclass", "index"])
        csv_path = os.path.join(self.save_dir, 'data_types.csv')
        write_csv_types(rows, csv_path)

        print("Saved at: {}".format(self.save_dir))

    def save_sample_data(self, X, Z):
        """
        Save a sample of the generated dataset to visualize the parameters ans states of the HMM.
        Args:
            X:
            Z:

        Returns:

        """
        types_data = self.x_real_dim * ['real'] + self.x_pos_dim * ['pos'] + \
                     self.x_bin_dim * ['bin'] + self.x_cat_dim * ['cat']

        num_dim = X.shape[-1]
        f, axs = plt.subplots(num_dim + 1, 2, figsize=(8, 10))
        real_var = 0
        pos_var = 0
        bin_var = 0
        cat_var = 0
        for d, var in enumerate(self.theta.keys()):

            # Plot Z
            axs[d, 0].plot(X[:, d])
            axs[d, 0].title.set_text(types_data[d])

            # Real Dist
            if types_data[d] == 'real':
                means, covars = self.theta[var].values()

                for state in range(self.n_components):
                    mean_d = means[state]
                    covar_d = covars[state, real_var, real_var]
                    x = np.random.randn(1000)*covar_d + mean_d
                    sns.kdeplot(x, shade=True, ax=axs[d, 1], label='state:'+str(state))
                axs[d, 1].legend(loc='upper right')
                real_var += 1
                continue

            # Pos Dist
            if types_data[d] == 'pos':
                means, covars = self.theta[var].values()

                for state in range(self.n_components):
                    mean_d = means[state]
                    covar_d = covars[state, pos_var, pos_var]
                    x = np.random.randn(1000) * covar_d + mean_d
                    sns.kdeplot(x, shade=True, ax=axs[d, 1], label='state:'+str(state))
                axs[d, 1].legend(loc='upper right')
                pos_var += 1
                continue

            # Bin probs
            if types_data[d] == 'bin':
                # axins = inset_axes(axs[d, 1], width="100%", height="100%",
                #                    bbox_to_anchor=(1.05, .6, .5, .4),
                #                    bbox_transform=axs[d, 1].transAxes, loc=2, borderpad=0)
                # axins = inset_axes(axs[d], width="30%", height="40%")
                prob = self.theta[var]["prob"]
                sns.heatmap(prob, annot=True, ax=axs[d, 1], cbar=False)
                axs[d, 1].set_xlabel('Categories')
                axs[d, 1].set_ylabel('States')
                bin_var += 1
                continue

            # Cat probs
            if types_data[d] == 'cat':
                # axins = inset_axes(axs[d, 1], width="100%", height="100%",
                #                    bbox_to_anchor=(1.05, .6, .5, .4),
                #                    bbox_transform=axs[d, 1].transAxes, loc=2, borderpad=0)
                # axins = inset_axes(axs[d, 1], width="30%", height="40%")
                prob = self.theta[var]["prob"]
                sns.heatmap(prob, annot=True, ax=axs[d, 1], cbar=False)
                axs[d, 1].set_xlabel('Categories')
                axs[d, 1].set_ylabel('States')
                cat_var += 1
                continue

            axs[d, 1].axis('off')

        axs[num_dim, 0].plot(Z)
        axs[num_dim, 0].title.set_text("States")

        sns.heatmap(self.transmat, annot=True, ax=axs[num_dim, 1], cbar=False)
        axs[num_dim, 1].set_xlabel('Categories')
        axs[num_dim, 1].set_ylabel('States')
        return f, axs


if __name__ == '__main__':
    N = 1  # num. samples
    T = 100  # time series length
    n_components = 3  # number of components
    # Always need real because states Z are generated from that HMM.
    x_real_dim = 1  # dimension for real data
    x_pos_dim = 1  # dimension for positive data
    x_bin_dim = 1  # dimension for binary data
    x_cat_dim = 1  # dimension for binary data
    x_cat_class = [3]  # num. classes for each cat data
    missing = True
    percent_miss = [0, 0.1, 0.2, 0.3, 0.4, 0.5]  # % missing
    num_masks = 10
    ToyDataset = ToyDatasetGenerator(x_real_dim=x_real_dim, x_pos_dim=x_pos_dim, x_bin_dim=x_bin_dim,
                                     x_cat_dim=x_cat_dim, x_cat_class=x_cat_class, n_components=n_components, T=T, N=N)
    ToyDataset.create_param_dict()
    ToyDataset.generate_toydataset(percent_miss, num_masks=10)
