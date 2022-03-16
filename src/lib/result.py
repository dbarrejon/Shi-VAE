# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021. by Daniel Barrejon, UC3M.                                         +
#  All rights reserved. This file is part of the Shi-VAE, and is released under the      +
#  "MIT License Agreement". Please see the LICENSE file that should have been included   +
#   as part of this package.                                                             +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
# Variable is like Placeholder, it indicates where to stop backpropagation
from torch.autograd import Variable
from tqdm import tqdm

from lib import plot, loss
from lib import utils
from lib.aux import cuda, set_device

filter_physionet = ["DiasABP", "NIDiasABP", "HR", "Temp", "Albumin", "NISysABP", "Urine"]


class Result:
    """
    General result class.
    """
    def __init__(self, test_loader, scaler, model, save_dir, args):
        r"""

        Args:
            test_loader (Dataloader): Dataloader for the test set.
            scaler (Object): Scaler object for the input-output normalization strategy.
            model (Object): Shi-VAE model.
            save_dir (string): Path to save files.
            args (args): Arguments.
        """
        self.device = set_device()
        self.model = model
        self.dataset = args.dataset

        if self.device == 'cuda':
            self.model.to(self.device)

        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Result
        self.result = ShiVAEResult(test_loader, scaler, model, args)

    def reconstruction(self, types_list=None):
        r"""
        Reconstruction function.
        Args:
            types_list (list of dictionaries): Each dictionary contains: name, type, dim, nclass, index; for every
            attribute.
        """
        self.result.reconstruction(self.save_dir, types_list=types_list)

    def generation(self, batch_size, types_list=None):
        r"""
        Generation function.
        Args:
            batch_size (int): Batch size.
            types_list (list of dictionaries): Each dictionary contains: name, type, dim, nclass, index; for every
            attribute.
        """
        self.result.generation(batch_size, self.save_dir, types_list=types_list)

    def avg_error(self, model_name="ShiVAE"):
        r"""
        Function to calcualte average error and correlation.
        Args:
            model_name (string): Name of the model.
        """
        self.result.avg_error(model_name=model_name)

    def plot_losses(self):
        r"""
        Plot losses.
        """
        self.result.plot_losses()


class BaseResult:
    """
    Base class for implementing specific Result for custom model.
    """
    def __init__(self, test_loader, scaler, model, args):
        """
        Args:
            test_loader (Dataloader): Dataloader for the test set.
            scaler (Object): Scaler object for the input-output normalization strategy.
            model (Object): Shi-VAE model.
            args (args): Arguments.
        """
        self.model = model
        self.model_nick = args.model

        self.dataset = args.dataset

        self.test_loader = test_loader
        self.test_iter = iter(self.test_loader)

        self.scaler = scaler

        # Dirs
        self.ckpt_file = args.ckpt_file
        self.model_name = args.result_dir.split("/")[-1]
        self.train_csv = os.path.join(args.result_dir, self.model_name) + '_train.csv'
        self.val_csv = os.path.join(args.result_dir, self.model_name) + '_val.csv'
        self.result_dir = args.result_dir

        # CSV
        self.train_df = pd.read_csv(self.train_csv, sep=';')
        self.val_df = pd.read_csv(self.val_csv, sep=';')

    def load_batch(self, test_iter=None):
        r"""
        Load a bath of data from the test data loader.
        Data is transformed into tensor, send to either CPU or CUDA and transformed.
        Args:
            test_iter (optional):  Iterative Dataloader.

        Returns:
            x_batch, mask and batch_attributes.
        """
        # padded data
        if test_iter is None:
            test_iter = self.test_iter
        x_batch, mask, batch_attributes = next(test_iter)

        x_batch, mask = self.preprocess_batch(x_batch, mask)
        return x_batch, mask, batch_attributes

    def preprocess_batch(self, data, mask):
        r"""
        Preprocess batch.
        Args:
            data (Tensor): Shape (BxTxD)
            mask (Tensor): Shape (BxTxD)

        Returns:
            data (TxBxD) and mask(TxBxD)

        """
        # 1. Batch normalize
        if self.scaler is not None:
            data, mask = self.scaler.transform(data, mask)  # (TxBxD))

        # 2. Transpose: TxBxD
        data = data.transpose(1, 0, 2)
        mask = mask.transpose(1, 0, 2)

        # 3. To tensor
        data = torch.tensor(data)
        mask = torch.tensor(mask, dtype=torch.bool)

        # 4. Data rewquires grad
        data = Variable(data).float()  # require grad

        # 5. to CPU or GPU.
        data = cuda(data)
        mask = cuda(mask)
        return data, mask

    def avg_error(self, model_name="ShiVAE"):
        """
        Calculates the average error and the cross correlation on the whole test set.
        Args:
            model_name (string): Model name.
        """
        self.test_loader.dataset.set_eval(True)
        test_iter = iter(self.test_loader)
        data_hat = []
        mask_list = []
        data_full_list = []
        mask_artificial_list = []
        for _ in tqdm(range(len(test_iter))):
            data, mask, data_full, mask_artificial, batch_attributes = next(test_iter)
            # Fetch Data
            data, mask = self.preprocess_batch(data, mask)

            # Zero filling
            data_zf = utils.zero_fill(data, mask)

            # Reconstruction
            dec_pack, latents = self.model.reconstruction(data_zf)

            likes, x_hat = dec_pack
            data_hat.append(x_hat)
            mask_list.append(mask)
            data_full_list.append(data_full)
            mask_artificial_list.append(mask_artificial)

        data_hat = np.hstack(data_hat)
        mask = np.hstack(mask_list)
        data_full = np.vstack(data_full_list)
        mask_artificial = np.vstack(mask_artificial_list)

        data_hat_denorm, mask = self.scaler.inverse_transform(data_hat, mask=mask, transpose=True)

        real_mask = mask_artificial ^ mask
        data_full_zf = np.where(real_mask, np.zeros_like(data_full), data_full)

        # Create dir to patient
        patients_dir = os.path.join(self.result_dir, 'patients')
        if not os.path.isdir(patients_dir):
            os.mkdir(patients_dir)

        _, error_miss, _, corr_miss = loss.compute_avg_error(data_full_zf, data_hat_denorm, mask_artificial,
                                                             self.model.types_list, img_path=patients_dir)
        plot.plot_avg_metric(error_miss, patients_dir, self.model.types_list, type="error")
        plot.plot_avg_metric(corr_miss, patients_dir, self.model.types_list, type="correlation")

        # Save to csv
        self.save_metrics_csv(error_miss, corr_miss, model_name)

        self.test_loader.dataset.set_eval(False)

    def save_metrics_csv(self, error_miss, corr_miss, model_name="ShiVAE"):
        r"""
        Save the metrics to a csv file.
        Args:
            error_miss (dict): Dictionary with the error on the missing.
            corr_miss (dict): Dictionary with the cross correlation on the missing.
            model_name (stirng): Name of the model.

        Returns:

        """
        var = list(error_miss.keys())
        err = list(error_miss.values())
        corr = list(corr_miss.values())
        model = len(error_miss) * [model_name]

        data_df = pd.DataFrame.from_dict({"Model": model,
                                          "var": var,
                                          "err": err,
                                          "corr": corr})

        # Load metrics if already exists and concat
        metrics_path = os.path.join(self.result_dir, "metrics.csv")
        # if os.path.isfile(metrics_path):
        #     data_old_df = pd.read_csv(metrics_path)
        #     data_df = pd.concat([data_old_df, data_df]).reset_index()

        data_df.to_csv(metrics_path, index=False)

    def plot_losses(self):
        r"""
        Plot  train and validation losses.
        """
        f, _ = plot.plot_elbo(self.train_df, self.cols2plot)
        train_str = 'train_loss_' + self.model_name + '.pdf'
        f.savefig(os.path.join(self.result_dir, train_str))
        plt.close()

        f, _ = plot.plot_elbo(self.val_df, self.cols2plot)
        val_str = 'val_loss_' + self.model_name + '.pdf'
        f.savefig(os.path.join(self.result_dir, val_str))
        plt.close()


class ShiVAEResult(BaseResult):
    """
    Result class for the Shi-VAE.
    """
    def __init__(self, test_loader, scaler, model, args):
        super().__init__(test_loader, scaler, model, args)
        self.cols2plot = ['n_elbo', 'nll_loss', 'nll_real', 'nll_pos', 'nll_bin', 'nll_cat', 'kld_q_z', 'kld_q_s']

    def reconstruction(self, save_dir, types_list=None):
        assert types_list is not None

        x_batch, mask, batch_attributes = self.load_batch()
        x_batch_fill = utils.zero_fill(x_batch, mask)
        x_data, latent = self.model.reconstruction(x_batch_fill)

        # Fetch data
        # x: params, x_dec
        likes = x_data[0]
        x_dec = x_data[1]
        x_real = x_batch
        z_data, s_data = latent
        # z: sample, mean, std
        z = z_data[0]
        z_mean = z_data[1]
        z_std = z_data[2]
        # s: sample
        s = s_data[0]

        # Denormalize, to numpy and transpose
        params = self.scaler.param_inverse_transform(likes, transpose=True)
        x_real, mask = self.scaler.inverse_transform(x_real, mask, transpose=True)
        x_dec, _ = self.scaler.inverse_transform(x_dec, transpose=True)

        # Transpose
        z_mean = z_mean.transpose(0, 1)
        z_std = z_std.transpose(0, 1)
        z = z.transpose(0, 1)
        s = s.transpose(0, 1)

        # To numpy
        z_mean = z_mean.cpu().data.numpy()
        z_std = z_std.cpu().data.numpy()
        z = z.cpu().data.numpy()
        s = s.cpu().data.numpy()

        # Create dir to store all reconstructions
        compare_dir = os.path.join(save_dir, 'reconstruction')
        if not os.path.isdir(compare_dir):
            os.mkdir(compare_dir)

        # Order patients by number of missing data
        num_missing_tensor = torch.tensor([m.sum() for m in mask])
        ordered_signals_list, signals_idx = num_missing_tensor.sort(0, descending=False)
        signals_idx = signals_idx.tolist()

        print("Compare real and reconstruction.")
        for i, signal in tqdm(enumerate(signals_idx)):
            # Filter by sequence length
            ith_seq_length = batch_attributes['sequence_lengths'][signal]
            # x
            x_real_seq = x_real[signal, :ith_seq_length, :]
            x_dec_seq = x_dec[signal, :ith_seq_length, :]
            mask_seq = mask[signal, :ith_seq_length, :]

            # Real vs Reconstruction
            if 'hmm' in self.dataset:
                f, _ = plot.real_vs_recon_heter_missing(x_real_seq, x_dec_seq, types_list=types_list, mask=mask_seq)
            elif "physionet" in self.dataset:
                temp_types_list = utils.filter_csv_types(types_list, filter_physionet)
                temp_idx = [int(type_d["index"]) for type_d in temp_types_list]
                f, _ = plot.real_vs_recon_heter_missing(x_real_seq[:, temp_idx], x_dec_seq[:, temp_idx],
                                                        mask_seq[:, temp_idx], temp_types_list)
            else:  # HMM Dataset
                f, _ = plot.real_vs_recon_heter_missing(x_real_seq, x_dec_seq, types_list=types_list, mask=mask_seq)

            f.savefig(os.path.join(compare_dir, str(signal) + '_real_vs_dec.pdf'))
            plt.close(f)

            # Real vs Params
            if 'hmm' in self.dataset:
                f, _ = plot.real_vs_params_heter_missing(x_real_seq, params, id=signal, mask=mask_seq,
                                                         types_list=types_list)
            elif "physionet" in self.dataset:
                temp_types_list = utils.filter_csv_types(types_list, filter_physionet)
                temp_idx = [int(type_d["index"]) for type_d in temp_types_list]
                f, _ = plot.real_vs_param_physionet(x_real_seq[:, temp_idx], params, mask_seq[:, temp_idx], signal,
                                                    ith_seq_length, temp_types_list)
            else:
                f, _ = plot.real_vs_params_heter_missing(x_real_seq, params, id=signal, mask=mask_seq,
                                                         types_list=types_list)

            f.savefig(os.path.join(compare_dir, str(signal) + '_real_vs_params.pdf'))
            plt.close(f)

            # z
            z_seq = z[signal, :ith_seq_length, :]
            z_mean_seq = z_mean[signal, :ith_seq_length, :]
            z_std_seq = z_std[signal, :ith_seq_length, :]

            f, _ = plot.plot_ts(z_seq, label="z")
            f.savefig(os.path.join(compare_dir, str(signal) + '_z.pdf'))
            plt.close(f)

            f, _ = plot.plot_ts(z_mean_seq, label="z $\mu$")
            f.savefig(os.path.join(compare_dir, str(signal) + '_z_mean.pdf'))
            plt.close(f)

            f, _ = plot.plot_ts(z_std_seq, label="z $\sigma$")
            f.savefig(os.path.join(compare_dir, str(signal) + '_z_std.pdf'))
            plt.close(f)

            # s
            s_seq = s[signal, :ith_seq_length, :]
            f, _ = plot.plot_discrete_latent(s_seq)
            f.savefig(os.path.join(compare_dir, str(signal) + '_s.pdf'))
            plt.close(f)

            f, _ = plot.s_vs_z(z_mean_seq, s_seq)
            f.savefig(os.path.join(compare_dir, str(signal) + '_s_vs_z.pdf'))
            plt.close(f)

    def generation(self, batch_size, save_dir, types_list=None):
        assert types_list is not None
        # Generate random lengths from categorical distribution
        lengths = np.array([40, 100, 150])
        prob_lenghts = [1 / len(lengths)] * len(lengths)
        categ_output = np.random.multinomial(1, prob_lenghts, size=batch_size)
        lengths_list = [lengths[x == 1][0] for x in categ_output]

        # Create dir to store all reconstructions
        gener_dir = os.path.join(save_dir, 'generation')
        if not os.path.isdir(gener_dir):
            os.mkdir(gener_dir)

        print("Generation.")
        for i, ts in tqdm(enumerate(lengths_list)):
            x_data, z_data, s_data = self.model.sample(1, ts)

            # Fetch data
            # x: params, x_dec
            params = x_data[0]
            x_dec = x_data[1]
            # z: sample, mean, std
            z = z_data[0]
            z_mean = z_data[1]
            z_std = z_data[2]
            # s: sample
            s = s_data[0]

            # Denormalize, to numpy and transpose
            params = self.scaler.param_inverse_transform(params, transpose=True)
            x_dec, _ = self.scaler.inverse_transform(x_dec, transpose=True)

            # Transpose
            z_mean = z_mean.transpose(0, 1)
            z_std = z_std.transpose(0, 1)
            z = z.transpose(0, 1)
            s_probs = s.transpose(0, 1)
            s = s.transpose(0, 1)

            # To numpy
            z_mean = z_mean.cpu().data.numpy()
            z_std = z_std.cpu().data.numpy()
            z = z.cpu().data.numpy()
            s = s.cpu().data.numpy()

            # Filter by sequence id
            # x
            x_dec = x_dec[0]

            # Generate plots
            f, _ = plot.gener_recon_heter_missing(x_dec, types_list)
            f.savefig(os.path.join(gener_dir, str(i) + '_gener_dec_x.pdf'))
            plt.close(f)

            f, _ = plot.gener_params_heter_missing(params, 0, types_list)
            f.savefig(os.path.join(gener_dir, str(i) + '_gener_params_x.pdf'))
            plt.close(f)

            # z
            z = z[0]
            z_mean = z_mean[0]
            z_std = z_std[0]

            f, _ = plot.plot_ts(z, label="z")
            f.savefig(os.path.join(gener_dir, str(i) + '_gener_z.pdf'))
            plt.close(f)

            f, _ = plot.plot_ts(z_mean, label="z $\mu$")
            f.savefig(os.path.join(gener_dir, str(i) + '_gener_z_mean.pdf'))
            plt.close(f)

            f, _ = plot.plot_ts(z_std, label="z $\sigma$")
            f.savefig(os.path.join(gener_dir, str(i) + '_gener_z_std.pdf'))
            plt.close(f)

            # s
            s = s[0]
            f, _ = plot.plot_discrete_latent(s)
            f.savefig(os.path.join(gener_dir, str(i) + '_s.pdf'))
            plt.close(f)

            # z vs. s
            f, _ = plot.s_vs_z(z, s)
            f.savefig(os.path.join(gener_dir, str(i) + '_gener_z_vs_s.pdf'))
            plt.close(f)
