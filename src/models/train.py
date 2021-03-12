# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021. by Daniel Barrejon, UC3M.                                         +
#  All rights reserved. This file is part of the Shi-VAE, and is released under the      +
#  "MIT License Agreement". Please see the LICENSE file that should have been included   +
#   as part of this package.                                                             +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
# Variable is like Placeholder, it indicates where to stop backpropagation
from torch.autograd import Variable

from lib import plot
from lib.aux import cuda, set_device
from lib.early_stopping import EarlyStopping

matplotlib.use("Pdf")


def update_loss_dict(old_loss_dict, new_loss_dict):
    r"""
    Update with new losses.
    Args:
        old_loss_dict (dictionary): Dictionary with old losses.
        new_loss_dict (dictionary): Dictionary with new losses.

    Returns:
        updated dict
    """
    for loss in old_loss_dict.keys():
        if torch.is_tensor(new_loss_dict[loss]):
            new_val = new_loss_dict[loss].detach().cpu().numpy()
        else:
            new_val = new_loss_dict[loss]
        old_loss_dict[loss].append(new_val)
    return old_loss_dict


class KL_Annealing:
    r"""
    Class for warming up the KL term with an annealing factor :math:`\beta`.
    """
    def __init__(self, annealing_epochs):
        self.annealing_epochs = annealing_epochs

    def update_beta(self, epoch):
        r"""
        Update de annealing factor beta.
        Args:
            epoch (int): Current epoch.

        Returns:
            beta: Weight for the KL term. Between 0-1.

        """
        if self.annealing_epochs > 0:
            if epoch < self.annealing_epochs:
                beta = (epoch * 1. / self.annealing_epochs) ** 2
            else:
                beta = 1.

        else:
            beta = 1.
        return np.clip(beta, 0.0, 1.0)


class BaseTrainer:
    """
    Base Trainer class.
    """
    def __init__(self, model, optimizer, args, scaler=None):
        """
        Init.
        Args:
            model (object): Model object.
            optimizer (object): Optimizer.
            args (args): Args.
            scaler (object): Heterogeneous Scaler.
        """
        self.loss_list = None
        self.device = set_device()
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.lr = args.l_rate
        self.init_epoch = 1
        self.n_epochs = args.n_epochs

        self.args = args

        # KL Annealing
        self.kl_annealing_epochs = args.kl_annealing_epochs  # epochs to apply annealing
        self.annealing_weight = KL_Annealing(self.kl_annealing_epochs)

        # Safety options
        self.clip = args.clip  # clip_grad_norm value
        self.early_stopping = EarlyStopping(patience=50, min_delta=0.2)  # applied on validation

        # Saving options
        self.save_every = args.save_every
        self.print_every = args.print_every
        self.plot_every = args.plot_every
        self.save_model = args.save_model
        self.result_dir = args.result_dir

        # Dirs

        self.model_name = args.result_dir.split("/")[-1]
        self.train_csv = os.path.join(args.result_dir, self.model_name) + '_train.csv'
        self.val_csv = os.path.join(args.result_dir, self.model_name) + '_val.csv'

        if self.device == 'cuda':
            self.model.cuda()

        self.cumulative_grad = []
        self.layers = []

    def accumulate_gradients(self, model):
        r"""
        Accumulate gradients.
        Args:
            model (object): Model.
        """
        cumulative_grads_epoch = []
        for n, p in model.named_parameters():
            if p.requires_grad and "bias" not in n:
                if p.grad is None:
                    print(n)
                    exit()
                cumulative_grads_epoch.append(p.grad.abs().sum().cpu().data.numpy())

        # first gradients
        if not self.cumulative_grad:
            self.cumulative_grad = cumulative_grads_epoch
        else:
            self.cumulative_grad = list(np.array(self.cumulative_grad) + np.array(cumulative_grads_epoch))

    def init_loss_dict(self):

        r"""
        Create a dictionary with the different losses for training and validation.
        Returns:
            train and validation loss dicts.
        """

        # init with empty lists
        train_loss_dict = {k: [] for k in self.loss_list}
        val_loss_dict = {k: [] for k in self.loss_list}
        return train_loss_dict, val_loss_dict

    def plot_train_loss(self, cols2plot):
        r"""
        Plot training losses.
        Args:
            cols2plot (list): List with the columns to plot.
        """
        f, _ = plot.plot_elbo(self.train_df, cols2plot)
        train_str = 'train_loss_' + self.model_name + '.pdf'
        f.savefig(os.path.join(self.result_dir, train_str))
        plt.close()

    def plot_val_loss(self, cols2plot):
        r"""
        Plot val losses.
        Args:
            cols2plot (list): List with the columns to plot.
        """
        f, _ = plot.plot_elbo(self.val_df, cols2plot)
        val_str = 'val_loss_' + self.model_name + '.pdf'
        f.savefig(os.path.join(self.result_dir, val_str))
        plt.close()

    def show_grad_flow(self, model):
        r"""
        Show the gradient updates.
        Args:
            model (object): Model.
        """
        f, _ = plot.plot_grad_flow(model.named_parameters())
        grad_str = 'gradient_' + self.model_name + '.pdf'
        f.savefig(os.path.join(self.result_dir, grad_str))
        plt.close()

    def show_accum_grad_flow(self):
        r"""
        Show the accumulated gradient.
        """
        for n, p in self.model.named_parameters():
            if p.requires_grad and ("bias" not in n):
                self.layers.append(n)

        f, _ = plot.plot_accum_grad(self.layers, self.cumulative_grad)
        grad_str = 'accum_gradient_' + self.model_name + '.pdf'
        f.savefig(os.path.join(self.result_dir, grad_str))
        plt.close()
        self.layers = []

    def show_grad_img(self, model):
        r"""
        Show the gradient.
        """
        grad_dir = os.path.join(self.result_dir, 'grad')
        if not os.path.exists(grad_dir):
            os.makedirs(grad_dir)
        grad_str = os.path.join(grad_dir, 'grad_')
        plot.plot_grad_img(model.named_parameters(), grad_str)

    def show_LSTM_weights(self, rnn_module):
        r"""
        Show the LSTM weights.
        Args:
            rnn_module (module): LSTM module.
        """

        lstm_dir = os.path.join(self.result_dir, 'lstm')
        if not os.path.exists(lstm_dir):
            os.makedirs(lstm_dir)
        lstm_str = os.path.join(lstm_dir, '')
        plot.plot_LSTM_weights(rnn_module, lstm_str)

    def load_checkpoint(self, model_dict):
        r"""
        Load model weights, optimizer parameters and last epoch from checkpoint.
        Args:
            model_dict (dict): Dictionary with the saved values in the checkpoint.
        """

        self.model.load_state_dict(model_dict['state_dict'])
        self.optimizer.load_state_dict(model_dict['optimizer'])
        self.init_epoch = model_dict['epoch'] + 1

    def get_df(self, cols, restore):
        r"""
        Create or restore the training and validation csv for losses.
        Args:
            cols (list): List with column names.
            restore (boolean): If true, restore DataFrame.

        Returns:
            train and validation DataFrames.
        """
        # Train
        if os.path.exists(self.train_csv):
            if restore:
                train_df = pd.read_csv(self.train_csv, sep=';')
                # Reset init epoch
                self.init_epoch = int(train_df.iloc[-1]['epoch'])
            else:  # train from scratch
                train_df = pd.DataFrame(columns=cols)
                train_df.to_csv(self.train_csv, sep=';', index=False)
        else:
            train_df = pd.DataFrame(columns=cols)
            train_df.to_csv(self.train_csv, sep=';', index=False)

        # Val
        if os.path.exists(self.val_csv):
            if restore:
                val_df = pd.read_csv(self.val_csv, sep=';')
                # Reset init epoch
                self.init_epoch = int(val_df.iloc[-1]['epoch'])
            else:  # train from scratch
                val_df = pd.DataFrame(columns=cols)
                val_df.to_csv(self.val_csv, sep=';', index=False)
        else:
            val_df = pd.DataFrame(columns=cols)
            val_df.to_csv(self.val_csv, sep=';', index=False)

        return train_df, val_df

    def save_csv_train(self, df, loss_dict, epoch):
        r"""
        Save training csv.
        Args:
            df (DataFrame): Data to save.
            loss_dict (dict.): Dictionary with losses.
            epoch (int): Epoch.

        Returns:
            Updated Dataframe.
        """
        loss_dict['epoch'] = epoch
        df = df.append(loss_dict, ignore_index=True)
        # df.loc[len(df)] = cols
        df.drop_duplicates(df.columns[0], keep='last', inplace=True)
        df.to_csv(self.train_csv, sep=';', index=False)
        return df

    def save_csv_val(self, df, loss_dict, epoch):
        r"""
        Save validation csv.
        Args:
            df (DataFrame): Data to save.
            loss_dict (dict.): Dictionary with losses.
            epoch (int): Epoch.

        Returns:
            Updated Dataframe.
        """

        loss_dict['epoch'] = epoch
        df = df.append(loss_dict, ignore_index=True)
        df.drop_duplicates(df.columns[0], keep='last', inplace=True)
        df.to_csv(self.val_csv, sep=';', index=False)
        return df

    def save(self, dict_params, epoch):
        """
        Save checkpoint.
        Args:
            dict_params (dict.): Dictionary with the parameters to save.
            epoch (int): Curren epoch.
        """

        ckpt_file = os.path.join(self.result_dir, self.model_name) + "_" + str(epoch) + ".pth"
        torch.save(dict_params, ckpt_file)
        print('Saved model to ' + ckpt_file)

    def save_best(self, dict_params):
        """
        Save checkpoint for best perfomring model.
        Args:
            dict_params (dict.): Dictionary with the parameters to save.
        """
        ckpt_file = os.path.join(self.result_dir, self.model_name) + "_best.pth"
        torch.save(dict_params, ckpt_file)
        print('Saved model to ' + ckpt_file)

    def preprocess_batch(self, data, mask):
        r"""
        Normalize the data, transpose, to Tensor, set to gradient and send to GPU or CPU accordingly.
        Args:
            data (Tensor): Shape (BxTxD)
            mask (Tensor): Shape (BxTxD)

        Returns:
            data (Tensor): Shape (TxBxD)
            mask (Tensor): Shape (TxBxD)
        """
        # 1. Batch normalize
        if self.scaler is not None:
            data, mask = self.scaler.transform(data, mask)  # (TxBxD))

        # 2. Transpose: BxTxD -> TxBxD
        data = data.transpose(1, 0, 2)
        mask = mask.transpose(1, 0, 2)

        # 3. To tensor
        data = torch.tensor(data)
        mask = torch.tensor(mask, dtype=torch.bool)

        # 4. Data requires grad
        data = Variable(data).float()  # require grad

        # 5. to CPU or GPU.
        data = cuda(data)
        mask = cuda(mask)
        return data, mask

    def train_epoch(self, train_loader, epoch):
        r"""
        Base training epoch.
        Args:
            train_loader (DataLoader): Training data loader.
            epoch (int): Current epoch.
        """
        pass

    def valid_epoch(self, valid_loader, epoch):
        r"""
        Base training epoch.
        Args:
           valid_loader (DataLoader): Training data loader.
           epoch (int): Current epoch.
        """
        pass
