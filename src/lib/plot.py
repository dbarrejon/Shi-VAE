# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021. by Daniel Barrejon, UC3M.                                         +
#  All rights reserved. This file is part of the Shi-VAE, and is released under the      +
#  "MIT License Agreement". Please see the LICENSE file that should have been included   +
#   as part of this package.                                                             +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import string

import matplotlib
matplotlib.use("Pdf")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

from lib import utils, loss


# # === Latex Options === #
# rcParams['font.size'] = 20
rc('font', family='serif')
rc('text', usetex=True)

# === Matplotlib Options === #
cm = plt.cm.get_cmap('tab10')
real_kwargs = {"linestyle": "-",
               "color": cm.colors[0],
               "alpha": 0.6,
               "linewidth": 4
               }
rec_kwargs = {"linestyle": "--",
              "color": cm.colors[1],
              "alpha": 0.6,
              "linewidth": 4
              }


def format2latex(str):
    """
    Format a string to appropriate latex. It replaces "_" by " " and capitalizes every word in the str.
    Args:
        string:
    Returns:
        Modified string.
    """
    return string.capwords(str.replace("_", " "))


def plot_discrete_latent(s):
    """
    Plots variable s in a heatmap.
    Args:
        s:

    Returns:

    """
    T = s.shape[0]
    t = np.arange(0, T)
    xticks = np.arange(0, T, step=10)
    with sns.axes_style("ticks"):
        f, ax = plt.subplots(1, 1)
        sns.heatmap(s.transpose(), cmap='Greys', cbar=True, ax=ax, xticklabels=xticks)
        ax.invert_yaxis()
        for _, spine in ax.spines.items():
            spine.set_visible(True)
        ax.set_xlim(min(t), max(t))
        ax.set_xticks(xticks)
        ax.title.set_text(r"s")
    f.set_tight_layout(True)
    return f, ax


def s_vs_z(z, s):
    """

    Args:
        z (Tesnor):
        s:

    Returns:

    """
    T = z.shape[0]
    t = np.arange(0, T)
    xticks = np.arange(0, T, step=10)
    with sns.axes_style("ticks"):
        f, axs = plt.subplots(1, 1)
        sns.heatmap(s.transpose(), cmap='Greys', cbar=False, ax=axs, xticklabels=xticks)
        axs.invert_yaxis()
        ax2 = axs.twinx()
        ax2.plot(z)
        ax2.set_xlim(min(t), max(t))
        ax2.set_xticks(xticks)
        ax2.title.set_text(r"s vs z")
    f.set_tight_layout(True)
    return f, axs


def real_vs_recon_heter_missing(x_real, x_dec, mask=None, types_list=None):
    r"""
    Plots real vs reconstruction.
    Args:
        x_real (Tensor): Real data, with shape (TxD).
        x_dec (Tensor): Reconstructed data, with shape (TxD)
        mask (Tensor, optional): Missing mask, with shape (TxD).
        types_list (list of dictionaries): Each dictionary contains: name, type, dim, nclass, index; for every
            attribute.

    Returns:
        fig, axs of the plot.

    """
    assert types_list is not None
    assert mask is not None
    n_rows = len(types_list)
    n_cols = 1

    T = x_real.shape[0]
    t = np.arange(0, T)
    with sns.axes_style("ticks"):
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(8, 10))
        for i, type_dict in enumerate(types_list):
            var = type_dict['name']
            type = type_dict['type']
            label = format2latex(var)

            marker_real_miss = np.where(mask[:, i])[0]
            ax_plot = axs[i] if n_rows != 1 else axs

            # Original
            ax_plot.plot(t, x_real[:, i], label='Real', **real_kwargs)
            # Recon
            ax_plot.plot(t, x_dec[:, i], label="Rec", markevery=marker_real_miss.tolist(), **rec_kwargs)
            ax_plot.set_xlim(min(t), max(t))
            ax_plot.title.set_text(label + ' ' + type)
            if i == len(types_list) - 1:
                ax_plot.legend(loc='upper center', frameon=False, bbox_to_anchor=(0.5, -0.1), ncol=2)

        fig.set_tight_layout(True)
    return fig, axs


def real_vs_params_heter_missing(x_real, params, id=0, mask=None, types_list=None):
    r"""
        Plots real vs reconstruction.
        Args:
            x_real (Tensor): Real data, with shape (TxD).
            params (dict of params): Dictionary with the parameters of the different distributions.
            id (int): Sequence id.
            mask (Tensor, optional): Missing mask, with shape (TxD).
            types_list (list of dictionaries): Each dictionary contains: name, type, dim, nclass, index; for every
                attribute.

        Returns:
            fig, axs of the plot.

        """
    assert types_list is not None
    assert mask is not None

    # Create figure with rows = # variables
    n_cols = 1
    n_rows = len(types_list)
    T = x_real.shape[0]
    t = np.arange(0, T)
    xticks = np.arange(0, T, step=10)
    with sns.axes_style("ticks"):

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(8, 10))
        for i, type_dict in enumerate(types_list):
            var = type_dict['name']
            type = type_dict['type']
            label = format2latex(var)

            marker_real_miss = np.where(mask[:, i])[0]
            ax_plot = axs[i] if n_rows != 1 else axs

            # Recon
            if type in ['real', 'pos']:
                # Original
                ax_plot.plot(t, x_real[:, i], label='Real', **real_kwargs)

                # mean
                x_mean = params[var][0][id].flatten()
                x_std = params[var][1][id].flatten()
                ax_plot.plot(t, x_mean, label="Rec", markevery=marker_real_miss.tolist(), **rec_kwargs)
                upper_mean = x_mean + 2 * x_std
                lower_mean = x_mean - 2 * x_std
                ax_plot.fill_between(t, upper_mean, lower_mean, alpha=0.2, color=cm.colors[1])

            elif type == 'bin':
                # Original
                ax_plot.plot(t, x_real[:, i], label='Real', **real_kwargs)

                # Recon
                p = params[var][id].flatten()
                ax_plot.plot(t, p, label="Rec", markevery=marker_real_miss.tolist(), **rec_kwargs)

            elif type == 'cat':
                # Recon
                p = params[var][id, :, :]
                sns.heatmap(p.transpose(), cmap='Greys', cbar=False, ax=ax_plot, label="Recon", xticklabels=xticks)
                ax_plot.invert_yaxis()

                ax_plot = ax_plot.twinx()
                # Original
                ax_plot.plot(t, x_real[:, i], label="Real", **real_kwargs)
                # Recon
                cat_real = np.argmax(p, axis=1)
                ax_plot.plot(t, cat_real, label="Recon", markevery=marker_real_miss.tolist(), **rec_kwargs)

            ax_plot.set_xlim(min(t), max(t))
            ax_plot.set_xticks(xticks)
            ax_plot.title.set_text(label + ' ' + type)

            if i == len(types_list) - 1:
                ax_plot.legend(loc='upper center', frameon=False, bbox_to_anchor=(0.5, -0.1), ncol=2)

        fig.set_tight_layout(True)

    return fig, axs


# === Gener === #
def gener_recon_heter_missing(x_dec, types_list=None):
    r"""

    Args:
        x_dec (Tensor): Generated data, with shape (TxD).
        types_list (list of dictionaries): Each dictionary contains: name, type, dim, nclass, index; for every
            attribute.

    Returns:
        fig, axs of the plot.

    """
    assert types_list is not None
    n_rows = len(types_list)
    n_cols = 1

    T = x_dec.shape[0]
    t = np.arange(0, T)
    with sns.axes_style("ticks"):
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(8, 10))
        for i, type_dict in enumerate(types_list):
            var = type_dict['name']
            type = type_dict['type']
            label = format2latex(var)

            ax_plot = axs[i] if n_rows != 1 else axs
            # Recon
            ax_plot.plot(t, x_dec[:, i], label="Sample", **real_kwargs)
            ax_plot.set_xlim(min(t), max(t))
            ax_plot.title.set_text(label + ' ' + type)
            if i == len(types_list) - 1:
                ax_plot.legend(loc='upper center', frameon=False, bbox_to_anchor=(0.5, -0.1), ncol=2)

        fig.set_tight_layout(True)
    return fig, axs


def gener_params_heter_missing(params, id=0, types_list=None):
    r"""

    Args:
        params (dict of params): Dictionary with the parameters of the different distributions.
        id (int): Sequence id.
        types_list (list of dictionaries): Each dictionary contains: name, type, dim, nclass, index; for every
            attribute.

    Returns:
        fig, axs of the plot.

    """
    assert types_list is not None

    # Create figure with rows = # variables
    n_cols = 1
    n_rows = len(types_list)

    with sns.axes_style("ticks"):

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(8, 10))
        for i, type_dict in enumerate(types_list):
            var = type_dict['name']
            type = type_dict['type']
            label = format2latex(var)

            ax_plot = axs[i] if n_rows != 1 else axs

            # Recon
            if type in ['real', 'pos']:
                # mean
                x_mean = params[var][0][id].flatten()
                x_std = params[var][1][id].flatten()
                T = x_mean.shape[0]
                t = np.arange(0, T)
                xticks = np.arange(0, T, step=10)

                ax_plot.plot(t, x_mean, label="Sample", **real_kwargs)
                upper_mean = x_mean + 2 * x_std
                lower_mean = x_mean - 2 * x_std
                ax_plot.fill_between(t, upper_mean, lower_mean, alpha=0.2, color=cm.colors[1])

            elif type == 'bin':
                # Recon
                p = params[var][id].flatten()
                T = p.shape[0]
                t = np.arange(0, T)
                xticks = np.arange(0, T, step=10)
                ax_plot.plot(p, label="Sample", **real_kwargs)

            elif type == 'cat':
                # Recon
                p = params[var][id, :, :]
                T = p.shape[0]
                t = np.arange(0, T)
                xticks = np.arange(0, T, step=10)

                sns.heatmap(p.transpose(), cmap='Greys', cbar=False, ax=ax_plot, label="Sample", xticklabels=xticks)
                ax_plot.invert_yaxis()
                ax_plot = ax_plot.twinx()
                # Recon
                cat_real = np.argmax(p, axis=1)
                ax_plot.plot(t, cat_real, label="Sample", **real_kwargs)

            ax_plot.set_xlim(min(t), max(t))
            ax_plot.set_xticks(xticks)
            ax_plot.title.set_text(label + ' ' + type)

            if i == len(types_list) - 1:
                ax_plot.legend(loc='upper center', frameon=False, bbox_to_anchor=(0.5, -0.1), ncol=2)

        fig.set_tight_layout(True)
    return fig, axs


# === Real vs Recon Physionet === #
# def real_vs_recon_physionet(x_real, x_dec, mask=None, types_list=None):
#     r"""
#         Plots real vs reconstruction.
#         Args:
#             x_real (Tensor): Real data, with shape (TxD).
#             x_dec (Tensor): Reconstructed data, with shape (TxD)
#             mask (Tensor, optional): Missing mask, with shape (TxD).
#             types_list (list of dictionaries): Each dictionary contains: name, type, dim, nclass, index; for every
#                 attribute.
#
#         Returns:
#             fig, axs of the plot.
#
#     """
#     assert types_list is not None
#     assert mask is not None
#
#     # Number of plots
#     n_rows = len(types_list)
#     n_cols = 1
#
#     T = x_real.shape[0]
#     t = np.arange(0, T)
#     with sns.axes_style("ticks"):
#         fig, axs = plt.subplots(n_rows, n_cols, figsize=(8, 10))
#         for i, type_dict in enumerate(types_list):
#             var = type_dict['name']
#             type = type_dict['type']
#             label = format2latex(var)
#
#             marker_real_miss = np.where(mask[:, i])[0]
#             ax_plot = axs[i] if n_rows != 1 else axs
#
#             # Original
#             ax_plot.plot(t, x_real[:, i], label='Real', **real_kwargs)
#             # Recon
#             ax_plot.plot(t, x_dec[:, i], label="Rec", markevery=marker_real_miss.tolist(), **rec_kwargs)
#             ax_plot.set_xlim(min(t), max(t))
#             ax_plot.title.set_text(label + ' ' + type)
#             if i == len(types_list) - 1:
#                 ax_plot.legend(loc='upper center', frameon=False, bbox_to_anchor=(0.5, -0.1), ncol=2)
#         fig.set_tight_layout(True)
#     return fig, axs


def real_vs_param_physionet(x_real, params, mask=None, id=0, seq_length=None, types_list=None):
    r"""
        Plots real vs reconstruction.
        Args:
            x_real (Tensor): Real data, with shape (TxD).
            params (dict of params): Dictionary with the parameters of the different distributions.
            id (int): Sequence id.
            seq_length (int): Length of the sequence.
            mask (Tensor, optional): Missing mask, with shape (TxD).
            types_list (list of dictionaries): Each dictionary contains: name, type, dim, nclass, index; for every
                attribute.

        Returns:
            fig, axs of the plot.

        """
    assert types_list is not None
    assert mask is not None
    assert seq_length is not None

    n_cols = 1
    n_rows = len(types_list)
    T = x_real.shape[0]
    t = np.arange(0, T)
    xticks = np.arange(0, T, step=10)
    with sns.axes_style("ticks"):

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(8, 10))
        for i, type_dict in enumerate(types_list):
            var = type_dict['name']
            type = type_dict['type']
            label = format2latex(var)

            marker_real_miss = np.where(mask[:, i])[0]
            ax_plot = axs[i] if n_rows != 1 else axs

            # Recon
            if type in ['real', 'pos']:
                # Original
                ax_plot.plot(t, x_real[:, i], label='Real', **real_kwargs)
                # mean
                x_mean = params[var][0][id, :seq_length].flatten()
                x_std = params[var][1][id, :seq_length].flatten()
                ax_plot.plot(t, x_mean, label="Rec", markevery=marker_real_miss.tolist(), **rec_kwargs)

                upper_mean = x_mean + 2 * x_std
                lower_mean = x_mean - 2 * x_std
                ax_plot.fill_between(t, upper_mean, lower_mean, alpha=0.2, color=cm.colors[1])

            elif type == 'bin':
                # Original
                ax_plot.plot(t, x_real[:, i], label='Real', **real_kwargs)
                # Recon
                p = params[var][id, :seq_length].flatten()
                ax_plot.plot(t, p, label="Rec", markevery=marker_real_miss.tolist(), **rec_kwargs)

            elif type == 'cat':
                # Recon
                p = params[var][id, :seq_length]
                sns.heatmap(p.transpose(), cmap='Greys', cbar=False, ax=ax_plot, label="Recon", xticklabels=xticks)
                ax_plot.invert_yaxis()

                ax_plot = ax_plot.twinx()
                # Original
                ax_plot.plot(t, x_real[:, i], label="Real", **real_kwargs)
                cat_real = np.argmax(p, axis=1)
                ax_plot.plot(t, cat_real, label="Recon", markevery=marker_real_miss.tolist(), **rec_kwargs)

            ax_plot.set_xlim(min(t), max(t))
            ax_plot.set_xticks(xticks)
            ax_plot.title.set_text(label + ' ' + type)

            if i == len(types_list) - 1:
                ax_plot.legend(loc='upper center', frameon=False, bbox_to_anchor=(0.5, -0.1), ncol=2)

        fig.set_tight_layout(True)
    return fig, axs


# x_mean, x_std, z_mean, z_std
def plot_ts(ts, label="z"):
    r"""
    Plot a time series tensor.
    Args:
        ts (Tensor): Time series to plot, of shape TxD.
        label (string): Label for the time series.

    Returns:
        fig, axs for the plot.
    """
    T = ts.shape[0]
    t = np.arange(0, T)
    with sns.axes_style("ticks"):
        f, ax = plt.subplots(1, 1)
        ax.plot(t, ts)
        ax.set_xlim(min(t), max(t))
        ax.set_xticks(np.arange(0, T, step=10))
        ax.title.set_text(r"" + label)
    f.set_tight_layout(True)
    return f, ax


def plot_elbo(loss_df, cols2plot=None, log=False):
    """

    Args:
        loss_df (DataFrame): Dataframe containing the losses along epochs.
        cols2plot (list): List containing the columns to plot.
        log (boolean): If true, apply log scale.

    Returns:
        fig, axs for the plot.
    """
    if cols2plot is None:
        cols2plot = list(loss_df)
        cols2plot.remove('epoch')

    losses_label = utils.losses2latexformat(cols2plot)

    loss_np = loss_df[cols2plot].to_numpy()
    epochs = loss_np.shape[0]
    epoch_range = np.linspace(1, epochs, epochs)
    # create a figure
    f, ax = plt.subplots(1, 1)
    for i in range(loss_np.shape[1]):
        if log ==  True:
            loss_np = np.log(loss_np)
            plt.yscale("log")

        ax.plot(epoch_range, loss_np[:, i], label=r""+losses_label[i])

    plt.xlim([1, epochs])
    plt.legend(loc='upper right')
    plt.xlabel(r'Epochs')
    plt.ylabel(r'ELBO')
    f.set_tight_layout(True)
    return f, ax


def plot_LSTM_weights(LSTM, save_path='.'):
    r"""
    Plot the LSTM weights for debugging.
    Args:
        LSTM (LSTM Module): LSTM layer, containing the weights.
        save_path (string): Path to save the
    """
    with sns.axes_style("whitegrid", {'axes.grid': False}):
        for layer in range(LSTM.num_layers):
            weights_layer = LSTM.all_weights[layer]

            weights_ih = weights_layer[0]
            W_ii = weights_ih[:10, :]
            W_if = weights_ih[10:20, :]
            W_ig = weights_ih[20:30, :]
            W_io = weights_ih[30:40, :]

            weights_hh = weights_layer[1]
            W_hi = weights_hh[:10, :]
            W_hf = weights_hh[10:20, :]
            W_hg = weights_hh[20:30, :]
            W_ho = weights_hh[30:40, :]

            bias_ih = weights_layer[2]
            b_ii = bias_ih[:10]
            b_if = bias_ih[10:20]
            b_ig = bias_ih[20:30]
            b_io = bias_ih[30:40]

            bias_hh = weights_layer[3]
            b_hi = bias_hh[:10]
            b_hf = bias_hh[10:20]
            b_hg = bias_hh[20:30]
            b_ho = bias_hh[30:40]

        # Plot Matrices ih
        fig, axs = plt.subplots(2, 2, figsize=(6, 5))
        im = axs[0, 0].imshow(W_ii.abs().cpu().data.numpy())
        fig.colorbar(im, ax=axs[0, 0], use_gridspec=True)
        axs[0, 0].title.set_text('$W_{ii}$')

        im = axs[0, 1].imshow(W_if.abs().cpu().data.numpy())
        fig.colorbar(im, ax=axs[0, 1], use_gridspec=True)
        axs[0, 1].title.set_text('$W_{if}$')

        im = axs[1, 0].imshow(W_ig.abs().cpu().data.numpy())
        fig.colorbar(im, ax=axs[1, 0], use_gridspec=True)
        axs[1, 0].title.set_text('$W_{ig}$')

        im = axs[1, 1].imshow(W_io.abs().cpu().data.numpy())
        fig.colorbar(im, ax=axs[1, 1], use_gridspec=True)
        axs[1, 1].title.set_text('$W_{io}$')

        plt.savefig(os.path.join(save_path, 'W_ih.pdf'))
        plt.close()

        # Plot Matrices hh
        fig, axs = plt.subplots(2, 2, figsize=(6, 5))

        im = axs[0, 0].imshow(W_hi.abs().cpu().data.numpy())
        fig.colorbar(im, ax=axs[0, 0], use_gridspec=True)
        axs[0, 0].title.set_text('$W_{hi}$')

        im = axs[0, 1].imshow(W_hf.abs().cpu().data.numpy())
        fig.colorbar(im, ax=axs[0, 1], use_gridspec=True)
        axs[0, 1].title.set_text('$W_{hf}$')

        im = axs[1, 0].imshow(W_hg.abs().cpu().data.numpy())
        fig.colorbar(im, ax=axs[1, 0], use_gridspec=True)
        axs[1, 0].title.set_text('$W_{hg}$')

        im = axs[1, 1].imshow(W_ho.abs().cpu().data.numpy())
        fig.colorbar(im, ax=axs[1, 1], use_gridspec=True)
        axs[1, 1].title.set_text('$W_{ho}$')

        plt.savefig(os.path.join(save_path, 'W_hh.pdf'))
        plt.close()

        # Plot Biases
        fig, axs = plt.subplots(1, 2)
        axs[0].plot(b_ii.cpu().data.numpy(), label='$b_{ii}$')
        axs[0].plot(b_if.cpu().data.numpy(), label='$b_{if}$')
        axs[0].plot(b_ig.cpu().data.numpy(), label='$b_{ig}$')
        axs[0].plot(b_io.cpu().data.numpy(), label='$b_{io}$')
        axs[0].legend(loc='lower right')
        axs[0].title.set_text('$bias_{hi}$')

        axs[1].plot(b_hi.cpu().data.numpy(), label='$b_{hi}$')
        axs[1].plot(b_hf.cpu().data.numpy(), label='$b_{hf}$')
        axs[1].plot(b_hg.cpu().data.numpy(), label='$b_{hg}$')
        axs[1].plot(b_ho.cpu().data.numpy(), label='$b_{ho}$')
        axs[1].title.set_text('$bias_{hh}$')

        axs[1].legend(loc='lower right')
        plt.savefig(os.path.join(save_path, 'biases.pdf'))
        plt.close()


def plot_grad_flow(named_parameters):
    r"""
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage:
        Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow

    Args:
        named_parameters (model parameters):

    Returns:
        fig, axs for the plot.
    """
    ave_grads = []
    max_grads = []
    layers = []

    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu().data.numpy())
            max_grads.append(p.grad.abs().max().cpu().data.numpy())

    index = [format2latex(layer) for layer in layers]
    df = pd.DataFrame({'Max. Grads': max_grads, 'Ave. Grad. ': ave_grads}, index=index)
    df = df.astype(float)  # avoid TypeError pandas.

    fig, ax = plt.subplots(1, 1)
    sep = 10
    y_ticks = np.arange(0, sep * len(ave_grads), sep)
    df.plot.barh(stacked=True, width=1, yticks=y_ticks, fontsize=8, position=0.5, ax=ax)
    ax.invert_yaxis()
    plt.title("Gradient flow")
    fig.set_tight_layout(True)
    return fig, ax


def plot_accum_grad(layers, accum_grads):
    r"""
    Plot the the accumulated gradient through time.
    Args:
        layers (list of string): List with the name of the layers.
        accum_grads (Tensor):

    Returns:
        fig, axs for the plot.
    """
    index = [format2latex(layer) for layer in layers]
    df = pd.DataFrame({'Accum. Grads': accum_grads}, index=index)
    df = df.astype(float)  # avoid TypeError pandas.
    fig, ax = plt.subplots(1, 1)
    sep = 10
    y_ticks = np.arange(0, sep * len(accum_grads), sep)
    df.plot.barh(stacked=True, width=1, yticks=y_ticks, fontsize=8, position=0.5, ax=ax)
    ax.invert_yaxis()
    plt.title("Cumulative Gradient flow")
    fig.set_tight_layout(True)
    return fig, ax


def plot_grad_img(named_parameters, save_path):
    r"""
    Plot gardient of every layer for debugging.
    Args:
        named_parameters (layers):
        save_path (string): Path to save the plots.

    """
    # format names for better visualization
    with sns.axes_style("whitegrid", {'axes.grid': False}):
        aspect = 20
        pad_fraction = 0.5
        for n, p in named_parameters:
            if (p.requires_grad) and ("bias" not in n):
                fig, axs = plt.subplots(1, 2)

                im = axs[0].imshow(p.abs().cpu().data.numpy())
                weight_label = format2latex('$' + n + '_weight$')
                axs[0].title.set_text(weight_label)
                divider = make_axes_locatable(axs[0])
                width = axes_size.AxesY(axs[0], aspect=1. / aspect)
                pad = axes_size.Fraction(pad_fraction, width)
                cax = divider.append_axes("right", size=width, pad=pad)
                fig.colorbar(im, cax=cax, use_gridspec=True)

                im = axs[1].imshow(p.grad.abs().cpu().data.numpy())
                grad_label = format2latex('$' + n + '_grad$')
                axs[1].title.set_text(grad_label)
                divider = make_axes_locatable(axs[1])
                width = axes_size.AxesY(axs[1], aspect=1. / aspect)
                pad = axes_size.Fraction(pad_fraction, width)
                cax = divider.append_axes("right", size=width, pad=pad)

                fig.colorbar(im, cax=cax, use_gridspec=True)
                fig.set_tight_layout(True)
                plt.savefig(save_path + n + '.pdf')
                plt.close()


def plot_avg_metric(metric_miss, path, types_list, type="error"):
    r"""
    Plot the avg error or the avg normalized cross correlation.
    Args:
        metric_miss (dict): Dictionary containing the error on missing values.
        path (string): Path to save the plot.
        types_list (list of dictionaries): Each dictionary contains: name, type, dim, nclass, index; for every
            attribute.
        type (string): It can be "error" or "correlation.
    """
    metric_df = pd.DataFrame.from_dict(data=metric_miss, orient="index", columns=["metric"]).reset_index()
    types = [type_dict["type"] for type_dict in types_list]
    metric_df["Types"] = types

    # Plot
    f, ax = plt.subplots(1, 1)
    sns.set_color_codes("pastel")
    sns.barplot(data=metric_df, x="index", y="metric", ax=ax, hue="Types", dodge=False, palette="Set2")

    # annotate axis = seaborn axis
    for p in ax.patches:
                 ax.annotate("%.4f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', fontsize=11, color='black', xytext=(0, 4),
                     textcoords='offset points')

    # Title
    metric_type = loss.get_type_error(metric_miss, types_list=types_list)
    if type == "error":
        title_str = " Avg error:  {:.3f} \n ".format(np.sum(list(metric_type.values())))
    elif type == "correlation":
        title_str = " Avg correlation:  {:.3f} \n ".format(np.sum(list(metric_type.values())))

    if "real" in metric_type:
        title_str += "Real: {:.3f}  ".format(metric_type["real"])

    if "pos" in metric_type:
        title_str += "Pos: {:.3f}  ".format(metric_type["pos"])

    if "bin" in metric_type:
        title_str += "Bin: {:.3f}  ".format(metric_type["bin"])

    if "cat" in metric_type:
        title_str += "Cat: {:.3f}  ".format(metric_type["cat"])

    plt.title(title_str)

    plt.xticks(rotation=75)
    ax.set_ylabel('Error')
    ax.set_xlabel('')
    save_path = path + "/error_miss.pdf" if type=="error" else path + "/correlation.pdf"
    f.savefig(save_path, bbox_inches='tight')
    plt.close()
