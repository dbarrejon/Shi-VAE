# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021. by Daniel Barrejon, UC3M.                                         +
#  All rights reserved. This file is part of the Shi-VAE, and is released under the      +
#  "MIT License Agreement". Please see the LICENSE file that should have been included   +
#   as part of this package.                                                             +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import argparse
import json


def get_args():
    """
    This function returns the parsed arguments.
    Returns:
        args: paser object
    """
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--model', type=str, default='shivae',
                        help='Used model.')
    parser.add_argument('--train', type=int, default=-1,
                        help='Training options [0: Results, 1:Train, -1:Train and Result]')
    parser.add_argument('--restore', type=int, default=0,
                        help='Restore training')
    parser.add_argument('--experiment', type=str, default='test',
                        help='Experiment name.')

    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--kl_annealing_epochs', type=int, default=20,
                        help='Number of epochs to apply annealing on KL divergence.')
    parser.add_argument('--l_rate', type=float, default=5e-3,
                        help='Learning rate')
    parser.add_argument('--percent_miss', type=int, default=30,
                        help='Missing rate for HMM dataset, or any synthetic dataset.')

    parser.add_argument('--save_model', type=bool, default=True,
                        help='Save model.')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save model every n epochs')
    parser.add_argument('--print_every', type=int, default=1,
                        help='Print model every n epochs')
    parser.add_argument('--plot_every', type=int, default=20,
                        help='Plot ELBO every n epochs')

    parser.add_argument('--learn_std', type=bool, default=True,
                        help='Learn std for real and positive distributions.')
    parser.add_argument('--z_dim', type=int, default=4,
                        help='Dimension of z')
    parser.add_argument('--h_dim', type=int, default=10,
                        help='Dimension of h')

    # ShiVAE
    parser.add_argument('--K', type=int, default=5,
                        help='Dimension of latent code s for ShiVAE')

    parser.add_argument('--local', type=int, default=1,
                        help='Set to 1 for local execution. 0 for cluster/server execution.')
    parser.add_argument('--gpu', type=str, default='-1',
                        help='Select gpu=-1: CPU. Else: GPU.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of samples per batch-')
    parser.add_argument('--result_imgs', type=int, default=10,
                        help='Number of images to show in reconstruction.')
    parser.add_argument('--clip', type=int, default=0.5,
                        help='Clipping value to avoid exploding gradients.')

    parser.add_argument('--data_dir', type=str, default='../data/',
                        help='Parent directory for data.')
    parser.add_argument('--filter_vars', default=None,
                        help='Variables to use. Use None for using all variabes.', nargs='+')

    parser.add_argument('--dataset', type=str, default='hmm_heter_1000_1real_1pos_1bin_1cat_3_100',
                        help='Name of the dataset for the experiment.')
    parser.add_argument('--ckpt_dir', type=str, default='saves',
                        help='Parent directory to save models')
    parser.add_argument('--ckpt_file', type=str, default=None,
                        help='Pytorch Weights (.pth), with absolute path.')

    args = parser.parse_args()
    return validate_args(args)


def validate_args(args):
    """
    This function validates the arguments.
    Args:
        args: parser object

    Returns:
        args: parser object.
        Validated args.

    """
    assert args.n_epochs > 0, "Number of epochs in non positive"
    assert args.kl_annealing_epochs >= 0
    assert args.z_dim > 0
    assert args.batch_size > 0
    assert args.clip > 0
    assert args.model in ["shivae"]
    return args


def save_args(args, args_path):
    """
    Saves the args object.
    Args:
        args: args object
        args_path: path to save
    """
    with open(args_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def load_args(args_path):
    """
    Loads args from json file.
    Args:
        args_path: path to json file.

    Returns:
        args: args object.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    with open(args_path, 'r') as f:
        args.__dict__ = json.load(f)
    return args
