# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021. by Daniel Barrejon, UC3M.                                         +
#  All rights reserved. This file is part of the Shi-VAE, and is released under the      +
#  "MIT License Agreement". Please see the LICENSE file that should have been included   +
#   as part of this package.                                                             +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import sys

import numpy as np
import torch

from lib import utils, datasets as dset
from lib.aux import set_device
from lib.process_args import get_args, save_args
from lib.scalers import HeterogeneousScaler

# CPU or GPU Run
args = get_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = set_device()
print("DEVICE: {}".format(device))

dataset = "physionet_burst"
args.dataset = dataset
args.train = -1

# Shi-VAE
args.model_name = '{}_{}_{}z_{}h_{}s'.format(args.model, args.dataset, args.z_dim, args.h_dim, args.K)

args.result_dir = os.path.join(args.ckpt_dir, args.experiment, args.model_name)
args.ckpt_file = os.path.join(args.result_dir, args.model_name + ".pth")
args.best_ckpt_file = os.path.join(args.result_dir, args.model_name + "_best.pth")

# Restore training
if (args.restore == 1):
    if (not os.path.isfile(args.ckpt_file)):
        print('Model not found at {}'.format(args.ckpt_file))
        sys.exit()
    model_dict = torch.load(args.ckpt_file)
    n = args.n_epochs
    # Restore args from training args.
    args = model_dict['params']
    args.n_epochs = n
    args.restore = 1

# Print Arguments
print('ARGUMENTS')
for arg in vars(args):
    print('{} = {}'.format(arg, getattr(args, arg)))

# Create checkpoint directory
if (not os.path.exists(args.ckpt_dir)):
    os.makedirs(args.ckpt_dir)
# Create results directory
if (not os.path.exists(args.result_dir)):
    os.makedirs(args.result_dir)

# ============= LOAD DATA ============= #
data = np.load(os.path.join(args.data_dir, dataset, dataset + ".npz"))

types_csv = os.path.join(args.data_dir, dataset, "data_types_real.csv")
types_list = utils.read_csv_types(types_csv)

# Train
x_train = data["x_train_miss"].astype(np.float32)
x_train_full = data["x_train_full"].astype(np.float32)
m_train = data["m_train_miss"].astype(bool)
m_train_artificial = data["m_train_artificial"].astype(bool)
y_train = data["y_train"]

# Val
x_val = data["x_val_miss"].astype(np.float32)
x_val_full = data["x_val_full"].astype(np.float32)
m_val = data["m_val_miss"].astype(bool)
m_val_artificial = data["m_val_artificial"].astype(bool)
y_val = data["y_val"]

# Test
x_test = data["x_test_miss"].astype(np.float32)
x_test_full = data["x_test_full"].astype(np.float32)
m_test = data["m_test_miss"].astype(bool)
m_test_artificial = data["m_test_artificial"].astype(bool)
y_test = data["y_test"]

# ===== Scaler  ===== #
scaler = HeterogeneousScaler(types_list)
scaler.fit(x_train, m_train)

# ===== Datasets  ===== #
data_train = dset.HeterDataset(x_train, m_train, x_train_full, m_train_artificial, types_list=types_list)
data_valid = dset.HeterDataset(x_val, m_val, x_val_full, m_val_artificial, types_list=types_list)
data_test = dset.HeterDataset(x_test, m_test, x_test_full, m_test_artificial, types_list=types_list)

# ===== DataLoaders  ===== #
train_loader = torch.utils.data.DataLoader(data_train, batch_size=64, shuffle=True,
                                           collate_fn=dset.standard_collate)
valid_loader = torch.utils.data.DataLoader(data_valid, batch_size=64, shuffle=False,
                                           collate_fn=dset.standard_collate)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=64, shuffle=False,
                                          collate_fn=dset.standard_collate)
# ============= MODEL ============= #
from models.trainers import Trainer
from models.shivae import ShiVAE
# Shi-VAE
model = ShiVAE(h_dim=args.h_dim, z_dim=args.z_dim, s_dim=args.K, types_list=types_list,
                    n_layers=1,
                    learn_std=args.learn_std)
optimizer = torch.optim.Adam(model.parameters(), lr=args.l_rate)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Trainable params: {}'.format(total_params))

# ============= TRAIN ============= #
if args.train == 1 or args.train == -1:
    trainer = Trainer(model, optimizer, args, scaler=scaler)

    # Train from pretrained model
    if (args.restore == 1 and os.path.isfile(args.ckpt_file)):
        print('Model loaded at {}'.format(args.ckpt_file))
        trainer.load_checkpoint(model_dict)

    print('Training points: {}'.format(len(train_loader.dataset)))
    trainer.train(train_loader, test_loader)

# ============= RESULTS ============= #
if args.train == 0 or args.train == -1:
    from lib.result import Result

    result_dir = os.path.dirname(args.ckpt_file)
    print('Save images in: {}'.format(result_dir))

    # Load pretrained model
    model_dict = torch.load(args.best_ckpt_file)
    model.load_state_dict(model_dict['state_dict'])

    # Create test loader
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=64, shuffle=False,
                                              collate_fn=dset.standard_collate)
    # ================== #
    # Reconstruction and generation
    # ================== #
    result = Result(test_loader, scaler, model, result_dir, args)
    model_name = "ShiVAE"
    result.avg_error(model_name=model_name)
    result.reconstruction(types_list=types_list)
    result.generation(args.result_imgs, types_list=types_list)

# ===== Save args ===== #
args_path = os.path.join(args.result_dir, args.model_name) + args.model_name + '.json'
save_args(args, args_path)
