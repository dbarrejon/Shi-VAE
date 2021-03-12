# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021. by Daniel Barrejon, UC3M.                                         +
#  All rights reserved. This file is part of the Shi-VAE, and is released under the      +
#  "MIT License Agreement". Please see the LICENSE file that should have been included   +
#   as part of this package.                                                             +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import os


def cuda(xs):
    r"""
    Send variable :attr:xs to cuda.
    Args:
        xs (Tensor or List of Tensors): Input data

    Returns:
        xs sent to cuda.
    """
    device = set_device()
    if not isinstance(xs, (list, tuple)):
        return xs.to(torch.device(device))
    else:
        return [x.to(torch.device(device)) for x in xs]


def set_gpu(gpu_number):
    r"""
    Chooses the GPU for training. Depends on the machine.
    Args:
        gpu_number (int): Number of GPU.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_number
    

def set_device():
    """
    Chooses CUDA(GPU) if available; otherwise, CPU.
    Returns:

    """
    # os.environ['CUDA_VISIBLE_DEVICES'] == -1 > CPU
    if torch.cuda.is_available() and os.environ['CUDA_VISIBLE_DEVICES'] != -1:
        return "cuda"
    else:
        return "cpu"
