# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021. by Daniel Barrejon, UC3M.                                         +
#  All rights reserved. This file is part of the Shi-VAE, and is released under the      +
#  "MIT License Agreement". Please see the LICENSE file that should have been included   +
#   as part of this package.                                                             +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class EarlyStopping(object):
    """
    EarlyStopping scheme to finish training if model is not improving any more.
    """
    def __init__(self, patience=15, min_delta=0.1):
        """

        Args:
            patience: (int) Patience value to stop training.
            min_delta: (float) Minimum margin for improving.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.patience_cnt = 0
        self.prev_loss_val = 2000000
        self.patient_cum = 0

    def stop(self, loss_val):
        """

        Args:
            loss_val: (int) Validation loss to compare and apply early stopping.

        Returns:
            boolean: True if stop, False if continue.
        """
        if (abs(self.prev_loss_val - loss_val)> self.min_delta):
            self.patience_cnt = 0
            self.prev_loss_val = loss_val

        else:
            self.patience_cnt += 1
            self.patient_cum += 1
            print('Patience count: ', self.patience_cnt)

        if (self.patience_cnt > self.patience or self.patient_cum > 80):
            return True
        else:
            return False

