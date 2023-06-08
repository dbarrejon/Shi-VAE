# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021. by Daniel Barrejon, UC3M.                                         +
#  All rights reserved. This file is part of the Shi-VAE, and is released under the      +
#  "MIT License Agreement". Please see the LICENSE file that should have been included   +
#   as part of this package.                                                             +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import matplotlib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from lib import utils, plot
from lib.loss import compute_avg_error
from models.train import BaseTrainer, update_loss_dict

matplotlib.use("Pdf")
shivae_losses = ['n_elbo', 'nll_loss', 'nll_real', 'nll_pos', 'nll_bin', 'nll_cat', 'kld', 'kld_q_z', 'kld_q_s']


class LinearTempScheduler:
    r"""
    Linear Temperature Scheduler.

    tau = max( temp_end, temp_init - temp_init * slope )
    """

    def __init__(self, init_temp=3, end_temp=0.01, annealing_epochs=10):
        r"""
        Args:
            init_temp (int): Initial Temperature.
            end_temp (int): Last Temperature.
            annealing_epochs (int): Number of annealing epochs.
        """
        self.temp_init = init_temp
        self.temp_end = end_temp
        self.annealing_epochs = annealing_epochs
        self.slope = (init_temp - end_temp) / annealing_epochs

    def update_temp(self, epoch):
        r"""
        Update temperature
        Args:
            epoch (int): Current epoch.

        Returns:
            Update temperature
        """
        tau = max(self.temp_init - self.slope * epoch, self.temp_end)
        print('Updated temperature: {}'.format(tau))
        return tau


class ExpTempScheduler:
    r"""
    Exponential Temperature Scheduler.

    tau = max( temp_end, exp(-alpha * t) )
    """

    def __init__(self, end_temp, annealing_epochs=20):
        r"""
        Args:
            end_temp (int): Last temperature.
            annealing_epochs (int): Number of annealing epochs.
        """
        self.temp_end = end_temp
        self.annealing_epochs = annealing_epochs
        self.alpha = -np.log(end_temp) / annealing_epochs

    def update_temp(self, epoch):
        r"""
        Update temperature
        Args:
            epoch (int): Current epoch.

        Returns:
            Update temperature
        """
        tau = np.exp(-self.alpha * epoch)
        tau = max(tau, self.temp_end)
        print('Updated temperature: {}'.format(tau))
        return tau


class Trainer(BaseTrainer):
    r"""
    Trainer Class.
    """
    def __init__(self, model, optimizer, args, scaler=None):
        r"""
        Args:
            model (object): Model.
            optimizer (object): Optimizer.
            args (args): Args.
            scaler (scaler): Heterogeneous Scaler.
        """
        super(Trainer, self).__init__(model, optimizer, args, scaler=scaler)

        self.model_nick = args.model

        self.loss_list = shivae_losses
        self.print_str = "-ELBO: {n_elbo:.3f} | KLD Loss: {kld:.3f}  NLL Loss: {nll_loss:.3f} |" \
                         "  Real Loss {nll_real:.3f}  Pos Loss {nll_pos:.3f}  Binary Loss {nll_bin:.3f}" \
                         "  Categ Loss {nll_cat:.3f}  |" \
                         "  KL_q_z {kld_q_z:.3f}  KL_q_s {kld_q_s:.3f}"
        self.temp = 3
        temp_anneal = round(self.n_epochs * 0.9)  # TODO: Important!
        self.temp_scheduler = LinearTempScheduler(init_temp=3, end_temp=0.1, annealing_epochs=temp_anneal)

        self.epoch_print_str = "\nEpoch: {}/{}\n" + self.print_str
        self.cols2plot = self.loss_list
        self.csv_cols = ['epoch'] + self.loss_list
        self.train_df, self.val_df = self.get_df(self.csv_cols, args.restore)

    def train(self, train_loader, valid_loader):
        """
        Train function, including the training epoch and the validation epoch.
        Args:
            train_loader (DataLoader): Training data loader.
            valid_loader (DataLoader): Validation data loader.

        Returns:
            train_loss (dict.): Dictionary with training losses.
            val_loss (dict.): Dictionary with validation losses.
        """
        n_elbo_list = []
        epoch = 0
        self.train_loss, self.val_loss = self.init_loss_dict()

        for epoch in range(self.init_epoch, self.n_epochs+1):
            print('Epoch {}/{}'.format(epoch, self.n_epochs))
            # ======= Train ======= #
            train_loss_epoch = self.train_epoch(train_loader, epoch)
            self.train_df = self.save_csv_train(self.train_df, train_loss_epoch, epoch)
            self.train_loss = update_loss_dict(self.train_loss, train_loss_epoch)

            # ======= SAVE BEST MODEL ======= #
            n_elbo_train = train_loss_epoch["n_elbo"]
            if all(n_elbo_train < np.array(n_elbo_list)) and self.save_model:
                self.save_best({'state_dict': self.model.state_dict(),
                                'epoch': epoch,
                                'optimizer': self.optimizer.state_dict(),
                                'params': self.args})
            n_elbo_list.append(n_elbo_train)

            # ======= EVAL ======= #
            self.model.eval()
            stop_training, val_loss_epoch = self.valid_epoch(valid_loader, epoch)
            self.val_df = self.save_csv_val(self.val_df, val_loss_epoch, epoch)
            self.val_loss = update_loss_dict(self.val_loss, val_loss_epoch)

            # Check early stopping
            if stop_training:
                break

            self.model.train()  # set to training
            self.temp = self.temp_scheduler.update_temp(epoch)  # update temperature on Gumbel-Softmax

        # Average over epochs
        self.train_loss = {loss: np.mean(val) for loss, val in self.train_loss.items()}
        self.val_loss = {loss: np.mean(val) for loss, val in self.val_loss.items()}

        # Plot loss evolution
        # train
        self.plot_train_loss(self.cols2plot)

        # val
        self.plot_val_loss(self.cols2plot)

        if self.save_model:
            self.save({'state_dict': self.model.state_dict(),
                       'epoch': epoch,
                       'optimizer': self.optimizer.state_dict(),
                       'params': self.args}, epoch)

        return self.train_loss, self.val_loss

    def train_epoch(self, train_loader, epoch):
        r"""
        Train one epoch.
        Args:
            train_loader (DataLoader): Training data loader.
            epoch (int): Current epoch.

        Returns:
            Dictionary with losses for current epoch.
        """
        train_iter = iter(train_loader)

        epoch_train_loss, _ = self.init_loss_dict()

        beta = self.annealing_weight.update_beta(epoch)

        for i in tqdm(range(len(train_iter))):
            # Fetch data
            data, mask, batch_attributes = next(train_iter)

            # Normalize, apply transformations and zero fill the mask.
            data, mask = self.preprocess_batch(data, mask)
            data = utils.zero_fill(data, mask)

            # Save cumulative gradients for sanity check.
            if i != 0:
                self.accumulate_gradients(self.model)

            self.optimizer.zero_grad()  # Set gradients to zero. Otherwise they accumulate

            # Forward pass
            loss_dict = self.model(data, mask=mask, beta=beta, temp=self.temp)

            n_elbo = loss_dict['n_elbo']
            n_elbo.backward()  # backprop

            if np.isnan(n_elbo.detach().cpu().numpy()):
                print('Encountered NaN, stopping training. Please check the learning_rate settings and the momentum.')
                break

            # grad norm clipping, only in pytorch version >= 1.10
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

            self.optimizer.step()

            epoch_train_loss = update_loss_dict(epoch_train_loss, loss_dict)

        # saving model
        if epoch % self.save_every == 0 and self.save_model:
            self.save({'state_dict': self.model.state_dict(),
                       'epoch': epoch,
                       'optimizer': self.optimizer.state_dict(),
                       'params': self.args}, epoch)

        # Average over batches
        epoch_train_loss = {loss: np.mean(val) for loss, val in epoch_train_loss.items()}

        if epoch % self.print_every == 0:
            epoch_print_str = "Epoch: {}/{}\n" + self.print_str
            print(epoch_print_str.format(epoch, self.n_epochs, **epoch_train_loss))

        if epoch % self.plot_every == 0:
            self.plot_train_loss(self.cols2plot)
            # self.show_grad_flow(model=self.model)
            # self.show_accum_grad_flow()
            # self.show_grad_img(model=self.model)
            # self.show_LSTM_weights(rnn_module=self.model.rnn)

        return epoch_train_loss

    def valid_epoch(self, valid_loader, epoch):
        r"""
        Validate one epoch.
        If :attr:plot_every is active, it also calculates the avg. error and avg. cross correlation.
        Args:
            valid_loader (DataLoader): Validation data loader.
            epoch (int): Current epoch.

        Returns:
            Dictionary with losses for current epoch.
        """
        valid_iter = iter(valid_loader)
        _, epoch_val_loss = self.init_loss_dict()
        beta = self.annealing_weight.update_beta(epoch)

        for _ in tqdm(range(len(valid_iter))):
            data, mask, batch_attributes = next(valid_iter)
            data, mask = self.preprocess_batch(data, mask)
            data = utils.zero_fill(data, mask)

            loss_dict = self.model(data, mask=mask, beta=beta, temp=self.temp)
            epoch_val_loss = update_loss_dict(epoch_val_loss, loss_dict)

        # Average over batches
        epoch_val_loss = {loss: np.mean(val) for loss, val in epoch_val_loss.items()}

        # Print Loss
        if epoch % self.print_every == 0:
            epoch_print_str = "Epoch (valid): {}/{}\n" + self.print_str
            print(epoch_print_str.format(epoch, self.n_epochs, **epoch_val_loss))

        # Plot loss across epochs
        if epoch % self.plot_every == 0:
            self.plot_val_loss(self.cols2plot)

            # =================== #
            # Avg Error and AUROC
            # =================== #

            # Avg Error
            valid_loader.dataset.set_eval(True)
            valid_iter = iter(valid_loader)
            data_hat = []
            mask_list = []
            data_full_list = []
            mask_artificial_list = []
            for i in tqdm(range(len(valid_iter))):
                data, mask, data_full, mask_artificial, batch_attributes = next(valid_iter)
                # Fetch Data
                data, mask = self.preprocess_batch(data, mask)

                # Zero filling
                data_zf = utils.zero_fill(data, mask)
                # Reconstruction
                dec_pack, latents = self.model.reconstruction(data_zf, temp=self.temp)

                likes, x_hat = dec_pack
                data_hat.append(x_hat)
                mask_list.append(mask)
                data_full_list.append(data_full)
                mask_artificial_list.append(mask_artificial)

            # Debug
            # if False:
            #     lib.save_theta_decoder(likes, types_list=self.model.types_list, path=self.result_dir)

            data_hat = np.hstack(data_hat)
            mask = np.hstack(mask_list)
            data_full = np.vstack(data_full_list)
            mask_artificial = np.vstack(mask_artificial_list)

            data_hat_denorm, mask = self.scaler.inverse_transform(data_hat, mask=mask, transpose=True)

            real_mask = mask_artificial ^ mask
            data_full_zf = np.where(real_mask, np.zeros_like(data_full), data_full)

            _, error_miss, _, corr_miss = compute_avg_error(data_full_zf, data_hat_denorm, mask_artificial,
                                                            self.model.types_list, img_path=self.result_dir)

            plot.plot_avg_metric(error_miss, self.result_dir, self.model.types_list, type="error")
            plot.plot_avg_metric(corr_miss, self.result_dir, self.model.types_list, type="correlation")

            valid_loader.dataset.set_eval(False)

        n_elbo = epoch_val_loss['n_elbo']

        # Early Stopping
        return self.early_stopping.stop(n_elbo), epoch_val_loss
