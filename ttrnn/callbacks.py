import io
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from scipy.linalg import LinAlgWarning
from sklearn.decomposition import PCA
import gym
from .trainer import SupervisedRNN

plt.switch_backend("Agg")

def envisinstance(env, type):
    if isinstance(env, type):
        return True
    elif hasattr(env, 'env'):
        return envisinstance(env.env, type)
    else:
        return False

### Helper functions written by Andrew Sedler

def has_image_loggers(loggers):
    """Checks whether any image loggers are available.

    Parameters
    ----------
    loggers : obj or list[obj]
        An object or list of loggers to search.
    """
    logger_list = loggers if isinstance(loggers, list) else [loggers]
    for logger in logger_list:
        if isinstance(logger, pl.loggers.TensorBoardLogger):
            return True
        elif isinstance(logger, pl.loggers.WandbLogger):
            return True
    return False


def log_figure(loggers, name, fig, step):
    """Logs a figure image to all available image loggers.

    Parameters
    ----------
    loggers : obj or list[obj]
        An object or list of loggers
    name : str
        The name to use for the logged figure
    fig : matplotlib.figure.Figure
        The figure to log
    step : int
        The step to associate with the logged figure
    """
    # Save figure image to in-memory buffer
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png")
    image = Image.open(img_buf)
    # Distribute image to all image loggers
    logger_list = loggers if isinstance(loggers, list) else [loggers]
    for logger in logger_list:
        if isinstance(logger, pl.loggers.TensorBoardLogger):
            logger.experiment.add_figure(name, fig, step)
        elif isinstance(logger, pl.loggers.WandbLogger):
            logger.log_image(name, [image], step)
    img_buf.close()


class TrajectoryPlot(pl.Callback):
    """Plots the top-3 PC's of the latent trajectory for
    all samples in the validation set and logs to tensorboard.
    """

    def __init__(self, split='val', log_every_n_epochs=100):
        """Initializes the callback.

        Parameters
        ----------
        log_every_n_epochs : int, optional
            The frequency with which to plot and log, by default 100
        """
        assert split in ['train', 'val']
        self.split = split
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        """Logs plots at the end of the validation epoch.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer currently handling the model.
        pl_module : pytorch_lightning.LightningModule
            The model currently being trained.
        """
        # Skip evaluation for most epochs to save time
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Check for any image loggers
        if not has_image_loggers(trainer.loggers):
            return
        # Get the validation dataloaders
        if self.split == 'train':
            dataloader = trainer.train_dataloaders
        else:
            dataloader = trainer.val_dataloaders
        dataloader = dataloader[0]
        dataloader.freeze()
        # Compute outputs and plot for one session at a time
        states = []
        for batch in dataloader:
            # Move data to the right device
            inputs, target = batch
            inputs = inputs.to(pl_module.device)
            # Perform the forward pass through the model
            outputs, hs = pl_module(inputs)
            states.append(hs)
        states = torch.cat(states).detach().cpu().numpy()
        # Reduce dimensionality if necessary
        n_samp, n_step, n_lats = states.shape
        if n_lats > 3:
            states_flat = states.reshape(-1, n_lats)
            pca = PCA(n_components=3)
            states = pca.fit_transform(states_flat)
            states = states.reshape(n_samp, n_step, 3)
            explained_variance = np.sum(pca.explained_variance_ratio_)
        else:
            explained_variance = 1.0
        # Create figure and plot trajectories
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        for traj in states:
            ax.plot(*traj.T, alpha=0.4, linewidth=0.5)
        ax.scatter(*states[:, 0, :].T, alpha=0.2, s=10, c="g")
        ax.scatter(*states[:, -1, :].T, alpha=0.2, s=10, c="r")
        ax.set_title(f"explained variance: {explained_variance:.2f}")
        plt.tight_layout()
        # Log the figure
        log_figure(
            trainer.loggers,
            f"{self.split}/trajectory_plot",
            fig,
            trainer.global_step,
        )
        plt.close()


class TaskPlot(pl.Callback):
    """Plots example trials
    TODO: better labeling, trial consistency, especially with multi-task
    """

    def __init__(self, split='val', n_samples=3, log_every_n_epochs=100):
        """Initializes the callback.

        Parameters
        ----------
        log_every_n_epochs : int, optional
            The frequency with which to plot and log, by default 100
        """
        assert split in ['train', 'val']
        self.split = 'val'
        self.n_samples = n_samples
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        """Logs plots at the end of the validation epoch.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer currently handling the model.
        pl_module : pytorch_lightning.LightningModule
            The model currently being trained.
        """
        # Skip evaluation for most epochs to save time
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Check for any image loggers
        if not has_image_loggers(trainer.loggers):
            return
        # Get the validation dataloaders
        if self.split == 'train':
            dataloader = trainer.train_dataloaders
        else:
            dataloader = trainer.val_dataloaders
        dataloader = dataloader[0]
        dataloader.freeze()
        tot_samples = 0
        samples = []
        iterator = iter(dataloader)
        while tot_samples < self.n_samples:
            # Move data to the right device
            inputs, target = next(iterator)
            inputs = inputs.to(pl_module.device)
            # Perform the forward pass through the model
            outputs, hs = pl_module(inputs)
            samples.append((inputs, target, outputs))
            tot_samples += outputs.shape[0]
        inputs, target, outputs = zip(*samples)
        # Move everything back to CPU
        inputs = torch.cat(inputs).detach().cpu().numpy()
        target = torch.cat(target).detach().cpu().numpy()
        outputs = torch.cat(outputs).detach().cpu().numpy()
        # Plot inputs and outputs on subplots
        fig, axes = plt.subplots(
            2,
            self.n_samples,
            sharex=True,
            sharey="row",
            figsize=(3 * self.n_samples, 4),
        )
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        for i, ax_col in enumerate(axes.T): # TODO: axis labels and stuff
            ax_col[0].plot(inputs[i])
            if target.shape[2] != outputs.shape[2]:
                if (target.shape[2] == 1): # cross_entropy
                    outputs = np.argmax(outputs, axis=2, keepdims=True)
                    ax_col[1].plot(target[i,:,0], '--', color=colors[0])
                    ax_col[1].plot(outputs[i,:,0], '-', color=colors[0])
                else:
                    for j, color in zip(range(max(target.shape[2], outputs.shape[2])), colors):
                        if (j < target.shape[2]):
                            ax_col[1].plot(target[i,:,j], '--', color=color)
                        if (j < outputs.shape[2]):
                            ax_col[1].plot(outputs[i,:,j], '-', color=color)
            else:
                for j, color in zip(range(target.shape[2]), colors):
                    ax_col[1].plot(target[i,:,j], '--', color=color)
                    ax_col[1].plot(outputs[i,:,j], '-', color=color)
        plt.tight_layout()
        # Log the figure
        log_figure(
            trainer.loggers,
            f"{self.split}/task_plot",
            fig,
            trainer.global_step,
        )
        dataloader.unfreeze()
        plt.close()


class SuccessRate(pl.Callback):
    """Computes task success rate
    """

    def __init__(self, split='val', threshold=0.5, include_abort=False, log_every_n_epochs=100):
        """Initializes the callback.

        Parameters
        ----------
        log_every_n_epochs : int, optional
            The frequency with which to plot and log, by default 100
        """
        assert split in ['train', 'val']
        self.split = split
        self.threshold = threshold # only useful for supervised
        self.include_abort = include_abort # include aborted trials in success rate
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        """Logs success rate at the end of the validation epoch.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer currently handling the model.
        pl_module : pytorch_lightning.LightningModule
            The model currently being trained.
        """
        # Skip evaluation for most epochs to save time
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Get the validation dataloaders
        if self.split == 'train':
            dataloader = trainer.train_dataloaders
        else:
            dataloader = trainer.val_dataloaders
        dataloader = dataloader[0]
        dataloader.freeze()
        model_outputs = []
        task_targets = []
        for batch in dataloader:
            # Move data to the right device
            inputs, target = batch
            inputs = inputs.to(pl_module.device)
            # Perform the forward pass through the model
            outputs, hs = pl_module(inputs)
            model_outputs.append(outputs)
            task_targets.append(target)
        model_outputs = torch.cat(model_outputs).detach().cpu().numpy()
        task_targets = torch.cat(task_targets).detach().cpu().numpy()
        env = dataloader.dataset.env
        if isinstance(pl_module, SupervisedRNN):
            loss_func_name = pl_module.hparams.get('loss_func', 'mse_loss')
            if loss_func_name == 'mse_loss':
                success_rate = self._mse_loss_success_rate(model_outputs, task_targets, env)
            elif loss_func_name == 'cross_entropy':
                success_rate = self._cross_entropy_success_rate(model_outputs, task_targets, env)
        else:
            return
        pl_module.log(f"{self.split}/success_rate", success_rate)
        # print(success_rate)
        
    def _mse_loss_success_rate(self, model_outputs, task_targets, env):
        """Probably needs cleanup.
        Currently assuming that target outputs are between 0 and 1 for Box.
        Unsure if safe to assume there's always fixation for Box - currently not assuming that.
        Unsure if safe to assume target outputs are one-hot for Box - currently not assuming that.
        """
        if model_outputs.shape[1] != task_targets.shape[1]:
            # TODO: print warning message
            return -1
        if isinstance(env.action_space, gym.spaces.Box):
            # import pdb; pdb.set_trace()
            success = []
            for idx in range(model_outputs.shape[0]):
                output = model_outputs[idx]
                target = task_targets[idx]
                fixation_idx = env.unwrapped.action_space.name.get('fixation', None)
                if fixation_idx is None or not isinstance(fixation_idx, (float, int)):
                    trial_splits = [0, target.shape[0]]
                else:
                    fixation_idx = int(round(fixation_idx))
                    trial_splits = [0] + (np.where(np.diff(target[:, fixation_idx]) > 0)[0] + 1).tolist() + [target.shape[0]]
                for trial_num in range(len(trial_splits) - 1):
                    trial_success = 1
                    start, end = trial_splits[trial_num], trial_splits[trial_num + 1]
                    trial_output = output[start:end]
                    trial_target = target[start:end]
                    # Check for trial completeness
                    if fixation_idx is not None:
                        resp_mask = (trial_target[:,fixation_idx] == 0)
                        if np.sum(resp_mask) == 0:
                            continue
                    else:
                        resp_mask = np.ones((trial_target.shape[0],), dtype=bool)
                    if not np.any(trial_target[resp_mask, :] > 0):
                        continue
                    # Evaluate success
                    # fail conditions (during fixation): 
                    # - fixation drops below (1 - threshold) -> abort
                    # - any other output rises above threshold during fixation -> abort
                    # fail conditions (after fixation):
                    # - while target is high, output rises above threshold at least once
                    # - while target is low, output remains below (1 - threshold)
                    for col in range(trial_output.shape[1]):
                        col_target = trial_target[:, col]
                        col_output = trial_output[:, col]
                        changes = [0] + (np.where(np.diff(col_target))[0] + 1).tolist() + [col_target.shape[0]]
                        for period_num in range(len(changes) - 1):
                            period_start, period_end = changes[period_num], changes[period_num + 1]
                            period_target = col_target[period_start:period_end]
                            period_output = col_output[period_start:period_end]
                            if period_target[0] == 0: # zero period
                                if np.any(period_output > ((1 - self.threshold))):
                                    trial_success = 0 # fail
                                    break
                            elif period_target[0] > 0: # response period
                                if col == fixation_idx:
                                    if np.any(period_output < (self.threshold)):
                                        trial_success = -1 # abort
                                        break
                                else:
                                    if np.all(period_output < (self.threshold)):
                                        trial_success = 0 # fail
                                        break
                        if trial_success < 1:
                            break
                    success.append(trial_success)
            success = np.array(success)
            if self.include_abort:
                success[success < 0] = 0.
            else:
                success = success[success >= 0]
            success_rate = np.mean(success) if len(success) > 0 else -1
            return success_rate
        elif isinstance(env.action_space, gym.spaces.Discrete):
            return -1
        else:
            return -1

    def _cross_entropy_success_rate(self, model_outputs, task_targets, env):
        """Current criteria:
        During fixation, output is always fixation.
        After fixation, output must at some point be target output, 
            and must always be either fixation or target output (allowing for reaction time). 
        """
        if isinstance(env.action_space, (gym.spaces.Discrete, gym.spaces.Box)):
            success = []
            for idx in range(model_outputs.shape[0]):
                output = model_outputs[idx]
                output = np.argmax(output, axis=1)
                target = task_targets[idx]
                if target.shape[1] > 1:
                    target = np.argmax(target, axis=1) # assuming one-hot
                elif target.shape[1] == 1:
                    target = target[:,0]
                # How to identify trial start?
                fixation_val = env.action_space.name.get('fixation', None)
                if fixation_val is None or not isinstance(fixation_val, (float, int)):
                    trial_splits = [0, target.shape[0]]
                else:
                    trial_splits = [0] + (np.where(np.diff(target == fixation_val))[0] + 1).tolist() + [target.shape[0]]
                for trial_num in range(len(trial_splits) - 1):
                    trial_success = 1
                    start, end = trial_splits[trial_num], trial_splits[trial_num + 1]
                    trial_output = output[start:end]
                    trial_target = target[start:end]
                    # Check for trial completeness
                    if fixation_val is not None:
                        resp_mask = (trial_target != fixation_val)
                        if np.sum(resp_mask) == 0:
                            continue
                    else:
                        resp_mask = np.ones((trial_target.shape[0],), dtype=bool)
                    if not np.any(trial_target[resp_mask] > 0):
                        continue
                    # Evaluate success
                    # fail conditions (during fixation): 
                    # - output != fixation -> abort
                    # fail conditions (after fixation):
                    # - output != target at any point -> fail
                    # - output not in [target, fixation] at all points -> fail
                    changes = [0] + (np.where(np.diff(trial_target))[0] + 1).tolist() + [trial_target.shape[0]]
                    for period_num in range(len(changes) - 1):
                        period_start, period_end = changes[period_num], changes[period_num + 1]
                        period_target = trial_target[period_start:period_end]
                        period_output = trial_output[period_start:period_end]
                        if period_target[0] == fixation_val: # fixation period.
                            if np.any(period_output != fixation_val):
                                trial_success = -1
                                break
                        else: # response period
                            if not np.any(period_output == period_target[0]):
                                trial_success = 0 # fail
                                break
                            elif np.any(~np.isin(period_output, [fixation_val, period_target[0]])):
                                trial_success = 0 # fail
                                break
                    success.append(trial_success)
            success = np.array(success)
            if self.include_abort:
                success[success < 0] = 0.
            else:
                success = success[success >= 0]
            success_rate = np.mean(success) if len(success) > 0 else -1
            return success_rate
        else:
            # print warning
            return -1