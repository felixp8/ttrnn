import io
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import copy
from PIL import Image
from scipy.linalg import LinAlgWarning
from sklearn.decomposition import PCA
import gym
from .trainer import Supervised, A2C

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


class TaskPerformance(pl.Callback):
    """Computes task success rate
    """

    def __init__(self, split='val', threshold=0.0, burn_in=5, log_every_n_epochs=100, n_test_trials=50):
        """Initializes the callback.

        Parameters
        ----------
        log_every_n_epochs : int, optional
            The frequency with which to plot and log, by default 100
        """
        assert split in ['train', 'val']
        self.split = split
        self.threshold = threshold # only useful for supervised MSE
        self.burn_in = burn_in # samples to ignore at the beginning
        # self.include_abort = include_abort # include aborted trials in success rate
        self.log_every_n_epochs = log_every_n_epochs
        self.n_test_trials = n_test_trials

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
        if isinstance(pl_module, Supervised):
            success_rate, mean_reward = self.supervised_success_rate(trainer, pl_module)
        elif isinstance(pl_module, A2C):
            success_rate, mean_reward = self.rl_success_rate(trainer, pl_module)
        print(success_rate, mean_reward)
    
    def supervised_success_rate(self, trainer, pl_module):
        # Get the validation dataloaders
        if self.split == 'train':
            dataloader = trainer.train_dataloaders
        else:
            dataloader = trainer.val_dataloaders
        dataloader = dataloader[0]
        if not dataloader.dataset.save_envs:
            return
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
        # should possibly be readout dependent - softmax vs. linear.
        # these, however, should also be tied to loss func ... I think
        loss_func_name = pl_module.hparams.get('loss_func', 'mse_loss')
        if loss_func_name == 'mse_loss':
            success_rate, mean_reward = self._mse_loss_success_rate(model_outputs, task_targets, dataloader.dataset)
        elif loss_func_name == 'cross_entropy':
            success_rate, mean_reward = self._cross_entropy_success_rate(model_outputs, task_targets, dataloader.dataset)
        # pl_module.log(f"{self.split}/success_rate", success_rate)
        # pl_module.log(f"{self.split}/mean_reward", mean_reward)
        # print(success_rate, mean_reward)
        return success_rate, mean_reward
        
    def _mse_loss_success_rate(self, model_outputs, task_targets, dataset):
        """Needs cleanup.
        """
        # if model_outputs.shape[-1] > 1:
        #     # 'Box' output
        #     def get_actions(output, env):
        #         # output is T x C dimensional
        #         actions = np.argmax(output, axis=1).astype(int)
        #         above_thresh = np.any(output >= self.threshold, axis=1)
        #         actions = actions * above_thresh # assumes fixation output is 0
        #         # valid = np.all(above_thresh == 1)
        #         valid = True # currently allows multiple above-threshold outputs
        #         return actions, valid
        # else:
        #     # 'Discrete' output, not recommended
        #     # Probably technically fails every time unless output jumps rapidly
        #     def get_actions(output, env):
        #         # output is T x 1 or T dimensional
        #         actions = np.round(output).astype(int)
        #         n = env.unwrapped.action_space.n
        #         valid = np.all(np.logical_and(actions >= 0, actions < n))
        #         return actions, valid
        success = []
        rewards = []
        for idx in range(model_outputs.shape[0]):
            env_list = dataset.stored_envs[idx]
            output = model_outputs[idx]
            target = task_targets[idx]
            start_t = 0
            for env in env_list:
                tlen = env.unwrapped.gt.shape[0]
                if (start_t + tlen > output.shape[0]): # incomplete trial
                    continue
                if hasattr(env, 'set_threshold'):
                    env.set_threshold(self.threshold)
                trial_output = output[start_t:(start_t + tlen)]
                trial_target = target[start_t:(start_t + tlen)]
                # actions, valid = get_actions(trial_output, env)
                actions = trial_output
                valid = True
                if not valid:
                    # treat invalid output as abort
                    success.append(env.unwrapped.rewards.get('abort', -0.1))
                    continue
                env = copy.deepcopy(env)
                while (env.unwrapped.t_ind < self.burn_in):
                    ob, reward, done, info = env.step(env.gt_now)
                if done:
                    print('WARNING: trial terminated in burn-in.')
                    # import pdb; pdb.set_trace()
                while not done and not info['new_trial'] and (env.unwrapped.t_ind < actions.shape[0]):
                    action = actions[env.unwrapped.t_ind]
                    ob, reward, done, info = env.step(action)
                # import pdb; pdb.set_trace()
                performance = info['performance']
                success.append(performance)
                rewards.append(reward)
                start_t += tlen
        success = np.array(success)
        rewards = np.array(rewards)
        # if self.include_abort: # assuming abort < 0, fail == 0, success > 0
        #     success[success < 0] = 0.
        # else:
        #     success = success[success >= 0]
        # success[success > 0] = 1.
        success_rate = np.mean(success) if len(success) > 0 else -0.1
        mean_reward = np.mean(rewards) if len(rewards) > 0 else -0.1
        # import pdb; pdb.set_trace()
        return success_rate, mean_reward

    def _cross_entropy_success_rate(self, model_outputs, task_targets, dataset):
        """Needs cleanup
        """
        # 'Box' output
        def get_actions(output, env):
            # output is T x C dimensional
            actions = np.argmax(output, axis=1).astype(int)
            # above_thresh = np.sum(output >= self.threshold, axis=1)
            # valid = np.all(above_thresh == 1)
            return actions # , valid
        success = []
        rewards = []
        for idx in range(model_outputs.shape[0]):
            env_list = dataset.stored_envs[idx]
            output = model_outputs[idx]
            # target = task_targets[idx]
            start_t = 0
            for env in env_list:
                tlen = env.unwrapped.gt.shape[0]
                if (start_t + tlen > output.shape[0]): # incomplete trial
                    continue
                if hasattr(env, 'set_threshold'):
                    env.set_threshold(self.threshold)
                trial_output = output[start_t:(start_t + tlen)]
                actions = get_actions(trial_output, env)
                # if not valid:
                #     # treat invalid output as abort
                #     success.append(env.unwrapped.rewards.get('abort', -0.1))
                #     continue
                env = copy.deepcopy(env)
                while (env.unwrapped.t_ind < self.burn_in):
                    ob, reward, done, info = env.step(env.gt_now)
                if done:
                    print('WARNING: trial terminated in burn-in.')
                while not done and not info['new_trial'] and (env.unwrapped.t_ind < actions.shape[0]):
                    action = actions[env.unwrapped.t_ind]
                    ob, reward, done, info = env.step(action)
                    # import pdb; pdb.set_trace()
                performance = info['performance']
                success.append(performance)
                rewards.append(reward)
                start_t += tlen
                # if reward == 0 and any([s > 0 for s in success]):
                #     import pdb; pdb.set_trace()
        success = np.array(success)
        rewards = np.array(rewards)
        # if self.include_abort: # assuming abort < 0, fail == 0, success > 0
        #     success[success < 0] = 0.
        # else:
        #     success = success[success >= 0]
        # success[success > 0] = 1.
        success_rate = np.mean(success) if len(success) > 0 else -0.1
        mean_reward = np.mean(rewards) if len(rewards) > 0 else -0.1
        return success_rate, mean_reward

    def rl_success_rate(self, trainer, pl_module):
        env = pl_module.env
        with torch.no_grad():
            success = []
            rewards = []
            for _ in range(self.n_test_trials):
                obs = env.reset()
                hx = pl_module.model.rnn.build_initial_state(1, pl_module.device, torch.float)
                n_steps = 0
                trial_rewards = []
                while True:
                    action_logits, _, hx = pl_module.model(
                        torch.from_numpy(obs).to(pl_module.device).unsqueeze(0), hx=hx)
                    action = action_logits.sample().item()
                    # action = action_logits.probs.argmax().item()
                    obs, reward, done, info = env.step(action)
                    trial_rewards.append(reward)
                    if done or info['new_trial']:
                        success.append(info.get('performance', 0.0))
                        rewards.append(np.sum(trial_rewards))
                        break
                    n_steps += 1
                    if n_steps > 500: # timeout
                        import pdb; pdb.set_trace()
                        success.append(0)
                        rewards.append(0)
                        break
        success_rate = np.mean(success) if len(success) > 0 else -0.1
        mean_reward = np.mean(rewards) if len(rewards) > 0 else -0.1
        return success_rate, mean_reward
        
