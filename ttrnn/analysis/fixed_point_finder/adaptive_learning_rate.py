"""
adaptive_learning_rate.py
Adapted from: https://github.com/mattgolub/recurrent-whisperer/blob/master/AdaptiveLearningRate.py
Original author: Matt Golub, August 2018.
Please direct correspondence to mgolub@stanford.edu.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pickle

class AdaptiveLearningRate(object):
    def __init__(self, optimizer, 
        lr_init = 1.0,
        min_lr = 1e-3,
        max_n_steps = 1e4,
        n_warmup_steps = 0,
        warmup_scale = 1e-3,
        warmup_shape = 'gaussian',
        do_decrease_rate = True,
        decrease_patience = 5,
        decrease_factor = 0.95,
        do_increase_rate = True,
        increase_patience = 5,
        increase_factor = 1/0.95,
        verbose = False):

        # Attach optimizer
        if not isinstance(optimizer, optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        self.do_decrease_rate = do_decrease_rate
        if decrease_factor >= 1.0:
            raise ValueError('decrease_factor should be < 1.0.')
        self.decrease_factor = decrease_factor

        self.do_increase_rate = do_increase_rate
        if increase_factor < 1.0:
            raise ValueError('increase_factor should be >= 1.0.')
        self.increase_factor = increase_factor

        if lr_init < 0:
            raise ValueError('lr_init should be >= 0.0.')
        self.lr_init = lr_init

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        if warmup_shape not in ['exp', 'gaussian', 'linear']:
            raise ValueError('warmup_shape must be \'exp\' or \'gaussian\', ' + \
                f'but was {warmup_shape}')
        
        if n_warmup_steps < 0:
            raise ValueError('n_warmup-steps must be >= 0.0.')
        self.n_warmup_steps = n_warmup_steps

        if decrease_patience < 0:
            raise ValueError('decrease_patience must be >= 0.0.')
        self.decrease_patience = decrease_patience

        if increase_patience < 0:
            raise ValueError('increase_patience must be >= 0.0.')
        self.increase_patience = increase_patience

        self.save_filename = 'learning_rate.pkl'

        self.curr_step = 0
        self.max_n_steps = max_n_steps
        self.warmup_scale = warmup_scale
        self.warmup_shape = warmup_shape.lower()
        self.verbose = verbose
        self.step_last_update = -1
        self.prev_rate = None
        self.metric_log = []
        self.last_epoch = 0
        self.warmup_rates = self._get_warmup_rates()
        if n_warmup_steps > 0:
            self._set_lr(self.warmup_rates[0])
        else:
            self._set_lr(lr_init)

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        self.metric_log.append(current)
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.last_epoch <= self.n_warmup_steps:
            '''If step indicates that we are still in the warm-up, the new rate is determined entirely based on the warm-up schedule.'''
            if self.last_epoch < self.n_warmup_steps:
                new_lr = self.warmup_rates[self.last_epoch]
                self._set_lr(new_lr, epoch=epoch)
            else: # step == n_warmup_steps:
                new_lr = self.lr_init
                self._set_lr(new_lr, epoch=epoch)
                if self.verbose:
                    print(f'Warm-up complete (or no warm-up). Learning rate set to {new_lr:.2e}')
            self.step_last_update = self.last_epoch

        elif self._conditional_decrease_rate():
            self._decrease_lr()
            self.step_last_update = self.last_epoch
        elif self._conditional_increase_rate():
            self._increase_lr()
            self.step_last_update = self.last_epoch

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _set_lr(self, new_lr, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            new_lr = max(new_lr, self.min_lrs[i])
            param_group['lr'] = new_lr
            if self.verbose:
                epoch_str = ("%.2f" if isinstance(epoch, float) else
                                "%.5d") % epoch
                print('Epoch {}: setting learning rate'
                        ' of group {} to {:.4e}.'.format(epoch_str, i, new_lr))

    def _decrease_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.decrease_factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    epoch_str = ("%.2f" if isinstance(epoch, float) else
                                    "%.5d") % epoch
                    print('Epoch {}: reducing learning rate'
                            ' of group {} to {:.4e}.'.format(epoch_str, i, new_lr))

    def _increase_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.increase_factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    epoch_str = ("%.2f" if isinstance(epoch, float) else
                                    "%.5d") % epoch
                    print('Epoch {}: increasing learning rate'
                            ' of group {} to {:.4e}.'.format(epoch_str, i, new_lr))

    def _conditional_decrease_rate(self):
        '''Decreases the learning rate if the most recent loss is worse than
        all of the previous n loss values, and if no learning rate changes
        have been made in the last n steps, where n=min_steps_per_decrease.
        Args:
            None.
        Returns:
            A bool indicating whether the learning rate was decreased.
        '''
        decrease_rate = False
        n = self.decrease_patience

        if self.do_decrease_rate and self.last_epoch>=(self.step_last_update + n):
            batch_loss_window = self.metric_log[-(1+n):]
            decrease_rate = all(np.greater(batch_loss_window[-1],batch_loss_window[:-1]))

        return decrease_rate

    def _conditional_increase_rate(self):
        '''Increases the learning rate if loss values have monotonically
        decreased over the past n steps, and if no learning rate changes have
        been made in the last n steps, where n=min_steps_per_increase.
        Args:
            None.
        Returns:
            A bool indicating whether the learning rate was increased.
        '''
        increase_rate = False
        n = self.increase_patience

        if self.do_increase_rate and self.last_epoch>=(self.step_last_update + n):
            batch_loss_window = self.metric_log[-(1+n):]
            increase_rate = all(np.less(batch_loss_window[1:],batch_loss_window[:-1]))

        return increase_rate

    def is_finished(self, do_check_step=True, do_check_rate=True):
        ''' Indicates termination of the optimization procedure. Note: this
        function is never used internally and does not influence the behavior
        of the adaptive learning rate.
        Args:
            do_check_step: Bool indicating whether to check if the step has
            reached max_n_steps.
            do_check_rate: Bool indicating whether to check if the learning rate
            has fallen below min_rate.
        Returns:
            Bool indicating whether any of the termination criteria have been
            met.
        '''

        if do_check_step and self.curr_step > self.max_n_steps:
            return True
        elif self.curr_step <= self.n_warmup_steps:
            return False
        elif do_check_rate and all([(lr <= self.min_lr) for lr in self._last_lr]):
            return True
        else:
            return False

    @property
    def min_steps(self):
        ''' Computes the minimum number of steps required before the learning
        rate falls below the min_rate, i.e., assuming the rate decreases at
        every opportunity permitted by the properties of this
        AdaptiveLearningRate object.
        Args:
            None.
        Returns:
            An int specifying the minimum number of steps in the adaptive
            learning rate schedule.
        '''
        n_decreases = np.ceil(np.divide(
            (np.log(self.min_lr) - np.log(self.lr_init)),
            np.log(self.decrease_factor)))
        return self.n_warmup_steps + self.decrease_patience * n_decreases

    def _get_warmup_rates(self):
        '''Determines the warm-up schedule of learning rates, culminating at
        the desired initial rate.
        Args:
            None.
        Returns:
            Shape (n_warmup_steps,) numpy array containing the learning rates
            for each step of the warm-up period.
        '''
        n = self.n_warmup_steps
        warmup_shape = self.warmup_shape
        scale = self.warmup_scale
        warmup_start = scale*self.lr_init
        warmup_stop = self.lr_init

        if warmup_shape == 'linear':
            warmup_rates = np.linspace(warmup_start, warmup_stop, n+1)[:-1]
        if self.warmup_shape == 'exp':
            warmup_rates = np.logspace(
                np.log10(warmup_start), np.log10(warmup_stop), n+1)[:-1]
        elif self.warmup_shape == 'gaussian':
            mu = np.float32(n)
            x = np.arange(mu)

            # solve for sigma s.t. warmup_rates[0] = warmup_start
            sigma = np.sqrt(-mu**2.0 / (2.0*np.log(warmup_start/warmup_stop)))

            warmup_rates = warmup_stop*np.exp((-(x-mu)**2.0)/(2.0*sigma**2.0))

        return warmup_rates.tolist()

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def save(self, save_dir):
        '''Saves the current state of the AdaptiveLearningRate object.
        Args:
            save_dir: A string containing the directory in which to save.
        Returns:
            None.
        '''
        if self.verbose:
            print('Saving AdaptiveLearningRate.')
        save_path = os.path.join(save_dir, self.save_filename)
        file = open(save_path,'wb')
        file.write(pickle.dumps(self.state_dict()))
        file.close

    def restore(self, restore_dir):
        '''Restores the state of a previously saved AdaptiveLearningRate
        object.
        Args:
            restore_dir: A string containing the directory in which to find a
            previously saved AdaptiveLearningRate object.
        Returns:
            None.
        '''
        if self.verbose:
            print('Restoring AdaptiveLearningRate.')
        restore_path = os.path.join(restore_dir, self.save_filename)
        file = open(restore_path,'rb')
        restore_data = file.read()
        file.close()
        self.load_state_dict(pickle.loads(restore_data))