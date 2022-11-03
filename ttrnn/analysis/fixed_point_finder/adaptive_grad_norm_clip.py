'''
AdaptiveGradNormClip.py
Written for Python 3.6.9
@ Matt Golub, August 2018.
Please direct correspondence to mgolub@stanford.edu.
'''
import numpy as np
import torch
import os
import pickle

class AdaptiveGradNormClip(object):
    """Class for managing adaptive gradient norm clipping for stabilizing any gradient-descent-like procedure.
    Essentially, just a running buffer of gradient norms from the last n gradient steps, with a hook into the x-th 
    percentile of those values, which is intended to be used to set the ceiling on the gradient applied at the next 
    iteration of a gradient-descent-like procedure.

    The standard usage is as follows:
    ```python
    # Set hyperparameters as desired.
    agnc_hps = dict()
    agnc_hps['sliding_window_len'] = 1.0
    agnc_hps['percentile'] = 95
    agnc_hps['init_clip_val' = 1.0
    agnc_hps['verbose'] = False
    agnc = AdaptiveGradNormClip(**agnc_hps)
    while some_conditions(...):
        # This loop defines one step of the training procedure.
        gradients = get_gradients(data, params)
        grad_norm = compute_gradient_norm(gradients)
        clip_val = agnc.update(grad_norm)
        clipped_gradients = clip_gradients(gradients, clip_val)
        params = apply_gradients(clipped_gradients)
        # (Optional): Occasionally save model checkpoints along with the
        # AdaptiveGradNormClip object (for seamless restoration of a training
        # session)
        if some_other_conditions(...):
            save_checkpoint(params, ...)
            agnc.save(...)
    ```
    """

    def __init__(self, 
        do_adaptive_clipping = True,
        sliding_window_len = 128,
        percentile = 95.0,
        init_clip_val = 1e12,
        max_clip_val = 1e12,
        verbose = False):

        self.curr_step = 0
        self.do_adaptive_clipping = do_adaptive_clipping
        self.sliding_window_len = sliding_window_len
        self.percentile = percentile
        self.max_clip_val = max_clip_val
        self.grad_norm_log = []
        self.verbose = verbose
        self.save_filename = 'norm_clip.pkl'

        if self.do_adaptive_clipping:
            self.clip_val = init_clip_val
        else:
            self.clip_val = self.max_clip_val
    
    def __call__(self, p):
        '''Clips gradients
        '''
        norm = torch.nn.utils.clip_grad_norm_(p, self.clip_val)
        self._update(norm.item())

    def _update(self, grad_norm):
        '''Update the log of recent gradient norms and the corresponding
        recommended clip value.
        Args:
            grad_norm: A float specifying the gradient norm from the most
            recent gradient step.
        Returns:
            None.
        '''
        if self.do_adaptive_clipping:
            if self.step < self.sliding_window_len:
                # First fill up an entire "window" of values
                self.grad_norm_log.append(grad_norm)
            else:
                # Once the window is full, overwrite the oldest value
                idx = np.mod(self.step, self.sliding_window_len)
                self.grad_norm_log[idx] = grad_norm

            proposed_clip_val = \
                np.percentile(self.grad_norm_log, self.percentile)

            self.clip_val = min(proposed_clip_val, self.max_clip_val)

        self.curr_step += 1

    def save(self, save_dir):
        '''Saves the current AdaptiveGradNormClip state, enabling seamless restoration of gradient descent training procedure.
        Args:
            save_dir: A string containing the directory in which to save the
            current object state.
        Returns:
            None.
        '''
        if self.verbose:
            print('Saving AdaptiveGradNormClip.')
        save_path = os.path.join(save_dir, self.save_filename)
        file = open(save_path,'wb')
        file.write(pickle.dumps(self.__dict__))
        file.close

    def restore(self, restore_dir):
        '''Loads a previously saved AdaptiveGradNormClip state, enabling seamless restoration of gradient descent training procedure.
        Args:
            restore_dir: A string containing the directory from which to load
            a previously saved object state.
        Returns:
            None.
        '''
        if self.verbose:
            print('Restoring AdaptiveGradNormClip.')
        restore_path = os.path.join(restore_dir, self.save_filename)
        file = open(restore_path,'rb')
        restore_data = file.read()
        file.close()
        self.__dict__ = pickle.loads(restore_data)