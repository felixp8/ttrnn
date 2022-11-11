'''
fixed_point_finder.py
Adapted from: https://github.com/mattgolub/fixed-point-finder/blob/master/FixedPointFinder.py
Original author: Matt Golub, October 2018.
Please direct correspondence to mgolub@stanford.edu.
'''
import numpy as np
import torch
import time
from copy import deepcopy

from .fixed_points import FixedPoints
from .adaptive_learning_rate import AdaptiveLearningRate
from .adaptive_grad_norm_clip import AdaptiveGradNormClip
# from .timer import Timer

def to_torch(*args, device=None):
    def arr_to_torch(arr, device):
        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr).to(device)
        elif isinstance(arr, torch.Tensor):
            if device is None:
                device = arr.device
            return arr.to(device)
        elif isinstance(arr, (tuple, list)):
            return (arr_to_torch(a, device) for a in arr)
        else:
            raise AssertionError(f"Cannot handle type {type(arr)}")
    return (arr_to_torch(arr, device) for arr in args)

def to_numpy(*args):
    def arr_to_numpy(arr):
        if isinstance(arr, torch.Tensor):
            return arr.cpu().detach().numpy()
        elif isinstance(arr, np.ndarray):
            return arr
        elif isinstance(arr, (tuple, list)):
            return (arr_to_numpy(a) for a in arr)
        else:
            raise AssertionError(f"Cannot handle type {type(arr)}")
    return (arr_to_numpy(arr) for arr in args)

class FixedPointFinder:
    def __init__(self, rnn_cell,
        feed_dict = {},
        tol_q = 1e-12,
        tol_dq = 1e-20,
        max_iters = 5000,
        method = 'joint',
        do_rerun_q_outliers = False,
        outlier_q_scale = 10.0,
        do_exclude_distance_outliers = True,
        outlier_distance_scale = 10.0,
        tol_unique = 1e-3,
        max_n_unique = np.inf,
        do_compute_jacobians = True,
        do_decompose_jacobians = True,
        dtype = 'float32',
        random_seed = 0,
        verbose = True,
        super_verbose = False,
        n_iters_per_print_update = 100,
        alr_hps = {}, # Note: ALR's termination criteria not currently used.
        agnc_hps = {},
        adam_hps = {'eps': 0.01},
        device = None):
        super(FixedPointFinder, self).__init__()

        self.rnn_cell = rnn_cell

        self.feed_dict = feed_dict
        self.dtype = dtype

        # Make random sequences reproducible
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

        self.tol_q = tol_q
        self.tol_dq = tol_dq
        self.method = method
        self.max_iters = max_iters
        self.do_rerun_q_outliers = do_rerun_q_outliers
        self.outlier_q_scale = outlier_q_scale
        self.do_exclude_distance_outliers = do_exclude_distance_outliers
        self.outlier_distance_scale = outlier_distance_scale
        self.tol_unique = tol_unique
        self.max_n_unique = max_n_unique
        self.do_compute_jacobians = do_compute_jacobians
        self.do_decompose_jacobians = do_decompose_jacobians
        self.verbose = verbose
        self.super_verbose = super_verbose
        self.n_iters_per_print_update = n_iters_per_print_update

        self.adaptive_learning_rate_hps = alr_hps
        self.grad_norm_clip_hps = agnc_hps
        self.adam_optimizer_hps = adam_hps

        self.device = device

    def sample_inputs_and_states(self, inputs, state_traj, n_inits,
        valid_bxt=None,
        noise_scale=0.0):
        '''Draws random paired samples from the RNN's inputs and hidden-state
        trajectories. Sampled states (but not inputs) can optionally be
        corrupted by independent and identically distributed (IID) Gaussian
        noise. These samples are intended to be used as initial states for
        fixed point optimizations.
        Args:
            inputs: [n_batch x n_time x n_inputs] numpy array containing input
            sequences to the RNN.
            state_traj: [n_batch x n_time x n_states] numpy array or
            LSTMStateTuple with .c and .h as [n_batch x n_time x n_states]
            numpy arrays. Contains state trajectories of the RNN, given inputs.
            n_inits: int specifying the number of sampled states to return.
            valid_bxt (optional): [n_batch x n_time] boolean mask indicated
            the set of trials and timesteps from which to sample. Default: all
            trials and timesteps are assumed valid.
            noise_scale (optional): non-negative float specifying the standard
            deviation of IID Gaussian noise samples added to the sampled
            states. Default: 0.0.
        Returns:
            inputs: Sampled RNN inputs as a [n_inits x n_inputs] numpy array.
            These are paired with the states in initial_states (below).
            initial_states: Sampled RNN states as a [n_inits x n_states] numpy
            array or as an LSTMStateTuple with .c and .h as
            [n_inits x n_states] numpy arrays (type matches that of
            state_traj).
        Raises:
            ValueError if noise_scale is negative.
        '''
        state_traj, inputs = to_torch(state_traj, inputs, device=self.device)
        if isinstance(state_traj, (tuple, list)):
            splits = [arr.shape[1] for arr in state_traj]
            state_traj_bxtxd = torch.cat(state_traj, dim=1)
            post_split = True
        else:
            state_traj_bxtxd = state_traj
            post_split = False

        [n_batch, n_time, n_states] = state_traj_bxtxd.shape
        n_inputs = inputs.shape[2]

        valid_bxt = self._get_valid_mask(n_batch, n_time, valid_bxt=valid_bxt)
        trial_indices, time_indices = \
            self._sample_trial_and_time_indices(valid_bxt, n_inits)

        # Draw random samples from inputs and state trajectories
        input_samples = torch.zeros([n_inits, n_inputs], device=self.device)
        state_samples = torch.zeros([n_inits, n_states], device=self.device)
        for init_idx in range(n_inits):
            trial_idx = trial_indices[init_idx]
            time_idx = time_indices[init_idx]
            input_samples[init_idx,:] = inputs[trial_idx,time_idx,:]
            state_samples[init_idx,:] = state_traj_bxtxd[trial_idx,time_idx,:]

        # Add IID Gaussian noise to the sampled states
        state_samples = self._add_gaussian_noise(
            state_samples, noise_scale)

        assert not torch.any(torch.isnan(state_samples)),\
            'Detected NaNs in sampled states. Check state_traj and valid_bxt.'

        assert not torch.any(torch.isnan(input_samples)),\
            'Detected NaNs in sampled inputs. Check inputs and valid_bxt.'

        if post_split:
            state_samples = torch.split(state_samples, splits, dim=1)
            return input_samples, state_samples
        else:
            return input_samples, state_samples

    def sample_states(self, state_traj, n_inits,
        valid_bxt=None,
        noise_scale=0.0):
        '''Draws random samples from trajectories of the RNN state. Samples
        can optionally be corrupted by independent and identically distributed
        (IID) Gaussian noise. These samples are intended to be used as initial
        states for fixed point optimizations.
        Args:
            state_traj: [n_batch x n_time x n_states] numpy array or
            LSTMStateTuple with .c and .h as [n_batch x n_time x n_states]
            numpy arrays. Contains example trajectories of the RNN state.
            n_inits: int specifying the number of sampled states to return.
            valid_bxt (optional): [n_batch x n_time] boolean mask indicated
            the set of trials and timesteps from which to sample. Default: all
            trials and timesteps are assumed valid.
            noise_scale (optional): non-negative float specifying the standard
            deviation of IID Gaussian noise samples added to the sampled
            states.
        Returns:
            initial_states: Sampled RNN states as a [n_inits x n_states] numpy
            array or as an LSTMStateTuple with .c and .h as [n_inits x
            n_states] numpy arrays (type matches than of state_traj).
        Raises:
            ValueError if noise_scale is negative.
        '''
        state_traj, = to_torch(state_traj, device=self.device)
        if isinstance(state_traj, (tuple, list)):
            splits = [arr.shape[1] for arr in state_traj]
            state_traj_bxtxd = torch.cat(state_traj, dim=1)
            post_split = True
        else:
            state_traj_bxtxd = state_traj
            post_split = False

        [n_batch, n_time, n_states] = state_traj_bxtxd.shape

        valid_bxt = self._get_valid_mask(n_batch, n_time, valid_bxt=valid_bxt)
        trial_indices, time_indices = self._sample_trial_and_time_indices(
            valid_bxt, n_inits)

        # Draw random samples from state trajectories
        states = torch.zeros([n_inits, n_states])
        for init_idx in range(n_inits):
            trial_idx = trial_indices[init_idx]
            time_idx = time_indices[init_idx]
            states[init_idx,:] = state_traj_bxtxd[trial_idx, time_idx]

        # Add IID Gaussian noise to the sampled states
        states = self._add_gaussian_noise(states, noise_scale)

        assert not torch.any(torch.isnan(states)),\
            'Detected NaNs in sampled states. Check state_traj and valid_bxt.'

        if post_split:
            state_samples = torch.split(state_samples, splits, dim=1)
            return state_samples
        else:
            return state_samples

    def _sample_trial_and_time_indices(self, valid_bxt, n):
        ''' Generate n random indices corresponding to True entries in
        valid_bxt. Sampling is performed without replacement.
        Args:
            valid_bxt: [n_batch x n_time] bool numpy array.
            n: integer specifying the number of samples to draw.
        returns:
            (trial_indices, time_indices): tuple containing random indices
            into valid_bxt such that valid_bxt[i, j] is True for every
            (i=trial_indices[k], j=time_indices[k])
        '''

        (trial_idx, time_idx) = torch.nonzero(valid_bxt, as_tuple=True)
        max_sample_index = len(trial_idx) # same as len(time_idx)
        sample_indices = self.rng.randint(max_sample_index, size=n).astype(int)
        sample_indices = torch.from_numpy(sample_indices).to(self.device)

        return trial_idx[sample_indices], time_idx[sample_indices]

    def find_candidate_fps(self, state_traj, fp_tol=1e-6, fp_min_t=4):
        tlen = state_traj.shape[1]
        dists = torch.cdist(state_traj, state_traj)
        # for now, just loop
        base_indices = torch.triu_indices(fp_min_t, fp_min_t, offset=1)
        candidate_fps = []
        candidate_dists = []
        for t in range(0, tlen - fp_min_t):
            blocks = dists[:, base_indices[0,:] + t, base_indices[1,:] + t]
            blocks_mean = blocks.mean(dim=1)
            matches = torch.nonzero(blocks_mean < fp_tol)
            if matches.numel() > 0:
                fps = state_traj[matches, t, :]
                candidate_fps.append(fps)
                candidate_dists.append(blocks_mean[matches])
        if len(candidate_fps) == 0:
            return torch.empty(0, state_traj.shape[2]), torch.empty()
        candidate_fps = torch.cat(candidate_fps, dim=0)
        candidate_dists = torch.cat(candidate_dists, dim=0)
        # remove duplicates
        fp_dists = torch.nn.functional.pdist(candidate_fps)
        dist_idxs = torch.triu_indices(candidate_fps.shape[0], candidate_fps.shape[0], offset=1)
        # again, just loop
        duplicates = []
        for i in range(len(fp_dists)): # don't need loop if using fp_tol
            dist = fp_dists[i]
            fp_1 = dist_idxs[0, i]
            fp_2 = dist_idxs[1, i]
            # if (dist < candidate_dists[fp_1]) and (dist < candidate_dists[fp_2]):
            if dist < fp_tol:
                duplicates.append((fp_1, fp_2))
        # remove most common fps until no duplicates
        remove_list = []
        while len(duplicates) > 0:
            dup_flat = [fp for tup in duplicates for fp in tup]
            fp, count = np.unique(dup_flat, return_counts=True)
            remove_fp = fp[np.argmax(count)]
            duplicates = [tup for tup in duplicates in remove_fp not in tup]
            remove_list.append(remove_fp)
        keep_idx = torch.Tensor([i for i in range(candidate_fps.shape[0]) if i not in remove_list])
        candidate_fps = candidate_fps[keep_idx]
        candidate_dists = candidate_dists[keep_idx]
        return candidate_fps, candidate_dists  
        
    def _build_state_vars(self, initial_states, inputs):
        initial_states, inputs = to_torch(initial_states, inputs, device=self.device)
        initial_states = self._concat_state(initial_states)
        return initial_states, inputs

    def find_fixed_points(self, initial_states, inputs, cond_ids=None):
        '''Finds RNN fixed points and the Jacobians at the fixed points.
        Args:
            initial_states: Either an [n x n_states] numpy array or an
            LSTMStateTuple with initial_states.c and initial_states.h as
            [n x n_states] numpy arrays. These data specify the initial
            states of the RNN, from which the optimization will search for
            fixed points. The choice of type must be consistent with state
            type of rnn_cell.
            inputs: Either a [1 x n_inputs] numpy array specifying a set of
            constant inputs into the RNN to be used for all optimization
            initializations, or an [n x n_inputs] numpy array specifying
            potentially different inputs for each initialization.
        Returns:
            unique_fps: A FixedPoints object containing the set of unique
            fixed points after optimizing from all initial_states. Two fixed
            points are considered unique if all absolute element-wise
            differences are less than tol_unique AND the corresponding inputs
            are unique following the same criteria. See FixedPoints.py for
            additional detail.
            all_fps: A FixedPoints object containing the likely redundant set
            of fixed points (and associated metadata) resulting from ALL
            initializations in initial_states (i.e., the full set of fixed
            points before filtering out putative duplicates to yield
            unique_fps).
        '''
        # get number of initial conditions
        # n = initial_states[0].shape[0] if isinstance(initial_states, tuple) else initial_states.shape[0]
        initial_states, inputs = to_torch(initial_states, inputs, device=self.device)

        # State shape checking
        if isinstance(initial_states, (tuple, list)):
            initial_states = list(initial_states)
            # States are of shape (N,H) or (H,)
            n_init = initial_states[0].shape[0] if (initial_states[0].dim() == 2) else 1
            # unbatched init state handling
            for i, arr in enumerate(initial_states):
                if arr.dim() == 1: # unbatched: shape = (H,)
                    initial_states[i] = arr.unsqueeze(0) # (H,) -> (N,H)
                    assert (n_init == 1), "Differing numbers of initial states found. " + \
                        f"Array 0 has {n_init} different states while array {i} has 1."
                elif arr.dim() == 2:
                    assert (n_init == arr.shape[0]), "Differing numbers of initial states found. " + \
                        f"Array 0 has {n_init} different states while array {i} has {arr.shape[0]}."
                else:
                    raise AssertionError(f"Initial state arrays must be 1-d or 2-d, found {arr.dim()}-d array.")
            splits = [arr.shape[1] for arr in initial_states]
            self._concat_state = lambda x: torch.cat(x, dim=1)
            self._split_state = lambda x: torch.split(x, splits, dim=1)
            self._concat_state_np = lambda x: np.concatenate(x, axis=1)
            self._split_state_np = lambda x: np.split(x, np.cumsum(splits)[:-1], axis=1)
        else:
            # States are of shape (N,H) or (H,)
            # unbatched init state handling
            if initial_states.dim() == 1:
                initial_states = initial_states.unsqueeze(0)
            n_init = initial_states.shape[0]
            self._concat_state = lambda x: x
            self._split_state = lambda x: x
            self._concat_state_np = lambda x: x
            self._split_state_np = lambda x: x
        
        # Input shape checking
        # Inputs are of shape (,U) or (U,)
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        # Tile repeated inputs
        if inputs.shape[0] == 1:
            inputs = torch.tile(inputs, [n_init, 1])
        assert (n_init == inputs.shape[0]), "Must provide either 1 input for all initializations or " + \
            f"the 1 input for each initialization. Got {inputs.shape[0]} inputs for {n_init} initializations."

        if self.method == 'sequential':
            all_fps = self._run_sequential_optimizations(
                initial_states, inputs, cond_ids=cond_ids)
        elif self.method == 'joint':
            all_fps = self._run_joint_optimization(
                initial_states, inputs, cond_ids=cond_ids)
        else:
            raise ValueError("Unsupported optimization method. Must be either " + \
                f"'joint' or 'sequential', but was '{self.method}'")
        
        # Filter out duplicates after from the first optimization round
        unique_fps = all_fps.get_unique()

        self._print_if_verbose(f'\tIdentified {unique_fps.n} unique fixed points.')
        
        if self.do_exclude_distance_outliers:
            unique_fps = \
                self._exclude_distance_outliers(unique_fps, initial_states)
        
        # Optionally run additional optimization iterations on identified
        # fixed points with q values on the large side of the q-distribution.
        if self.do_rerun_q_outliers:
            unique_fps = \
                self._run_additional_iterations_on_outliers(unique_fps)

            # Filter out duplicates after from the second optimization round
            unique_fps = unique_fps.get_unique()
        
        # Optionally subselect from the unique fixed points (e.g., for
        # computational savings when not all are needed.)
        if unique_fps.n > self.max_n_unique:
            self._print_if_verbose(f'\tRandomly selecting {self.max_n_unique} unique '
                'fixed points to keep.')
            max_n_unique = int(self.max_n_unique)
            idx_keep = self.rng.choice(
                unique_fps.n, max_n_unique, replace=False)
            unique_fps = unique_fps[idx_keep]
        
        if self.do_compute_jacobians:
            if unique_fps.n > 0:

                self._print_if_verbose(f'\tComputing input and recurrent Jacobians at {unique_fps.n} '
                    'unique fixed points.')
                dFdx, dFdu = self._compute_jacobians(unique_fps)
                unique_fps.J_xstar = dFdx
                unique_fps.dFdu = dFdu

            else:
                # Allocate empty arrays, needed for robust concatenation
                n_states = unique_fps.n_states
                n_inputs = unique_fps.n_inputs

                shape_dFdx = (0, n_states, n_states)
                shape_dFdu = (0, n_states, n_inputs)

                unique_fps.J_xstar = unique_fps._alloc_nan(shape_dFdx)
                unique_fps.dFdu = unique_fps._alloc_nan(shape_dFdu)

            if self.do_decompose_jacobians:
                # self._test_decompose_jacobians(unique_fps, J_np, J_tf)
                unique_fps.decompose_jacobians(str_prefix='\t')

        self._print_if_verbose('\tFixed point finding complete.\n')

        return unique_fps, all_fps

    def _run_joint_optimization(self, initial_states, inputs, cond_ids=None):
        x, u = self._build_state_vars(initial_states, inputs)

        xstar, F_xstar, qstar, dq, n_iters = \
            self._run_optimization_loop(x, u)

        fps = FixedPoints(
            xstar=xstar,
            x_init=x.cpu().detach().numpy(),
            inputs=inputs,
            cond_id=cond_ids,
            F_xstar=F_xstar,
            qstar=qstar,
            dq=dq,
            n_iters=n_iters,
            tol_unique=self.tol_unique,
            dtype=self.dtype)

        return fps

    def _run_sequential_optimizations(self, initial_states, inputs,
                                      cond_ids=None,
                                      q_prior=None):
        '''Finds fixed points sequentially, running an optimization from one
        initial state at a time.
        Args:
            initial_states: Either an [n x n_states] numpy array or an
            LSTMStateTuple with initial_states.c and initial_states.h as
            [n_inits x n_states] numpy arrays. These data specify the initial
            states of the RNN, from which the optimization will search for
            fixed points. The choice of type must be consistent with state
            type of rnn_cell.
            inputs: An [n x n_inputs] numpy array specifying a set of constant
            inputs into the RNN.
            q_prior (optional): An [n,] numpy array containing q values from a
            previous optimization round. Provide these if performing
            additional optimization iterations on a subset of outlier
            candidate fixed points. Default: None.
        Returns:
            fps: A FixedPoints object containing the optimized fixed points
            and associated metadata.
        '''
        is_fresh_start = q_prior is None

        if is_fresh_start:
            self._print_if_verbose('\tFinding fixed points via '
                                   'sequential optimizations...')

        x, u = self._build_state_vars(initial_states, inputs)

        n_inits, n_states = x.shape
        n_inputs = u.shape[1]

        # Allocate memory for storing results
        fps = FixedPoints(do_alloc_nan=True,
                          n=n_inits,
                          n_states=n_states,
                          n_inputs=n_inputs,
                          dtype=self.dtype)

        for init_idx in range(n_inits):

            initial_states_i = x[init_idx:(init_idx+1), :] # preserve 2-d
            inputs_i = inputs[init_idx:(init_idx+1), :]

            if cond_ids is None:
                colors_i = None
            else:
                colors_i = cond_ids[init_idx:(init_idx+1)]

            if is_fresh_start:
                self._print_if_verbose(f'\n\tInitialization {init_idx+1} of {n_inits}:')
            else:
                self._print_if_verbose(f'\n\tOutlier {init_idx+1} of {n_inits} ' + \
                    f'(q={q_prior[init_idx]:.2e}):')

            fps[init_idx] = self._run_single_optimization(
                initial_states_i, inputs_i, cond_id=colors_i)

        return fps

    def _run_single_optimization(self, x, u, cond_id=None):
        '''Finds a single fixed point from a single initial state.
        Args:
            initial_state: A [1 x n_states] numpy array or an
            LSTMStateTuple with initial_state.c and initial_state.h as
            [1 x n_states/2] numpy arrays. These data specify an initial
            state of the RNN, from which the optimization will search for
            a single fixed point. The choice of type must be consistent with
            state type of rnn_cell.
            inputs: A [1 x n_inputs] numpy array specifying the inputs to the
            RNN for this fixed point optimization.
        Returns:
            A FixedPoints object containing the optimized fixed point and
            associated metadata.
        '''

        xstar, F_xstar, qstar, dq, n_iters = \
            self._run_optimization_loop(x, u)

        fp = FixedPoints(
            xstar=xstar,
            x_init=x.cpu().detach().numpy(),
            inputs=u.cpu().detach().numpy(),
            cond_id=cond_id,
            F_xstar=F_xstar,
            qstar=qstar,
            dq=dq,
            n_iters=n_iters,
            tol_unique=self.tol_unique,
            dtype=self.dtype)

        return fp

    def _run_optimization_loop(self, x, u):

        def print_update(iter_count, q, dq, lr, is_final=False):

            t = time.time()
            t_elapsed = t - t_start
            avg_iter_time = t_elapsed / iter_count

            if is_final:
                delimiter = "\n\t\t"
                print(f'\t\t{iter_count} iters{delimiter}', end='')
            else:
                delimiter = ', '
                print(f'\tIter: {iter_count}{delimiter}', end='')

            if q.numel() == 1:
                print(f'q = {q.item():.2e}{delimiter}dq = {dq.item():.2e}{delimiter}', end='')
            else:
                mean_q = torch.mean(q).item()
                std_q = torch.std(q).item()

                mean_dq = torch.mean(dq).item()
                std_dq = torch.std(dq).item()

                print(f'q = {mean_q:.2e} +/- {std_q:.2e}{delimiter}' +
                      f'dq = {mean_dq:.2e} +/- {std_dq:.2e}{delimiter}',
                      end='')

            print(f'learning rate = {lr:.2e}{delimiter}', end='')

            print(f'avg iter time = {avg_iter_time:.2e} sec', end='')

            if is_final:
                print('') # Just for the endline
            else:
                print('.')

        x.requires_grad_(True)
        opt = torch.optim.Adam([x], lr=1e-2, **self.adam_optimizer_hps)
        opt.zero_grad()
        lr_scheduler = AdaptiveLearningRate(opt,
            **self.adaptive_learning_rate_hps)
        grad_norm_clip = AdaptiveGradNormClip(
            **self.grad_norm_clip_hps)
        
        iter_count = 1
        t_start = time.time()
        
        q_prev = torch.full((x.shape[0],), np.nan, device=x.device)
        dq = None

        while True:
            x_in = self._split_state(x)
            F = self.rnn_cell(u, x_in)
            F = self._concat_state(F)

            q = 0.5 * torch.sum(torch.square(F - x), axis=1)
            dq = torch.abs(q - q_prev)

            q_scalar = q.mean()
            q_scalar.backward()

            grad_norm_clip(x)
            opt.step()
            lr_scheduler.step(q_scalar)
            opt.zero_grad()

            if self.super_verbose and \
                np.mod(iter_count, self.n_iters_per_print_update)==0:
                print_update(iter_count, q, dq, lr_scheduler._last_lr[0])

            if iter_count > 1 and \
                torch.all(torch.logical_or(
                    dq < self.tol_dq*lr_scheduler._last_lr[0],
                    q < self.tol_q)).item():
                '''Here dq is scaled by the learning rate. Otherwise very
                small steps due to very small learning rates would spuriously
                indicate convergence. This scaling is roughly equivalent to
                measuring the gradient norm.'''
                self._print_if_verbose('\tOptimization complete '
                                       'to desired tolerance.')
                break
        
            if iter_count + 1 > self.max_iters:
                self._print_if_verbose('\tMaximum iteration count reached. '
                                       'Terminating.')
                break

            q_prev = q
            iter_count += 1

        if self.verbose:
            print_update(iter_count,
                         q, dq,
                         lr_scheduler._last_lr[0],
                         is_final=True)

        iter_count = np.tile(iter_count, q.shape)

        x, F, q, dq = to_numpy(x, F, q, dq)
        return x, F, q, dq, iter_count

    def _compute_jacobians(self, fps):

        # Compute derivatives at the fixed points
        x, u = self._build_state_vars(fps.xstar, fps.inputs)

        def J_func(u, x):
            # Adjust dimensions and pass through RNN
            u = u[None, :]
            x = x[None, :]
            F = self.rnn_cell(u, x)
            return F.squeeze()

        # Compute the Jacobian for each fixed point
        all_J_rec = []
        all_J_inp = []
        for i in range(fps.n):
            single_x = x[i, :]
            single_u = u[i, :]
            # Simultaneously compute input and recurrent Jacobians
            J_inp, J_rec = torch.autograd.functional.jacobian(
                J_func, (single_u, single_x))
            all_J_rec.append(J_rec)
            all_J_inp.append(J_inp)

        # Recombine Jacobians for the whole batch
        J_rec = torch.stack(all_J_rec).cpu().detach().numpy()
        J_inp = torch.stack(all_J_inp).cpu().detach().numpy()

        return J_rec, J_inp

    # @staticmethod
    def _get_valid_mask(self, n_batch, n_time, valid_bxt=None):
        ''' Returns an appropriately sized boolean mask.
        Args:
            (n_batch, n_time) is the shape of the desired mask.
            valid_bxt: (optional) proposed boolean mask.
        Returns:
            A shape (n_batch, n_time) boolean mask.
        Raises:
            AssertionError if valid_bxt does not have shape (n_batch, n_time)
        '''
        if valid_bxt is None:
            valid_bxt = torch.ones((n_batch, n_time)).to(torch.bool).to(self.device)
        else:
            assert (valid_bxt.shape[0] == n_batch and
                valid_bxt.shape[1] == n_time),\
                (f'valid_bxt.shape should be {(n_batch, n_time)}, but is {valid_bxt.shape}')

            if isinstance(valid_bxt, np.ndarray):
                valid_bxt, = to_torch(valid_bxt, device=self.device)
            if (valid_bxt.dtype != torch.bool):
                valid_bxt = valid_bxt.to(bool)

        return valid_bxt

    def _add_gaussian_noise(self, data, noise_scale=0.0):
        ''' Adds IID Gaussian noise to Numpy data.
        Args:
            data: Numpy array.
            noise_scale: (Optional) non-negative scalar indicating the
            standard deviation of the Gaussian noise samples to be generated.
            Default: 0.0.
        Returns:
            Numpy array with shape matching that of data.
        Raises:
            ValueError if noise_scale is negative.
        '''

        # Add IID Gaussian noise
        if noise_scale == 0.0:
            return data # no noise to add
        if noise_scale > 0.0:
            return data + noise_scale * self.rng.randn(*data.shape)
        elif noise_scale < 0.0:
            raise ValueError('noise_scale must be non-negative,'
                             ' but was %f' % noise_scale)

    @staticmethod
    def identify_q_outliers(fps, q_thresh):
        '''Identify fixed points with optimized q values that exceed a
        specified threshold.
        Args:
            fps: A FixedPoints object containing optimized fixed points and
            associated metadata.
            q_thresh: A scalar float indicating the threshold on fixed
            points' q values.
        Returns:
            A numpy array containing the indices into fps corresponding to
            the fixed points with q values exceeding the threshold.
        Usage:
            idx = identify_q_outliers(fps, q_thresh)
            outlier_fps = fps[idx]
        '''
        return np.where(fps.qstar > q_thresh)[0]

    @staticmethod
    def identify_q_non_outliers(fps, q_thresh):
        '''Identify fixed points with optimized q values that do not exceed a
        specified threshold.
        Args:
            fps: A FixedPoints object containing optimized fixed points and
            associated metadata.
            q_thresh: A scalar float indicating the threshold on fixed points'
            q values.
        Returns:
            A numpy array containing the indices into fps corresponding to the
            fixed points with q values that do not exceed the threshold.
        Usage:
            idx = identify_q_non_outliers(fps, q_thresh)
            non_outlier_fps = fps[idx]
        '''
        return np.where(fps.qstar <= q_thresh)[0]
    
    @staticmethod
    def identify_distance_non_outliers(fps, initial_states, dist_thresh):
        ''' Identify fixed points that are "far" from the initial states used
        to seed the fixed point optimization. Here, "far" means a normalized
        Euclidean distance from the centroid of the initial states that
        exceeds a specified threshold. Distances are normalized by the average
        distances between the initial states and their centroid.
        Empirically this works, but not perfectly. Future work: replace
        [distance to centroid of initial states] with [nearest neighbors
        distance to initial states or to other fixed points].
        Args:
            fps: A FixedPoints object containing optimized fixed points and
            associated metadata.
            initial_states: Either an [n x n_states] numpy array or an
            LSTMStateTuple with initial_states.c and initial_states.h as
            [n_inits x n_states] numpy arrays. These data specify the initial
            states of the RNN, from which the optimization will search for
            fixed points. The choice of type must be consistent with state
            type of rnn_cell.
            dist_thresh: A scalar float indicating the threshold of fixed
            points' normalized distance from the centroid of the
            initial_states. Fixed points with normalized distances greater
            than this value are deemed putative outliers.
        Returns:
            A numpy array containing the indices into fps corresponding to the
            non-outlier fixed points.
        '''

        initial_states, = to_numpy(initial_states)

        if isinstance(initial_states, (tuple, list)):
            initial_states = np.concatenate(initial_states, axis=1)

        n_inits = initial_states.shape[0]
        n_fps = fps.n

        # Centroid of initial_states, shape (n_states,)
        centroid = np.mean(initial_states, axis=0)

        # Distance of each initial state from the centroid, shape (n,)
        init_dists = np.linalg.norm(initial_states - centroid, axis=1)
        avg_init_dist = np.mean(init_dists)

        # Normalized distances of initial states to the centroid, shape: (n,)
        scaled_init_dists = np.true_divide(init_dists, avg_init_dist)

        # Distance of each FP from the initial_states centroid
        fps_dists = np.linalg.norm(fps.xstar - centroid, axis=1)

        # Normalized
        scaled_fps_dists = np.true_divide(fps_dists, avg_init_dist)

        init_non_outlier_idx = np.where(scaled_init_dists < dist_thresh)[0]
        n_init_non_outliers = init_non_outlier_idx.size
        print(f'\t\tinitial_states: {n_inits - n_init_non_outliers} outliers detected (of {n_inits}).')

        fps_non_outlier_idx = np.where(scaled_fps_dists < dist_thresh)[0]
        n_fps_non_outliers = fps_non_outlier_idx.size
        print(f'\t\tfixed points: {n_fps - n_fps_non_outliers} outliers detected (of {n_fps}).')

        return fps_non_outlier_idx
    
    def _exclude_distance_outliers(self, fps, initial_states):
        ''' Removes putative distance outliers from a set of fixed points.
        See docstring for identify_distance_non_outliers(...).
        '''
        idx_keep = self.identify_distance_non_outliers(
            fps,
            initial_states,
            self.outlier_distance_scale)
        return fps[idx_keep]

    def _print_if_verbose(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)


    def _run_additional_iterations_on_outliers(self, fps):
        '''Detects outlier states with respect to the q function and runs
        additional optimization iterations on those states This should only be
        used after calling either _run_joint_optimization or
        _run_sequential_optimizations.
        Args:
            A FixedPoints object containing (partially) optimized fixed points
            and associated metadata.
        Returns:
            A FixedPoints object containing the further-optimized fixed points
            and associated metadata.
        '''

        '''
        Known issue:
            Additional iterations do not always reduce q! This may have to do
            with learning rate schedules restarting from values that are too
            large.
        '''

        def perform_outlier_optimization(fps, method):

            idx_outliers = self.identify_q_outliers(fps, outlier_min_q)
            n_outliers = len(idx_outliers)

            outlier_fps = fps[idx_outliers]
            n_prev_iters = outlier_fps.n_iters
            initial_states, inputs = self._build_state_vars(outlier_fps.xstar, outlier_fps.inputs)
            cond_ids = outlier_fps.cond_id

            if method == 'joint':

                self._print_if_verbose('\tPerforming another round of '
                                       'joint optimization, '
                                       'over outlier states only.')

                updated_outlier_fps = self._run_joint_optimization(
                    initial_states, inputs,
                    cond_ids=cond_ids)

            elif method == 'sequential':

                self._print_if_verbose('\tPerforming a round of sequential '
                                       'optimizations, over outlier '
                                       'states only.')

                updated_outlier_fps = self._run_sequential_optimizations(
                    initial_states, inputs,
                    cond_ids=cond_ids,
                    q_prior=outlier_fps.qstar)

            else:
                raise ValueError(f'Unsupported method: {method}.')

            updated_outlier_fps.n_iters += n_prev_iters
            fps[idx_outliers] = updated_outlier_fps

            return fps

        def outlier_update(fps):

            idx_outliers = self.identify_q_outliers(fps, outlier_min_q)
            n_outliers = len(idx_outliers)

            self._print_if_verbose(f'\n\tDetected {n_outliers} putative outliers '
                                   f'(q>{outlier_min_q:.2e}).')

            return idx_outliers

        outlier_min_q = np.median(fps.qstar)*self.outlier_q_scale
        idx_outliers = outlier_update(fps)

        if len(idx_outliers) == 0:
            return fps

        '''
        Experimental: Additional rounds of joint optimization. This code
        currently runs, but does not appear to be very helpful in eliminating
        outliers.
        '''
        if self.method == 'joint':
            N_ROUNDS = 0 # consider making this a hyperparameter
            for round in range(N_ROUNDS):

                fps = perform_outlier_optimization(fps, 'joint')

                idx_outliers = outlier_update(fps)
                if len(idx_outliers) == 0:
                    return fps

        # Always perform a round of sequential optimizations on any (remaining)
        # "outliers".
        fps = perform_outlier_optimization(fps, 'sequential')
        outlier_update(fps) # For print output only

        return fps