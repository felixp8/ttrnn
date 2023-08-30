import numpy as np
import copy

import gym
import neurogym as ngym

class DiscreteToBoxWrapper(ngym.TrialWrapper):
    def __init__(self, env, threshold=0.0):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Discrete), \
            "Should only be used to wrap Discrete envs."
        self.n = self.action_space.n
        self.action_space = gym.spaces.Box(0, 1, (self.n,))
        self.threshold = threshold
    
    def set_threshold(self, threshold):
        self.threshold = threshold
    
    def new_trial(self, **kwargs):
        trial = self.env.new_trial(**kwargs)
        old_gt = self.unwrapped.gt
        env_gt = np.zeros((self.unwrapped.gt.shape[0], self.n),
                              dtype=np.float32)
        env_gt[np.arange(env_gt.shape[0]), old_gt] = 1.
        self.gt = env_gt
        return trial

    def step(self, action):
        if isinstance(action, np.ndarray):
            action_sel = np.argmax(action)
            if action_sel != 0 and np.max(action[1:]) < self.threshold:
                action_sel = 0
        else:
            try:
                action_sel = int(round(action))
            except:
                raise ValueError()
        return self.env.step(action_sel)

class DiscreteTo1DBoxWrapper(ngym.TrialWrapper):
    def __init__(self, env, act_map, threshold=0.0):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Discrete), \
            "Should only be used to wrap Discrete envs."
        self.act_map = act_map
        self.values = np.array(list(act_map.values()))
        self.action_space = gym.spaces.Box(np.min(self.values), np.max(self.values), (1,))
        self.threshold = threshold # TODO: use this?
    
    def set_threshold(self, threshold):
        self.threshold = threshold
    
    def new_trial(self, **kwargs):
        trial = self.env.new_trial(**kwargs)
        old_gt = self.unwrapped.gt
        env_gt = np.zeros((self.unwrapped.gt.shape[0], 1),
                              dtype=np.float32)
        env_gt[np.arange(env_gt.shape[0]), old_gt] = 1.
        self.gt = env_gt
        return trial

    def step(self, action):
        if isinstance(action, np.ndarray):
            action_sel = np.argmax(action)
            if action_sel != 0 and np.max(action[1:]) < self.threshold:
                action_sel = 0
        else:
            try:
                action_sel = int(round(action))
            except:
                raise ValueError()
        return self.env.step(action_sel)

class RingToBoxWrapper(ngym.TrialWrapper):
    def __init__(self, env, threshold=0.0):
        super().__init__(env)
        assert hasattr(env.unwrapped, 'theta')
        if hasattr(env.unwrapped, 'delaycomparison'):
            assert env.unwrapped.delaycomparison
        self.dim_ring = len(self.theta)
        self.threshold = threshold
        self._get_obs_shapes()
        self._get_act_shapes()
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (self.obs_dim,)) # TODO: use ngym spaces, add names
        self.action_space = gym.spaces.Box(
            -np.inf, np.inf, (self.act_dim,))
    
    def set_threshold(self, threshold):
        self.threshold = threshold
    
    def _get_obs_shapes(self):
        assert isinstance(self.env.observation_space, gym.spaces.Box)
        max_ob_idx = self.env.observation_space.shape[0]
        obs_names = list(self.unwrapped.observation_space.name.keys())
        obs_idxs = [(list(v) if isinstance(v, range) else [v]) 
            for v in self.unwrapped.observation_space.name.values()]
        obs_names, obs_idxs, map_obs_idxs, obs_dim = self._make_mapping(
            obs_names, obs_idxs, max_ob_idx, 'stim'
        )
        self.obs_names = obs_names
        self.orig_obs_idxs = obs_idxs
        self.obs_idxs = map_obs_idxs
        self.obs_dim = obs_dim
    
    def _get_act_shapes(self):
        assert isinstance(self.env.action_space, gym.spaces.Discrete)
        max_act_idx = self.env.action_space.n
        act_names = list(self.unwrapped.action_space.name.keys())
        act_idxs = [(list(idx) if isinstance(idx, range) else [idx]) 
            for idx in self.unwrapped.action_space.name.values()]
        act_names, act_idxs, map_act_idxs, act_dim = self._make_mapping(
            act_names, act_idxs, max_act_idx, 'choice'
        )
        self.act_names = act_names
        self.orig_act_idxs = act_idxs
        self.act_idxs = map_act_idxs
        self.act_dim = act_dim
    
    def _make_mapping(self, field_names, field_idxs, max_idx, ring_name='stim'):
        # way overly complex
        order = np.argsort([fi[0] for fi in field_idxs])
        field_names = [field_names[i] for i in order]
        field_idxs = [field_idxs[i] for i in order]
        i = 0
        u_count = 0
        while i < (len(field_idxs)):
            this_end = field_idxs[i][-1] + 1
            if (i == len(field_idxs) - 1):
                next_start = max_idx
            else:
                next_start = field_idxs[i + 1][0]
            gap = next_start - this_end
            if gap == 0:
                pass
            elif gap % self.dim_ring == 0:
                gap_size = gap // self.dim_ring
                print(f"WARNING: assuming {gap_size} empty ring stimulus blocks spanning " + 
                    f"indices {this_end} to {next_start}")
                for j in range(gap_size):
                    field_names.insert(i+1, f'emptystimulus_{i}{j}')
                    field_idxs.insert(i+1, list(
                        range(this_end+j*self.dim_ring, this_end+(j+1)*self.dim_ring)))
                    i += 1
            else:
                print(f"WARNING: found unknown stimulus block spanning " + 
                    f"indices {this_end} to {next_start}")
                field_names.insert(i+1, f'unknown_{u_count}')
                field_idxs.insert(i+1, list(range(this_end, next_start)))
                u_count += 1
                i += 1
            i += 1
        map_field_idxs = []
        out_dim = 0
        for name, idx in zip(field_names, field_idxs):
            if 'fixation' in name:
                map_field_idxs.append([out_dim])
                out_dim += 1
            elif (len(idx) == self.dim_ring) and (ring_name in name):
                map_field_idxs.append([out_dim, out_dim + 1])
                out_dim += 2
            else:
                map_field_idxs.append(list(range(out_dim, out_dim + len(idx))))
                out_dim += len(idx)
        return field_names, field_idxs, map_field_idxs, out_dim
    
    def new_trial(self, **kwargs):
        trial = self.env.new_trial(**kwargs)
        tlen = self.unwrapped.ob.shape[0]
        assert (len(self.unwrapped.gt.shape) == 1)
        periods = list(self.unwrapped.start_ind.keys())

        ob = np.zeros((tlen, self.obs_dim))
        gt = np.zeros((tlen, self.act_dim))

        for period in periods:
            period_ob = self.unwrapped.view_ob(period)
            period_gt = self.unwrapped.view_groundtruth(period)
            period_slice = slice(self.unwrapped.start_ind[period], self.unwrapped.end_ind[period])

            for obs_group, orig_obs_idx, obs_idx in zip(self.obs_names, self.orig_obs_idxs, self.obs_idxs):
                if obs_group == 'fixation':
                    ob_fixation = period_ob[:, orig_obs_idx]
                    if np.mean(ob_fixation) > 0.2: # in case of noise
                        ob[period_slice, obs_idx] = 1.
                    else:
                        ob[period_slice, obs_idx] = 0.
                elif (len(orig_obs_idx) == self.dim_ring) and ('stim' in obs_group):
                    assert len(obs_idx) == 2
                    ob_stim = period_ob[:, orig_obs_idx]
                    if np.all(ob_stim == 0): # all zeros
                        ob[period_slice, obs_idx] = 0.
                        continue
                    stim_vals = ob_stim.argmax(axis=1)
                    vals, counts = np.unique(stim_vals, return_counts=True)
                    stim_idx = vals[np.argmax(counts)]
                    stim_theta = self.theta[stim_idx]
                    stim_cos = np.round(np.cos(stim_theta), 8)
                    stim_sin = np.round(np.sin(stim_theta), 8)
                    if ('coh1_mod1' in trial):
                        if ('mod1' in obs_group) and ('stim1' in period):
                            coh = trial['coh1_mod1']
                        if ('mod1' in obs_group) and ('stim2' in period):
                            coh = trial['coh2_mod1']
                        if ('mod2' in obs_group) and ('stim1' in period):
                            coh = trial['coh1_mod2']
                        if ('mod2' in obs_group) and ('stim2' in period):
                            coh = trial['coh2_mod2']
                        if hasattr(self.unwrapped, 'cohs'):
                            try:
                                scaled_coh = coh / np.max(self.unwrapped.cohs)
                                coh = scaled_coh
                            except:
                                pass
                        stim_cos *= coh
                        stim_sin *= coh
                    ob[period_slice, obs_idx[0]] = stim_cos
                    ob[period_slice, obs_idx[1]] = stim_sin
                else:
                    assert len(orig_obs_idx) == len(obs_idx)
                    ob[period_slice, obs_idx] = period_ob[:, orig_obs_idx]

            # assume constant output within period
            gt_val = period_gt[0]
            for act_group, orig_act_idx, act_idx in zip(self.act_names, self.orig_act_idxs, self.act_idxs):
                if not (gt_val in orig_act_idx):
                    continue
                if act_group == 'fixation':
                    gt[period_slice, act_idx] = 1.
                elif (len(orig_act_idx) == self.dim_ring) and ('choice' in act_group):
                    stim_idx = gt_val - orig_act_idx[0]
                    stim_theta = self.theta[stim_idx]
                    stim_cos = np.round(np.cos(stim_theta), 8)
                    stim_sin = np.round(np.sin(stim_theta), 8)
                    gt[period_slice, act_idx[0]] = stim_cos
                    gt[period_slice, act_idx[1]] = stim_sin
                else:
                    assert len(orig_act_idx) == len(act_idx)
                    idx = np.nonzero(np.array(orig_act_idx) == gt_val)[0].item(0)
                    gt[period_slice, act_idx[idx]] = 1.

        self.ob = ob
        self.gt = gt
        return trial
    
    def step(self, action):
        # Action is assumed to be [fixation, cos(theta), sin(theta)]
        # import pdb; pdb.set_trace()
        if isinstance(action, np.ndarray):
            if action.size == self.act_dim:
                assert self.act_dim == 3 # don't know how to handle anything else rn
                theta = np.arctan2(action[2], action[1])
                mag = np.sqrt(action[1] ** 2 + action[2] ** 2)
                if mag > action[0] and mag >= self.threshold: # fixation broken
                    dists = np.concatenate([theta - self.unwrapped.theta, theta + 2 * np.pi - self.unwrapped.theta])
                    match_idx = (np.argmin(np.abs(dists)) % self.dim_ring) + 1
                else:
                    match_idx = 0
            else:
                raise ValueError
        elif isinstance(action, (int, float, np.int32, np.int64)):
            match_idx = int(round(action))
        else:
            raise ValueError
        
        return self.env.step(match_idx)

class LossMaskWrapper(gym.Wrapper):
    def __init__(self, env, mask_config):
        super().__init__(env)
        self.mask_mode = mask_config.get('mode', 'firstlast')
        assert self.mask_mode in ['firstlast', 'last', 'timing']
        if self.mask_mode == 'timing':
            self.mask_timing = mask_config.get('mask_timing', {})
        else:
            self.mask_timing = {}
    
    def new_trial(self, **kwargs):
        trial = self.env.new_trial(**kwargs)
        mask = np.zeros_like(self.env.gt)
        if self.mask_mode == 'firstlast':
            mask[0, :] = 1.
            mask[-1, :] = 1.
        elif self.mask_mode == 'last':
            mask[-1, :] = 1.
        else:
            for period, idxs in self.mask_timing.items():
                if len(idxs) == 0:
                    idxs = np.arange(self.end_ind[period] - self.start_ind[period])
                mask[self.start_ind[period]:self.end_ind[period]][idxs] = 1.
        self.loss_mask = mask
        return trial

class ParallelEnvs(gym.Wrapper): # idk what I'm doing
    def __init__(self, env, num_envs):
        super().__init__(env)
        self.num_envs = num_envs
        self.env_list = [self.env] + [copy.deepcopy(self.env) for _ in range(self.num_envs - 1)]
    
    def sync_envs(self):
        self.env_list = [self.env] + [copy.deepcopy(self.env) for _ in range(self.num_envs - 1)]
    
    def reset(self, **kwargs):
        obs, infos = zip(*[env.reset(**kwargs) for env in self.env_list])
        obs = np.stack(obs)
        infos = self.build_info(infos)
        return obs, infos

    def single_reset(self, idx=0, **kwargs):
        return self.env_list[idx].reset(**kwargs)

    def build_info(self, info_list):
        type_defaults = {
            float: np.nan,
            bool: False,
            int: 0,
        }
        # dtype_dict = {}
        info_stacked = {}
        for i, info in enumerate(info_list):
            for key, val in info.items():
                if key in info_stacked:
                    # assert isinstance(val, dtype_dict[key])
                    pass
                else:
                    # dtype_dict[key] = type(val)
                    info_stacked[key] = np.full(
                        self.num_envs, 
                        type_defaults.get(type(val), 0), 
                        dtype=type(val))
                info_stacked[key][i] = val
        return info_stacked

    def step(self, action):
        assert action.shape[0] == self.num_envs
        obs_list = []
        reward_list = []
        done_list = []
        trunc_list = []
        info_list = []

        for i in range(action.shape[0]):
            ob, reward, done, trunc, info = self.env_list[i].step(action[i])

            obs_list.append(ob)
            reward_list.append(reward)
            done_list.append(done)
            trunc_list.append(trunc)
            info_list.append(info)

        obs = np.stack(obs_list) # N x O
        rewards = np.stack(reward_list) # N
        dones = np.stack(done_list)
        truncs = np.stack(trunc_list)
        infos = self.build_info(info_list)
        return obs, rewards, dones, truncs, infos