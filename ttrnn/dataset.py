import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, TensorDataset, DataLoader
import gym
import neurogym as ngym
import copy

def to_torch_dtype(np_dtype):
    numpy_to_torch_dtype_dict = {
        np.dtype('bool')       : torch.bool,
        np.dtype('uint8')      : torch.uint8,
        np.dtype('int8')       : torch.int8,
        np.dtype('int16')      : torch.int16,
        np.dtype('int32')      : torch.int32,
        np.dtype('int64')      : torch.int64,
        np.dtype('float16')    : torch.float16,
        np.dtype('float32')    : torch.float32,
        np.dtype('float64')    : torch.float64,
        np.dtype('complex64')  : torch.complex64,
        np.dtype('complex128') : torch.complex128
    }
    th_dtype = numpy_to_torch_dtype_dict.get(np.dtype(np_dtype))
    if th_dtype is None: # TODO: use a logger
        print(f"WARNING: No matching PyTorch dtype found for dtype {np.dtype(np_dtype)}")
    return th_dtype

class NeurogymTaskDataset(Dataset):
    def __init__(self, env, env_kwargs={}, wrappers=[], num_trials=400, seq_len=1000, batch_first=False, save_envs=False, seed=None):
        if isinstance(env, gym.Env):
            self.env = copy.deepcopy(env)
        else:
            assert isinstance(env, str), 'env must be gym.Env or str'
            self.env = gym.make(env, **env_kwargs)
        if len(wrappers) > 0:
            for wrapper, wrapper_kwargs in wrappers:
                self.env = wrapper(self.env, **wrapper_kwargs)
        self.env.reset()
        self.env.seed(seed)

        self.batch_first = batch_first
        self.num_trials = num_trials
        self.seq_len = seq_len
        self.save_envs = save_envs

        obs_shape = self.env.observation_space.shape
        action_shape = self.env.action_space.shape
        obs_dtype = to_torch_dtype(self.env.observation_space.dtype)
        act_dtype = to_torch_dtype(self.env.action_space.dtype)
        
        if self.batch_first:
            self.input_shape = num_trials, seq_len, int(np.prod(obs_shape))
            self.target_shape = num_trials, seq_len, int(np.prod(action_shape))
        else:
            self.input_shape = seq_len, num_trials, int(np.prod(obs_shape))
            self.target_shape = seq_len, num_trials, int(np.prod(action_shape))

        if len(action_shape) == 0:
            self._expand_action = True
        else:
            self._expand_action = False
        
        self._inputs = torch.empty(self.input_shape, dtype=obs_dtype)
        self._target = torch.empty(self.target_shape, dtype=act_dtype)
        self.stored_envs = []
        
        self._build_dataset() # to fix: when not static, first build data will never be used

    def _build_dataset(self, **kwargs):
        env = self.env
        for i in range(self.num_trials):
            seq_start = 0
            seq_end = 0
            env_list = []
            while seq_end < self.seq_len:
                env.new_trial(**kwargs)
                if self.save_envs:
                    # ideally would only save env.unwrapped, but
                    # currently with DiscreteToBoxWrapper, need 
                    # overridden gt_now. Potential fix: move
                    # expanding action space to Dataset obj
                    env_list.append(copy.deepcopy(env))
                ob, gt = env.ob, env.gt
                if self._expand_action:
                    gt = gt[:, None]
                seq_len = ob.shape[0]
                seq_end = seq_start + seq_len
                if seq_end > self.seq_len:
                    seq_end = self.seq_len
                    seq_len = seq_end - seq_start
                if self.batch_first:
                    self._inputs[i, seq_start:seq_end, ...] = torch.from_numpy(ob[:seq_len])
                    self._target[i, seq_start:seq_end, ...] = torch.from_numpy(gt[:seq_len])
                else:
                    self._inputs[seq_start:seq_end, i, ...] = torch.from_numpy(ob[:seq_len])
                    self._target[seq_start:seq_end, i, ...] = torch.from_numpy(gt[:seq_len])
                seq_start = seq_end
            if self.save_envs:
                self.stored_envs.append(env_list)

    def __len__(self):
        return self.num_trials
    
    def __getitem__(self, idx):
        if self.batch_first:
            return self._inputs[idx], self._target[idx]
        else:
            return self._inputs[:, idx], self._target[:, idx]
    
class NeurogymDataLoader(DataLoader):
    def __init__(self, dataset, static=False, **kwargs):
        super(NeurogymDataLoader, self).__init__(dataset, **kwargs)
        self.static = static
        self.frozen = False
    
    def __iter__(self):
        if not self.static and not self.frozen:
            self.dataset._build_dataset()
        return super(NeurogymDataLoader, self).__iter__()
    
    def freeze(self):
        self.frozen = True
    
    def unfreeze(self):
        self.frozen = False