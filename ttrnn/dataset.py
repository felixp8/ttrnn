import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, TensorDataset, DataLoader
import gym
import neurogym as ngym
import copy

def to_torch_dtype(np_dtype):
    numpy_to_torch_dtype_dict = {
        np.bool       : torch.bool,
        np.uint8      : torch.uint8,
        np.int8       : torch.int8,
        np.int16      : torch.int16,
        np.int32      : torch.int32,
        np.int64      : torch.int64,
        np.float16    : torch.float16,
        np.float32    : torch.float32,
        np.float64    : torch.float64,
        np.complex64  : torch.complex64,
        np.complex128 : torch.complex128
    }
    return numpy_to_torch_dtype_dict.get(np_dtype)

class NeurogymTaskDataset(Dataset):
    def __init__(self, env, env_kwargs={}, num_trials=400, seq_len=1000, batch_first=False, seed=None):
        if isinstance(env, gym.Env):
            self.env = copy.deepcopy(env)
        else:
            assert isinstance(env, str), 'env must be gym.Env or str'
            self.env = gym.make(env, **env_kwargs)
        self.env.reset()
        self.env.seed(seed)

        self.batch_first = batch_first
        self.num_trials = num_trials
        self.seq_len = seq_len

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
        
        # self._build_dataset()

    def _build_dataset(self, **kwargs):
        env = self.env
        for i in range(self.num_trials):
            seq_start = 0
            seq_end = 0
            while seq_end < self.seq_len:
                env.new_trial(**kwargs)
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
    
    def __iter__(self) -> '_BaseDataLoaderIter':
        if not self.static:
            self.dataset._build_dataset()
        return super(NeurogymDataLoader, self).__iter__()

# class NeurogymMultiTaskDataset(Dataset):
#     def __init__(self):