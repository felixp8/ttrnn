import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import gym
import neurogym as ngym
import copy
import numpy as np

from typing import Optional, List, Tuple, Any

from .. import models
from ..models.actorcritic import ActorCritic, LinearCategoricalActor, \
    LinearGaussianActor, MLPCategoricalActor, MLPGaussianActor
from ..tasks.wrappers import ParallelEnvs


class PartialMLPEncoder(nn.Module):
    def __init__(
        self,
        obs_dim,
        act_dim,
        reward_dim,
        hidden_dim,
        output_dim,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.reward_dim = reward_dim

    def forward(self, X):
        assert X.shape[-1] == (self.obs_dim + self.act_dim + self.reward_dim)
        obs = X[..., :self.obs_dim]
        enc = self.mlp(obs)
        out = torch.cat([enc, X[..., self.obs_dim:]], dim=-1)
        return out


class A2C(pl.LightningModule):
    """adapted from lightning_bolts"""
    def __init__(
        self, 
        env,
        env_kwargs={},
        rnn_type='RNN', 
        rnn_params={}, 
        actor_type='linear',
        critic_type='linear',
        encoder_type='none',
        optim_type='SGD', 
        optim_params={}, 
        epoch_len: int = 100,
        discount_gamma: float = 0.99,
        gae_lambda: float = 1.0,
        avg_reward_len: int = 100,
        entropy_beta: float = 0.01,
        critic_beta: float = 0.25,
        reset_state_per_episode: bool = True,
        trials_per_episode: int = -1,
        max_episode_len: int = -1,
        entropy_anneal_len: int = -1,
        **kwargs: Any,
    ):
        super(A2C, self).__init__()
        if isinstance(env, gym.Env):
            self.env = copy.deepcopy(env)
        else:
            assert isinstance(env, str), 'env must be gym.Env or str'
            self.env = gym.make(env, **env_kwargs)
        self.trial_env = isinstance(self.env.unwrapped, ngym.TrialEnv)
        if hasattr(self.env, 'num_envs'):
            self.num_envs = self.env.num_envs
        else:
            self.env = ParallelEnvs(self.env, num_envs=1) # TODO: use Gym VectorEnvs
            self.num_envs = self.env.num_envs 
        self.save_hyperparameters()
        self.build_model()

        # Hyperparameters
        # self.batches_per_epoch = batch_size * epoch_len
        self.entropy_weight = 1.0

        # Tracking metrics
        self.total_rewards = [[0]] * self.num_envs
        self.episode_reward = np.zeros(self.num_envs, dtype=float)
        self.done_episodes = np.zeros(self.num_envs, dtype=int)
        self.avg_rewards = np.zeros(self.num_envs, dtype=float)
        self.avg_reward_len = avg_reward_len
        self.episode_trial_count = np.zeros(self.num_envs, dtype=int)
        self.episode_len = np.zeros(self.num_envs, dtype=int)
        self.avg_episode_len = np.zeros(self.num_envs, dtype=float)
        if trials_per_episode > 1:
            self.episode_trial_rewards = np.zeros((self.num_envs, trials_per_episode))
            self.avg_episode_trial_rewards = np.zeros((self.num_envs, trials_per_episode))
        else:
            self.episode_trial_rewards = None
            self.avg_episode_trial_rewards = None
        # self.eps = np.finfo(np.float32).eps.item()
        # self.batch_states: List = []
        # self.batch_values: List = []
        # self.batch_actions: List = []
        # self.batch_action_probs: List = []
        # self.batch_rewards: List = []
        # self.batch_masks: List = []

        self.episode_states = []

        self.obs, _ = self.env.reset()
        # if len(self.obs.shape) == 1:
        #     self.obs = self.obs[None, :]
        self.state = self.model.rnn.build_initial_state(self.num_envs, self.device, self.dtype)

    def build_model(self):
        input_size = self.env.observation_space.shape[0]

        encoder_type = self.hparams.get('encoder_type', 'none')
        if encoder_type.lower() == 'none': # TODO: add more flexibility
            encoder = None
            rnn_input_size = input_size
        elif encoder_type.lower() == 'linear':
            encoder = nn.Linear(input_size, 64)
            rnn_input_size = 64
        elif encoder_type.lower() == 'mlp':
            encoder = PartialMLPEncoder(
                obs_dim=input_size - self.env.action_space.n - 1,
                act_dim=self.env.action_space.n,
                reward_dim=1,
                hidden_dim=64,
                output_dim=128
            )
            rnn_input_size = 128 + 1 + self.env.action_space.n
        rnn_type = self.hparams.get('rnn_type', 'RNN')
        rnn_params = self.hparams.get('rnn_params', {})
        if rnn_params['input_size'] != rnn_input_size:
            print("Warning: overriding RNN input size")
            rnn_params['input_size'] = rnn_input_size
        if hasattr(models.rnn, rnn_type):
            rnn = getattr(models.rnn, rnn_type)(**rnn_params)
        else:
            raise ValueError()
        rnn_out_size = rnn_params.get('hidden_size') if \
            rnn_params.get('output_size', None) is None else \
            rnn_params.get('output_size')
        
        actor_type = self.hparams.get('actor_type', 'linear')
        action_space = self.env.action_space
        if isinstance(action_space, gym.spaces.Discrete):
            actor_init = LinearCategoricalActor if actor_type == 'linear' \
                else MLPCategoricalActor
        elif isinstance(action_space, gym.spaces.Box):
            actor_init = LinearGaussianActor if actor_type == 'linear' \
                else MLPGaussianActor
        else:
            raise ValueError()
        actor = actor_init(
            obs_dim=rnn_out_size,
            act_dim=self.env.action_space.n,
        )

        critic_type = self.hparams.get('critic_type', 'linear')
        if critic_type == 'linear':
            critic = nn.Linear(rnn_out_size, 1)

        self.model = ActorCritic(
            rnn=rnn,
            actor=actor,
            critic=critic,
            encoder=encoder,
        )

    def forward(self, X: torch.Tensor, hx: Optional[torch.Tensor] = None, cached: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Passes in a state x through the network and gets the log prob of each action and the value for the state
        as an output.
        Args:
            x: environment state
        Returns:
            action log probabilities, values
        """
        action_logits, values, hx = self.model(X, hx, cached=cached)
        return action_logits, values, hx
    
    def agent_observe(self, cached: bool = False):
        action_logits, value, hx = self.forward(
            torch.from_numpy(self.obs).to(device=self.device, dtype=self.dtype), 
            hx=self.state, 
            cached=cached)
        return action_logits, value, hx

    def agent_act(self, action_logits):
        action = action_logits.sample()
        logprob = action_logits.log_prob(action)
        action = action.detach().cpu().numpy()

        next_obs, reward, done, trunc, info = self.env.step(action)
        # if len(next_obs.shape) == 1:
        #     next_obs = next_obs[None, :]

        return action, logprob, next_obs, reward, done, trunc, info

    def agent_step(self, cached: bool = False):
        assert self.state is not None, \
            "Please set RNN initial state before calling `agent_step()`"
        
        action_logits, value, hx = self.agent_observe(cached=cached)
        action, logprob, next_obs, reward, done, trunc, info = self.agent_act(action_logits)
        entropy = action_logits.entropy()

        self.state = hx # TODO: keep this here?

        return next_obs, action, reward, value, logprob, entropy, done, trunc, info

    def compute_returns(
        self,
        rewards: List[float],
        values: List[torch.Tensor],
        dones: List[bool],
        last_value: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the discounted rewards of the batched rewards.
        Args:
            rewards: list of rewards
            dones: list of done masks
            last_value: the predicted value for the last state (for bootstrap)
        Returns:
            tensor of discounted rewards
        """
        advantages = torch.zeros((self.num_envs, len(rewards)), dtype=torch.float, device=self.device)
        returns = torch.zeros((self.num_envs, len(rewards)), dtype=torch.float, device=self.device)
        
        with torch.no_grad():
            tfn = lambda x: torch.as_tensor(x).to(self.device).to(self.dtype)
            advantages[:, -1] = tfn(rewards[-1]) + self.hparams.discount_gamma * last_value.squeeze() * (1 - tfn(dones[-1])) - values[-1].squeeze()
            returns[:, -1] = tfn(rewards[-1]) + self.hparams.discount_gamma * last_value.squeeze() * (1 - tfn(dones[-1]))
            for t in reversed(range(len(rewards) - 1)):
                delta = tfn(rewards[t]) + self.hparams.discount_gamma * values[t + 1].squeeze() * (1 - tfn(dones[t])) - values[t].squeeze()
                advantages[:, t] = delta + self.hparams.discount_gamma * self.hparams.gae_lambda * (1 - tfn(dones[t])) * advantages[:, t + 1]
                # returns[t] = rewards[t] + self.hparams.discount_gamma * returns[t + 1]
                returns[:, t] = advantages[:, t] + values[t].squeeze()

        return advantages, returns

    def inner_loop(self):
        """Contains the logic for generating a new batch of data to be passed to the DataLoader.
        Returns:
            yields a tuple of Lists containing tensors for
            states, actions, and returns of the batch.
        Note:
            This is what's taken by the dataloader:
            states: a list of numpy array
            actions: a list of list of int
            returns: a torch tensor
        """
        self.model.rnn.update_cache()
        
        batch_rewards = []
        batch_action_probs = []
        batch_entropies = []
        batch_values = []
        batch_dones = []
        batch_advantages = []
        batch_returns = []
        batch_states = []
        batch_actions = []

        for step in range(self.hparams.epoch_len):

            out = self.agent_step(cached=True)
            next_obs, action, reward, value, logprob, entropy, done, trunc, info = out

            if self.trial_env:
                self.episode_trial_count += info.get('new_trial', np.zeros(self.num_envs, dtype=bool)).astype(int)
                # truth = np.array([(env.num_tr - 1) for env in self.env.env_list])
                # if not np.all(np.mod(self.episode_trial_count, self.hparams.trials_per_episode) == truth):
                #     import pdb; pdb.set_trace()
            self.episode_len += 1

            trial_count_reached = (self.episode_trial_count == self.hparams.trials_per_episode)
            max_len_reached = (self.episode_len == self.hparams.max_episode_len)
            episode_done = np.any([done, trunc, trial_count_reached, max_len_reached], axis=0)

            batch_rewards.append(reward)
            batch_action_probs.append(logprob)
            batch_values.append(value)
            batch_entropies.append(entropy)
            batch_dones.append(episode_done)
            batch_actions.append(action) # remove
            batch_states.append(self.obs)

            # self.episode_states.append(self.obs[0])

            self.obs = next_obs.copy()
            
            self.episode_reward += reward

            for i in range(self.num_envs):
                if self.trial_env and info.get('new_trial', np.zeros(self.num_envs, dtype=bool))[i]:
                    if self.episode_trial_rewards is not None:
                        self.episode_trial_rewards[i, self.episode_trial_count[i]-1] = \
                            self.episode_reward[i] - np.sum(self.episode_trial_rewards[i, :self.episode_trial_count[i]-1])

                if episode_done[i]:
                    self.done_episodes[i] += 1
                    self.total_rewards[i].append(self.episode_reward[i])
                    self.avg_rewards[i] = float(np.mean(self.total_rewards[i][-self.avg_reward_len :]))

                    self.avg_episode_len[i] = self.avg_episode_len[i] * 0.95 + self.episode_len[i] * 0.05

                    self.episode_trial_count[i] = 0
                    self.episode_len[i] = 0
                    self.episode_reward[i] = 0

                    # if i == 0:
                    #     if self.current_epoch > 100:
                    #         import pdb; pdb.set_trace()
                    #     self.episode_states = []

                    if self.episode_trial_rewards is not None:
                        # exponential moving average
                        self.avg_episode_trial_rewards[i] = self.avg_episode_trial_rewards[i] * 0.95 \
                              + 0.05 * self.episode_trial_rewards[i]
                        self.episode_trial_rewards[i, :] = 0.
                    # break

                    # self.obs = self.env.reset()
                    if self.hparams.reset_state_per_episode:
                        if isinstance(self.state, tuple):
                            reset_hstate, reset_cstate = self.model.rnn.build_initial_state(1, self.device, self.dtype)
                            self.state = (
                                torch.cat([self.state[0][:i,:], reset_hstate, self.state[0][i+1:,:]], dim=0), 
                                torch.cat([self.state[1][:i,:], reset_cstate, self.state[1][i+1:,:]], dim=0))
                        else:
                            reset_state = self.model.rnn.build_initial_state(1, self.device, self.dtype)
                            self.state = torch.cat([self.state[:i,:], reset_state, self.state[i+1:,:]], dim=0)

        if isinstance(self.state, tuple):
            self.state = tuple([s.detach() for s in self.state])
        else:
            self.state = self.state.detach() # stop gradients per batch

        with torch.no_grad():
            last_value = self.agent_observe(cached=True)[1].detach()
            last_value = torch.where(
                torch.as_tensor(episode_done, dtype=torch.bool, device=self.device).unsqueeze(1),
                torch.zeros((self.num_envs, 1), device=self.device, dtype=self.dtype),
                last_value)

        batch_advantages, batch_returns = self.compute_returns(
            rewards=batch_rewards, 
            values=batch_values, 
            dones=batch_dones,
            last_value=last_value,
        )

        batch_values = torch.cat(batch_values, dim=1)
        batch_action_probs = torch.stack(batch_action_probs, dim=1)
        batch_entropies = torch.stack(batch_entropies, dim=1)

        # import pdb; pdb.set_trace()

        if self.current_epoch % 250 == 0:
            # import pdb; pdb.set_trace()
            if self.avg_episode_trial_rewards is not None:
                print(self.avg_episode_trial_rewards)
            # import pdb; pdb.set_trace()
        
        return batch_advantages, batch_action_probs, batch_values, batch_returns, batch_entropies

    def loss(
        self,
        advantages: torch.Tensor,
        logprobs: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        entropies: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates the loss for A2C which is a weighted sum of actor loss (MSE), critic loss (PG), and entropy
        (for exploration)
        Args:
            states: tensor of shape (batch_size, state dimension)
            actions: tensor of shape (batch_size, )
            returns: tensor of shape (batch_size, )
        """
        # entropy loss
        entropy = self.hparams.entropy_beta * self.entropy_weight * entropies.mean()

        # actor loss
        actor_loss = -(logprobs * advantages).mean()

        # critic loss
        critic_loss = self.hparams.critic_beta * torch.square(returns - values).mean()

        # total loss (weighted sum)
        total_loss = actor_loss + critic_loss - entropy

        # self.log('train/actor_loss', actor_loss.item())
        # self.log('train/critic_loss', critic_loss.item())
        # self.log('train/entropy_loss', entropy.item())
        return total_loss, (actor_loss, critic_loss, entropy)

    def training_step(self, batch, batch_idx):
        """Perform one actor-critic update using a batch of data.
        Args:
            batch: a batch of (states, actions, returns)
        """
        advantages, action_probs, values, returns, entropies = self.inner_loop()
        
        loss, (al, cl, ent) = self.loss(advantages, action_probs, values, returns, entropies)

        for i in range(self.num_envs):
            self.log(f'train/env{i:02d}/episodes', float(self.done_episodes[i]))
            self.log(f'train/env{i:02d}/reward', float(self.total_rewards[i][-1]))
            self.log(f'train/env{i:02d}/avg_reward_100', self.avg_rewards[i])
            self.log(f'train/env{i:02d}/avg_episode_len', float(self.avg_episode_len[i]))
        self.log('train/actor_loss', al.item())
        self.log('train/critic_loss', cl.item())
        self.log('train/entropy', ent.item())
        self.log('train/loss', loss.item())
        return loss
    
    def configure_optimizers(self):
        optim_type = self.hparams.get('optim_type', 'RMSprop')
        optim_params = self.hparams.get('optim_params', {})
        # optimizer = self.get_optimizer(optim_type, optim_params)
        if (optim_type.lower() == 'sgd'):
            optimizer = optim.SGD(self.parameters(), **optim_params)
        elif (optim_type.lower() == 'adam'):
            optimizer = optim.Adam(self.parameters(), **optim_params)
        elif (optim_type.lower() == 'rmsprop'):
            optimizer = optim.RMSprop(self.parameters(), **optim_params)
        elif hasattr(optim, optim_type):
            optimizer = getattr(optim, optim_type)(self.parameters(), **optim_params)
        else:
            raise ValueError()
        return {
            'optimizer': optimizer,
        }
    
    def on_train_batch_end(self, batch_end_outputs, batch, batch_idx):
        if self.hparams.entropy_anneal_len > 0:
            self.entropy_weight = 1 - min(self.current_epoch / self.hparams.entropy_anneal_len, 1.0)

    def train_dataloader(self):
        return DataLoader(dataset=TensorDataset(torch.zeros(1)), batch_size=1)
    
    def validation_step(self, batch, batch_idx):
        return torch.zeros(1, device=self.device)

    def val_dataloader(self):
        return DataLoader(dataset=TensorDataset(torch.zeros(1)), batch_size=1)

    # def _dataloader(self) -> DataLoader:
    #     """Initialize the Replay Buffer dataset used for retrieving experiences."""
    #     dataset = ExperienceSourceDataset(self.train_batch)
    #     dataloader = DataLoader(dataset=dataset, batch_size=self.hparams.batch_size)
    #     return dataloader

    # def train_dataloader(self) -> DataLoader:
    #     """Get train loader."""
    #     return self._dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0][0][0].device.index if self.on_gpu else "cpu"

    # def train_batch(self) -> Iterator[Tuple[np.ndarray, int, torch.Tensor]]:
    #     """Contains the logic for generating a new batch of data to be passed to the DataLoader.
    #     Returns:
    #         yields a tuple of Lists containing tensors for
    #         states, actions, and returns of the batch.
    #     Note:
    #         This is what's taken by the dataloader:
    #         states: a list of numpy array
    #         actions: a list of list of int
    #         returns: a torch tensor
    #     """
    #     trial_count = 0
    #     hx = self.model.rnn.build_initial_state(1, self.device, self.dtype)
    #     for _ in range(self.hparams.max_episode_len):
    #         action_logits, values, hx = self.model(
    #             torch.as_tensor(self.state), hx=hx)
    #         action = action_logits.sample()
    #         action_prob = action_logits.log_prob(action)
    #         action = action.detach().cpu().numpy()

    #         next_state, reward, done, _ = self.env.step(action)

    #         self.batch_rewards.append(reward)
    #         self.batch_actions.append(action)
    #         self.batch_action_probs.append(action_prob)
    #         self.batch_values.append(values)
    #         self.batch_states.append(self.state)
    #         self.batch_masks.append(done)
    #         self.state = next_state
    #         self.episode_reward += reward

    #         if done:
    #             self.done_episodes += 1
    #             self.state = self.env.reset()
    #             self.total_rewards.append(self.episode_reward)
    #             self.episode_reward = 0
    #             self.avg_rewards = float(np.mean(self.total_rewards[-self.avg_reward_len :]))

    #     _, last_value = self.forward(self.state)

    #     returns = self.compute_returns(self.batch_rewards, self.batch_masks, last_value)
    #     for idx in range(self.hparams.batch_size):
    #         yield self.batch_states[idx], self.batch_actions[idx], returns[idx]

    #     self.batch_states = []
    #     self.batch_actions = []
    #     self.batch_rewards = []
    #     self.batch_masks = []

    # def loss(
    #     self,
    #     states: torch.Tensor,
    #     actions: torch.Tensor,
    #     returns: torch.Tensor,
    # ) -> torch.Tensor:
    #     """Calculates the loss for A2C which is a weighted sum of actor loss (MSE), critic loss (PG), and entropy
    #     (for exploration)
    #     Args:
    #         states: tensor of shape (batch_size, state dimension)
    #         actions: tensor of shape (batch_size, )
    #         returns: tensor of shape (batch_size, )
    #     """

    #     logprobs, values = self.net(states)

    #     # calculates (normalized) advantage
    #     with torch.no_grad():
    #         # critic is trained with normalized returns, so we need to scale the values here
    #         advs = returns - values * returns.std() + returns.mean()
    #         # normalize advantages to train actor
    #         advs = (advs - advs.mean()) / (advs.std() + self.eps)
    #         # normalize returns to train critic
    #         targets = (returns - returns.mean()) / (returns.std() + self.eps)

    #     # entropy loss
    #     entropy = -logprobs.exp() * logprobs
    #     entropy = self.hparams.entropy_beta * entropy.sum(1).mean()

    #     # actor loss
    #     logprobs = logprobs[range(self.hparams.batch_size), actions]
    #     actor_loss = -(logprobs * advs).mean()

    #     # critic loss
    #     critic_loss = self.hparams.critic_beta * torch.square(targets - values).mean()

    #     # total loss (weighted sum)
    #     total_loss = actor_loss + critic_loss - entropy
    #     return total_loss

    # def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> OrderedDict:
    #     """Perform one actor-critic update using a batch of data.
    #     Args:
    #         batch: a batch of (states, actions, returns)
    #     """
    #     states, actions, returns = batch
    #     loss = self.loss(states, actions, returns)

    #     log = {
    #         "episodes": self.done_episodes,
    #         "reward": self.total_rewards[-1],
    #         "avg_reward": self.avg_rewards,
    #     }
    #     return {
    #         "loss": loss,
    #         "avg_reward": self.avg_rewards,
    #         "log": log,
    #         "progress_bar": log,
    #     }