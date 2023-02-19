import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import gym
import copy
import numpy as np

from typing import Optional, List, Tuple, Any

from .. import models
from ..models.actorcritic import ActorCritic, LinearCategoricalActor, \
    LinearGaussianActor, MLPCategoricalActor, MLPGaussianActor


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
        optim_type='SGD', 
        optim_params={}, 
        epoch_len: int = 100,
        discount: float = 0.99,
        gae_lambda: float = 1.0,
        avg_reward_len: int = 100,
        entropy_beta: float = 0.01,
        critic_beta: float = 0.25,
        reset_state_per_episode: bool = True,
        trials_per_episode: int = 10,
        **kwargs: Any,
    ):
        super(A2C, self).__init__()
        if isinstance(env, gym.Env):
            self.env = copy.deepcopy(env)
        else:
            assert isinstance(env, str), 'env must be gym.Env or str'
            self.env = gym.make(env, **env_kwargs)
        self.save_hyperparameters()
        self.build_model()

        # Hyperparameters
        # self.batches_per_epoch = batch_size * epoch_len

        # Tracking metrics
        self.total_rewards = [0]
        self.episode_reward = 0
        self.done_episodes = 0
        self.avg_rewards = 0.0
        self.avg_reward_len = avg_reward_len
        self.episode_trial_count = 0
        # self.eps = np.finfo(np.float32).eps.item()
        # self.batch_states: List = []
        # self.batch_values: List = []
        # self.batch_actions: List = []
        # self.batch_action_probs: List = []
        # self.batch_rewards: List = []
        # self.batch_masks: List = []

        self.obs = None
        self.state = None
        self.done = True

    def build_model(self):
        rnn_type = self.hparams.get('rnn_type', 'RNN')
        rnn_params = self.hparams.get('rnn_params', {})
        assert rnn_params['input_size'] == self.env.observation_space.shape[0]
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
            torch.from_numpy(self.obs).to(self.device).unsqueeze(0), 
            hx=self.state, 
            cached=cached)
        return action_logits, value, hx

    def agent_act(self, action_logits):
        action = action_logits.sample()
        logprob = action_logits.log_prob(action)
        action = action.detach().cpu().numpy()
        next_obs, reward, done, info = self.env.step(action)

        return action, logprob, next_obs, reward, done, info

    def agent_step(self, cached: bool = False):
        assert self.state is not None, \
            "Please set RNN initial state before calling `agent_step()`"
        
        action_logits, value, hx = self.agent_observe(cached=cached)
        action, logprob, next_obs, reward, done, info = self.agent_act(action_logits)
        entropy = action_logits.entropy()

        self.state = hx # TODO: keep this here?

        return next_obs, action, reward, value, logprob, entropy, done, info

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
        advantages = torch.zeros(len(rewards), dtype=torch.float, device=self.device)
        returns = torch.zeros(len(rewards), dtype=torch.float, device=self.device)

        advantages[-1] = rewards[-1] + self.hparams.discount * last_value * (1 - dones[-1]) - values[-1].item()
        returns[-1] = rewards[-1] + self.hparams.discount * last_value * (1 - dones[-1])
        for t in reversed(range(len(rewards) - 1)):
            delta = rewards[t] + self.hparams.discount * values[t + 1].item() * (1 - dones[t]) - values[t].item()
            advantages[t] = delta + self.hparams.discount * self.hparams.gae_lambda * (1 - dones[t]) * advantages[t + 1]
            # returns[t] = rewards[t] + self.hparams.discount * returns[t + 1]
            returns[t] = advantages[t] + values[t].item()

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
        if self.done:
            self.done = False
            self.obs = self.env.reset()
            if self.hparams.reset_state_per_episode or self.state is None:
                self.state = self.model.rnn.build_initial_state(1, self.device, self.dtype)
        
        batch_rewards = []
        batch_action_probs = []
        batch_entropies = []
        batch_values = []
        batch_dones = []
        batch_advantages = []
        batch_returns = []
        # batch_states = []
        # batch_actions = []

        for step in range(self.hparams.epoch_len):

            out = self.agent_step(cached=True)
            next_obs, action, reward, value, logprob, entropy, done, info = out

            batch_rewards.append(reward)
            batch_action_probs.append(logprob)
            batch_values.append(value)
            batch_entropies.append(entropy)
            batch_dones.append(done)
            # batch_actions.append(action.detach().cpu().numpy()) # remove
            # batch_states.append(obs)
            
            self.episode_reward += reward
            self.obs = next_obs

            # batch_end = (step == self.hparams.epoch_len - 1)

            if info.get('new_trial', True):
                self.episode_trial_count += 1

            if done or (self.episode_trial_count == self.hparams.trials_per_episode):
                # prematurely stop batch if done reached
                # not really required though
                self.done_episodes += 1
                self.total_rewards.append(self.episode_reward)
                self.avg_rewards = float(np.mean(self.total_rewards[-self.avg_reward_len :]))
                self.done = done

                self.episode_trial_count = 0
                
                self.episode_reward = 0
                break

        if isinstance(self.state, tuple):
            self.state = tuple([s.detach() for s in self.state])
        else:
            self.state = self.state.detach() # stop gradients per batch

        if done:
            last_value = 0
        else:
            with torch.no_grad():
                last_value = self.agent_observe(cached=True)[1].item()
        batch_advantages, batch_returns = self.compute_returns(
            rewards=batch_rewards, 
            values=batch_values, 
            dones=batch_dones,
            last_value=last_value,
        )

        batch_values = torch.cat(batch_values, dim=0).squeeze()
        batch_action_probs = torch.cat(batch_action_probs, dim=0).squeeze()
        batch_entropies = torch.cat(batch_entropies, dim=0).squeeze()

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
        entropy = self.hparams.entropy_beta * entropies.mean()

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

        self.log('train/episodes', float(self.done_episodes))
        self.log('train/reward', float(self.total_rewards[-1]))
        self.log('train/avg_reward_100', self.avg_rewards)
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