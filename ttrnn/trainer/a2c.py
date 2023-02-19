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
        max_batch_len=250,
        max_batch_episodes=1,
        unroll_len=100,
        discount: float = 0.99,
        gae_lambda: float = 1.0,
        avg_reward_len: int = 100,
        entropy_beta: float = 0.01,
        critic_beta: float = 0.25,
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
        # self.episode_reward = 0
        self.done_episodes = 0
        self.avg_rewards = 0.0
        self.avg_reward_len = avg_reward_len
        # self.eps = np.finfo(np.float32).eps.item()
        # self.batch_states: List = []
        # self.batch_values: List = []
        # self.batch_actions: List = []
        # self.batch_action_probs: List = []
        # self.batch_rewards: List = []
        # self.batch_masks: List = []

        # self.state = self.env.reset()

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

    def forward(self, X: torch.Tensor, hx: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Passes in a state x through the network and gets the log prob of each action and the value for the state
        as an output.
        Args:
            x: environment state
        Returns:
            action log probabilities, values
        """
        action_logits, values, hx = self.model(X, hx)
        return action_logits, values, hx

    def compute_returns(
        self,
        rewards: List[float],
        values: List[float],
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
        episode_count = 0
        episode_reward = 0
        episode_start = 0
        hx = self.model.rnn.build_initial_state(1, self.device, self.dtype)
        self.model.rnn.rnn_cell.weights()
        obs = self.env.reset()
        
        batch_rewards = []
        batch_action_probs = []
        batch_entropies = []
        batch_values = []
        batch_dones = []
        batch_advantages = []
        batch_returns = []
        batch_states = []
        batch_actions = []

        for step in range(self.hparams.max_batch_len):
            if (step != episode_start) and ((step - episode_start) % self.hparams.unroll_len == 0):
                hx = hx.clone().detach() # stop gradients ??

            action_logits, values, hx = self.model(
                torch.from_numpy(obs).to(self.device).unsqueeze(0), hx=hx)
            action = action_logits.sample()
            action_prob = action_logits.log_prob(action)
            # action = action.detach().cpu().numpy()

            obs, reward, done, info = self.env.step(action.detach().cpu().numpy())

            trial_end = info.get('new_trial', False)
            done = done or trial_end

            batch_rewards.append(reward)
            batch_action_probs.append(action_prob)
            batch_values.append(values)
            batch_entropies.append(action_logits.entropy())
            batch_actions.append(action.detach().cpu().numpy()) # remove
            batch_states.append(obs) # remove
            batch_dones.append(done)
            episode_reward += reward

            batch_end = (step == self.hparams.max_batch_len - 1)

            if done or batch_end:
                episode_count += 1
                self.done_episodes += 1
                self.total_rewards.append(episode_reward)
                self.avg_rewards = float(np.mean(self.total_rewards[-self.avg_reward_len :]))

                if done:
                    last_value = 0
                else:
                    last_value = self.forward(torch.from_numpy(obs).to(self.device).unsqueeze(0), hx=hx)[1].item()
                advantages, returns = self.compute_returns(
                    rewards=batch_rewards[episode_start:], 
                    values=batch_values[episode_start:], 
                    dones=batch_dones[episode_start:],
                    last_value=last_value,
                )
                batch_advantages.append(advantages)
                batch_returns.append(returns)

                if episode_count >= self.hparams.max_batch_episodes:
                    break
                
                obs = self.env.reset()
                hx = self.model.rnn.build_initial_state(1, self.device, self.dtype)
                episode_reward = 0
                episode_start = step + 1
                # break        

        batch_advantages = torch.cat(batch_advantages, dim=0).squeeze()
        batch_action_probs = torch.cat(batch_action_probs, dim=0).squeeze()
        batch_values = torch.cat(batch_values, dim=0).squeeze()
        batch_returns = torch.cat(batch_returns, dim=0).squeeze()
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

        # self.log('train/actor_loss', (-actor_loss).item())
        # self.log('train/critic_loss', critic_loss.item())
        # self.log('train/entropy_loss', entropy.item())
        return total_loss

    def training_step(self, batch, batch_idx):
        """Perform one actor-critic update using a batch of data.
        Args:
            batch: a batch of (states, actions, returns)
        """
        advantages, action_probs, values, returns, entropies = self.inner_loop()
        
        loss = self.loss(advantages, action_probs, values, returns, entropies)

        self.log('train/episodes', float(self.done_episodes))
        self.log('train/reward', float(self.total_rewards[-1]))
        self.log('train/avg_reward_100', self.avg_rewards)
        self.log('train/loss', loss.item())
        return loss
    
    def configure_optimizers(self):
        optim_type = self.hparams.get('optim_type', 'SGD')
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