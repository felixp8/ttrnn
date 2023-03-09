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
from .a2c import A2C


class MetaA2C(A2C):
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
        super(MetaA2C, self).__init__(
            env=env,
            env_kwargs=env_kwargs,
            rnn_type=rnn_type, 
            rnn_params=rnn_params, 
            actor_type=actor_type,
            critic_type=critic_type,
            optim_type=optim_type, 
            optim_params=optim_params, 
            epoch_len=epoch_len,
            discount_gamma=discount_gamma,
            gae_lambda=gae_lambda,
            avg_reward_len=avg_reward_len,
            entropy_beta=entropy_beta,
            critic_beta=critic_beta,
            reset_state_per_episode=reset_state_per_episode,
            trials_per_episode=trials_per_episode,
            max_episode_len=max_episode_len,
            entropy_anneal_len=entropy_anneal_len,
            **kwargs
        )
        self.last_reward = 0
        self.last_action = 0

    def build_model(self):
        rnn_type = self.hparams.get('rnn_type', 'RNN')
        rnn_params = self.hparams.get('rnn_params', {})
        input_shape = self.env.observation_space.shape[0] + self.env.action_space.n + 1
        assert rnn_params['input_size'] == input_shape
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
    
    def agent_observe(self, cached: bool = False):
        agent_input = torch.cat([
            torch.from_numpy(self.obs).to(self.device).unsqueeze(0),
            torch.eye(self.env.action_space.n)[self.last_action].to(self.device).unsqueeze(0),
            torch.tensor([self.last_reward]).to(self.device).unsqueeze(0),
        ], dim=1)
        action_logits, value, hx = self.forward(
            agent_input, 
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
        self.last_action = int(round(action.item()))
        self.last_reward = reward

        return next_obs, action, reward, value, logprob, entropy, done, info
