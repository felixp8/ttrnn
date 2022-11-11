"""Driscoll tasks."""

import numpy as np
import gym

import neurogym as ngym
from neurogym import spaces
from neurogym.wrappers.block import ScheduleEnvs
from neurogym.wrappers.noise import Noise
from neurogym.utils import scheduler
from neurogym.core import TrialWrapper

from .wrappers import RingToBoxWrapper

def DelayPro(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['anti'] = False
    return _Delay(**env_kwargs)

def DelayAnti(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['anti'] = True
    return _Delay(**env_kwargs)

def MemoryPro(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['anti'] = False
    return _Memory(**env_kwargs)
    
def MemoryAnti(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['anti'] = True
    return _Memory(**env_kwargs)
    
def ReactPro(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['anti'] = False
    return _React(**env_kwargs)

def ReactAnti(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['anti'] = True
    return _React(**env_kwargs)

def IntegrationModality1(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['stim_on'] = (True, False)
    env_kwargs['stim_attend'] = (True, False)
    return _ContextDM(**env_kwargs)

def IntegrationModality2(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['stim_on'] = (False, True)
    env_kwargs['stim_attend'] = (False, True)
    return _ContextDM(**env_kwargs)

def ContextIntModality1(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['stim_on'] = (True, True)
    env_kwargs['stim_attend'] = (True, False)
    return _ContextDM(**env_kwargs)

def ContextIntModality2(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['stim_on'] = (True, True)
    env_kwargs['stim_attend'] = (False, True)
    return _ContextDM(**env_kwargs)

def IntegrationMultimodal(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['stim_on'] = (True, True)
    env_kwargs['stim_attend'] = (False, True)
    return _ContextDM(**env_kwargs)

def ReactMatch2Sample(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['matchto'] = 'sample'
    env_kwargs['matchgo'] = True
    return _DelayMatch(**env_kwargs)
    
def ReactMatch2Sample(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['matchto'] = 'sample'
    env_kwargs['matchgo'] = True
    return _DelayMatch(**env_kwargs)

def ReactNonMatch2Sample(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['matchto'] = 'sample'
    env_kwargs['matchgo'] = False
    return _DelayMatch(**env_kwargs)

def ReactCategoryPro(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['matchto'] = 'category'
    env_kwargs['matchgo'] = True
    return _DelayMatch(**env_kwargs)

def ReactCategoryAnti(**kwargs):
    env_kwargs = kwargs.copy()
    env_kwargs['matchto'] = 'category'
    env_kwargs['matchgo'] = False
    return _DelayMatch(**env_kwargs)

def Driscoll2022():
    env_list = [
        DelayPro(),
        DelayAnti(),
        MemoryPro(),
        MemoryAnti(),
        ReactPro(),
        ReactAnti(),
        IntegrationModality1(),
        IntegrationModality2(),
        ContextIntModality1(),
        ContextIntModality2(),
        IntegrationMultimodal(),
        ReactMatch2Sample(),
        ReactNonMatch2Sample(),
        ReactCategoryPro(),
        ReactCategoryAnti(),
    ]
    # wrapper_list = [
    #     (RingToBoxWrapper, {}),
    # ]
    # for wrapper, wrapper_kwargs in wrapper_list:
    #     env_list = [wrapper(env, **wrapper_kwargs) for env in env_list]
    sched = WeightedRandomSchedule(15, 
        weights=[1., 1., 1., 1., 1., 1., 1., 1., 5., 5., 1., 1., 1., 1., 1.]
    )
    env = ScheduleEnvs(env_list, sched, env_input=True)
    env = RingToBoxWrapper(env)
    env = Noise(env, std_noise=(0.1 * np.sqrt(2 / 0.2)))
    return env

class WeightedRandomSchedule(scheduler.BaseSchedule):
    """Weighted random schedules"""
    def __init__(self, n, weights=None):
        super().__init__(n)
        if weights is not None:
            assert len(weights) == n, "Number of weights must match number " + \
                "of items to schedule."
            self.weights = np.array(weights) / np.sum(weights)
        else:
            self.weights = None

    def __call__(self):
        if self.n > 1:
            js = list(range(self.n)) # allow repeats
            self.i = self.rng.choice(js, p=self.weights)
        else:
            self.i = 0
        self.total_count += 1
        return self.i

class _Delay(ngym.TrialEnv):
    """Delay task.
    """
    metadata = {
        'paper_link': None,
        'paper_name': None,
        'tags': [],
    }

    def __init__(self, dt=20, anti=False, rewards=None, timing=None,
                 dim_ring=10, modality=None):
        super().__init__(dt=dt)

        self.anti = anti
        self.modalities = [0, 1]
        if modality:
            self.modalities = [modality]

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': lambda: self.rng.uniform(300, 700),
            'stimulus': lambda: self.rng.uniform(200, 1500),
            'decision': lambda: self.rng.uniform(300, 700)}
        if timing:
            self.timing.update(timing)

        self.abort = False

        # action and observation spaces
        self.dim_ring = dim_ring
        self.theta = np.arange(0, 2 * np.pi, 2 * np.pi / dim_ring)
        self.choices = np.arange(dim_ring)

        name = {
            'fixation': 0,
            'stimulus_mod1': range(1, dim_ring + 1),
            'stimulus_mod2': range(dim_ring + 1, 2 * dim_ring + 1)}
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(1 + 2 * dim_ring,),
            dtype=np.float32, name=name)

        name = {'fixation': 0, 'choice': range(1, dim_ring + 1)}
        self.action_space = spaces.Discrete(1+dim_ring, name=name)

    def _new_trial(self, **kwargs):
        # Trial info
        trial = {
            'ground_truth': self.rng.choice(self.choices),
            'anti': self.anti,
            'modality': self.rng.choice(self.modalities),
        }
        trial.update(kwargs)

        ground_truth = trial['ground_truth']
        if trial['anti']:
            stim_theta = np.mod(self.theta[ground_truth] + np.pi, 2*np.pi)
        else:
            stim_theta = self.theta[ground_truth]
        stim = np.cos(self.theta - stim_theta) / 2 + 0.5

        periods = ['fixation', 'stimulus', 'decision']
        self.add_period(periods)

        # currently always using modality 1
        self.add_ob(1, period=['fixation', 'stimulus'], where='fixation')
        if trial['modality'] == 0:
            self.add_ob(stim, period=['stimulus', 'decision'], where='stimulus_mod1')
        elif trial['modality'] == 1:
            self.add_ob(stim, period=['stimulus', 'decision'], where='stimulus_mod2')

        self.set_groundtruth(ground_truth, period='decision', where='choice')

        return trial

    def _step(self, action):
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now
        # observations
        if self.in_period('fixation'):
            if action != 0:  # action = 0 means fixating
                new_trial = self.abort
                reward += self.rewards['abort']
        elif self.in_period('decision'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward += self.rewards['correct']
                    self.performance = 1
                else:
                    reward += self.rewards['fail']

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}


class _Memory(ngym.TrialEnv):
    """Memory task.
    """
    metadata = {
        'paper_link': None,
        'paper_name': None,
        'tags': [],
    }

    def __init__(self, dt=20, anti=True, rewards=None, timing=None,
                 dim_ring=10, modality=0):
        super().__init__(dt=dt)

        self.anti = anti
        self.modalities = [0, 1]
        if modality:
            self.modalities = [modality]

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': lambda: self.rng.uniform(300, 700),
            'stimulus': lambda: self.rng.uniform(200, 1600),
            'delay': lambda: self.rng.uniform(200, 1600),
            'decision': lambda: self.rng.uniform(300, 700)}
        if timing:
            self.timing.update(timing)

        self.abort = False

        # action and observation spaces
        self.dim_ring = dim_ring
        self.theta = np.arange(0, 2 * np.pi, 2 * np.pi / dim_ring)
        self.choices = np.arange(dim_ring)

        name = {
            'fixation': 0,
            'stimulus_mod1': range(1, dim_ring + 1),
            'stimulus_mod2': range(dim_ring + 1, 2 * dim_ring + 1)}
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(1 + 2 * dim_ring,),
            dtype=np.float32, name=name)

        name = {'fixation': 0, 'choice': range(1, dim_ring + 1)}
        self.action_space = spaces.Discrete(1+dim_ring, name=name)

    def _new_trial(self, **kwargs):
        # Trial info
        trial = {
            'ground_truth': self.rng.choice(self.choices),
            'anti': self.anti,
            'modality': self.rng.choice(self.modalities),
        }
        trial.update(kwargs)

        ground_truth = trial['ground_truth']
        if trial['anti']:
            stim_theta = np.mod(self.theta[ground_truth] + np.pi, 2*np.pi)
        else:
            stim_theta = self.theta[ground_truth]
        stim = np.cos(self.theta - stim_theta) / 2 + 0.5

        periods = ['fixation', 'stimulus', 'delay', 'decision']
        self.add_period(periods)

        self.add_ob(1, period=['fixation', 'stimulus', 'delay'], where='fixation')
        if trial['modality'] == 0:
            self.add_ob(stim, period=['stimulus'], where='stimulus_mod1')
        elif trial['modality'] == 1:
            self.add_ob(stim, period=['stimulus'], where='stimulus_mod2')

        self.set_groundtruth(ground_truth, period='decision', where='choice')

        return trial

    def _step(self, action):
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now
        # observations
        if self.in_period('fixation'):
            if action != 0:  # action = 0 means fixating
                new_trial = self.abort
                reward += self.rewards['abort']
        elif self.in_period('decision'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward += self.rewards['correct']
                    self.performance = 1
                else:
                    reward += self.rewards['fail']

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}


class _React(ngym.TrialEnv):
    """React task.
    """
    metadata = {
        'paper_link': None,
        'paper_name': None,
        'tags': [],
    }

    def __init__(self, dt=20, anti=True, rewards=None, timing=None,
                 dim_ring=10, modality=0):
        super().__init__(dt=dt)

        self.anti = anti
        self.modalities = [0, 1]
        if modality:
            self.modalities = [modality]

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': lambda: self.rng.uniform(500, 2500),
            'decision': lambda: self.rng.uniform(300, 1700)}
        if timing:
            self.timing.update(timing)

        self.abort = False

        # action and observation spaces
        self.dim_ring = dim_ring
        self.theta = np.arange(0, 2 * np.pi, 2 * np.pi / dim_ring)
        self.choices = np.arange(dim_ring)

        name = {
            'fixation': 0,
            'stimulus_mod1': range(1, dim_ring + 1),
            'stimulus_mod2': range(dim_ring + 1, 2 * dim_ring + 1)}
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(1 + 2 * dim_ring,),
            dtype=np.float32, name=name)

        name = {'fixation': 0, 'choice': range(1, dim_ring + 1)}
        self.action_space = spaces.Discrete(1+dim_ring, name=name)

    def _new_trial(self, **kwargs):
        # Trial info
        trial = {
            'ground_truth': self.rng.choice(self.choices),
            'anti': self.anti,
            'modality': self.rng.choice(self.modalities),
        }
        trial.update(kwargs)

        ground_truth = trial['ground_truth']
        if trial['anti']:
            stim_theta = np.mod(self.theta[ground_truth] + np.pi, 2*np.pi)
        else:
            stim_theta = self.theta[ground_truth]
        stim = np.cos(self.theta - stim_theta) / 2 + 0.5

        periods = ['fixation', 'decision']
        self.add_period(periods)

        self.add_ob(1, period=['fixation'], where='fixation')
        if trial['modality'] == 0:
            self.add_ob(stim, period=['decision'], where='stimulus_mod1')
        elif trial['modality'] == 1:
            self.add_ob(stim, period=['decision'], where='stimulus_mod2')

        self.set_groundtruth(ground_truth, period='decision', where='choice')

        return trial

    def _step(self, action):
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now
        # observations
        if self.in_period('fixation'):
            if action != 0:  # action = 0 means fixating
                new_trial = self.abort
                reward += self.rewards['abort']
        elif self.in_period('decision'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward += self.rewards['correct']
                    self.performance = 1
                else:
                    reward += self.rewards['fail']

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}


class _ContextDM(ngym.TrialEnv):
    """Context DM task
    """
    metadata = {
        'paper_link': None,
        'paper_name': None,
        'tags': [],
    }

    def __init__(self, dt=20, rewards=None, timing=None, cohs_range=None,
                 dim_ring=10, w_mean_range=None, stim_on=(True, True), stim_attend=(True, True)):
        super().__init__(dt=dt)

        if cohs_range is None:
            self.cohs_range = (0.005, 0.8)
        else:
            self.cohs_range = cohs_range
        if w_mean_range is None:
            self.w_mean_range = (0.8, 1.2)
        else:
            self.w_mean_range = w_mean_range
        self.mod1_on, self.mod2_on = stim_on
        self.mod1_attend, self.mod2_attend = stim_attend

        assert any([self.mod1_on, self.mod2_on])
        if self.mod1_attend:
            assert self.mod1_on
        if self.mod2_attend:
            assert self.mod2_on

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': lambda: self.rng.uniform(200, 600),
            'stim1': lambda: self.rng.uniform(200, 1600),
            'delay1': lambda: self.rng.uniform(200, 1600),
            'stim2': lambda: self.rng.uniform(200, 1600),
            'delay2': lambda: self.rng.uniform(100, 300),
            'decision': lambda: self.rng.uniform(300, 700),}
        if timing:
            self.timing.update(timing)

        self.abort = False

        # action and observation space
        self.dim_ring = dim_ring
        self.theta = np.linspace(0, 2*np.pi, dim_ring+1)[:-1]
        self.choices = np.arange(dim_ring)

        if dim_ring < 2:
            raise ValueError('dim ring can not be smaller than 2')

        name = {
            'fixation': 0,
            'stimulus_mod1': range(1, dim_ring + 1),
            'stimulus_mod2': range(dim_ring + 1, 2 * dim_ring + 1)}
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(1 + 2 * dim_ring,),
            dtype=np.float32, name=name)
        name = {'fixation': 0, 'choice': range(1, dim_ring + 1)}
        self.action_space = spaces.Discrete(1+dim_ring, name=name)

    def _new_trial(self, **kwargs):
        trial = {}
        i_theta1 = self.rng.choice(self.choices)
        theta1 = self.theta[i_theta1]
        theta2_dist = self.rng.choice(
            self.choices[(self.choices >= np.pi / 2) & (self.choices <= 3 * np.pi / 2)])
        theta2 = (theta1 + self.rng.choice([-1, +1]) * theta2_dist) % (2 * np.pi)
        i_theta2 = np.argmax(np.cos(self.theta - theta2))
        theta2 = self.theta[i_theta2]
        trial['theta1'] = theta1
        trial['theta2'] = theta2

        # Periods
        periods = ['fixation', 'stim1', 'delay1', 'stim2', 'delay2', 'decision']
        self.add_period(periods)

        if all([self.mod1_attend, self.mod2_attend]):
            coh = self.rng.uniform(*self.cohs_range)
            coh_sign = self.rng.choice([-1, +1])
            stim_mean = self.rng.uniform(*self.w_mean_range)

            stim1_strength = stim_mean + coh*coh_sign
            stim2_strength = stim_mean - coh*coh_sign

            stim1_diff = stim1_strength * self.rng.uniform(0.2, 0.8) * self.rng.choice([-1, +1])
            stim2_diff = stim2_strength * self.rng.uniform(0.2, 0.8) * self.rng.choice([-1, +1])

            stim1_strength_mod1 = stim1_strength + stim1_diff/2
            stim2_strength_mod1 = stim2_strength + stim2_diff/2
            stim1_strength_mod2 = stim1_strength - stim1_diff/2
            stim2_strength_mod2 = stim2_strength - stim2_diff/2

            resp_idx = np.argmax([stim1_strength, stim2_strength])
        else:
            coh_mod1, coh_mod2 = self.rng.uniform(*self.cohs_range, size=2)
            coh_sign_mod1, coh_sign_mod2 = self.rng.choice([-1, +1], size=2)
            stim_mean_mod1, stim_mean_mod2 = self.rng.uniform(*self.w_mean_range, size=2)

            stim1_strength_mod1 = stim_mean_mod1 + coh_mod1*coh_sign_mod1
            stim2_strength_mod1 = stim_mean_mod1 - coh_mod1*coh_sign_mod1
            stim1_strength_mod2 = stim_mean_mod2 + coh_mod2*coh_sign_mod2
            stim2_strength_mod2 = stim_mean_mod2 - coh_mod2*coh_sign_mod2

            if self.mod1_attend:
                resp_idx = np.argmax([stim1_strength_mod1, stim2_strength_mod1])
            if self.mod2_attend:
                resp_idx = np.argmax([stim1_strength_mod2, stim2_strength_mod2])

        trial['coh1_mod1'] = stim1_strength_mod1
        trial['coh2_mod1'] = stim2_strength_mod1
        trial['coh1_mod2'] = stim1_strength_mod2
        trial['coh2_mod2'] = stim2_strength_mod2

        stim1_mod1 = np.cos(self.theta - theta1) * stim1_strength_mod1 / 2 + 0.5
        stim2_mod1 = np.cos(self.theta - theta2) * stim2_strength_mod1 / 2 + 0.5
        stim1_mod2 = np.cos(self.theta - theta1) * stim1_strength_mod2 / 2 + 0.5
        stim2_mod2 = np.cos(self.theta - theta2) * stim2_strength_mod2 / 2 + 0.5

        self.add_ob(1, period=['fixation', 'stim1', 'delay1', 'stim2', 'delay2'], where='fixation')

        if self.mod1_on:
            self.add_ob(stim1_mod1, period='stim1', where='stimulus_mod1')
            self.add_ob(stim2_mod1, period='stim2', where='stimulus_mod1')
        if self.mod2_on:
            self.add_ob(stim1_mod2, period='stim1', where='stimulus_mod2')
            self.add_ob(stim2_mod2, period='stim2', where='stimulus_mod2')

        i_target = i_theta1 if (resp_idx == 0) else i_theta2
        self.set_groundtruth(i_target, period='decision', where='choice')

        return trial

    def _step(self, action):
        new_trial = False
        gt = self.gt_now
        ob = self.ob_now
        # rewards
        reward = 0
        if self.in_period('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards['abort']
        elif self.in_period('decision'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward = self.rewards['correct']
                    self.performance = 1
                else:
                    reward = self.rewards['fail']

        return ob, reward, False, {'new_trial': new_trial, 'gt': gt}


class _DelayMatch(ngym.TrialEnv):
    """Delay match task
    """
    metadata = {
        'paper_link': None,
        'paper_name': None,
        'tags': [],
    }

    def __init__(self, dt=20, rewards=None, timing=None,
                 dim_ring=10, matchto='sample', matchgo=True, modality=None):
        super().__init__(dt=dt)
        self.matchto = matchto
        if self.matchto not in ['sample', 'category']:
            raise ValueError('Match has to be either sample or category')
        self.matchgo = matchgo
        self.choices = ['match', 'non-match']  # match, non-match
        self.modalities = [0, 1]
        if modality:
            self.modalities = [modality]

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': lambda: self.rng.uniform(200, 600),
            'sample': lambda: self.rng.uniform(200, 1600),
            'delay': lambda: self.rng.uniform(200, 1600),
            'decision': lambda: self.rng.uniform(300, 700)}
        if timing:
            self.timing.update(timing)

        self.abort = False

        if np.mod(dim_ring, 2) != 0:
            raise ValueError('dim ring should be an even number')
        self.dim_ring = dim_ring
        self.half_ring = int(self.dim_ring/2)
        self.theta = np.linspace(0, 2 * np.pi, dim_ring + 1)[:-1]

        name = {
            'fixation': 0,
            'stimulus_mod1': range(1, dim_ring + 1),
            'stimulus_mod2': range(dim_ring + 1, 2 * dim_ring + 1)}
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(1 + 2 * dim_ring,),
            dtype=np.float32, name=name)

        name = {'fixation': 0, 'choice': range(1, dim_ring + 1)}
        self.action_space = spaces.Discrete(1+dim_ring, name=name)

    def _new_trial(self, **kwargs):
        # Trial info
        trial = {
            'ground_truth': self.rng.choice(self.choices),
            'modality': self.rng.choice(self.modalities),
        }
        trial.update(**kwargs)

        ground_truth = trial['ground_truth']
        i_sample_theta = self.rng.choice(self.dim_ring)
        if self.matchto == 'category':
            sample_category = (i_sample_theta > self.half_ring) * 1
            if ground_truth == 'match':
                test_category = sample_category
            else:
                test_category = 1 - sample_category
            i_test_theta = self.rng.choice(self.half_ring)
            i_test_theta += test_category * self.half_ring
        else:  # match to sample
            if ground_truth == 'match':
                i_test_theta = i_sample_theta
            else:
                # non-match is 180 degree apart
                i_test_theta = np.mod(
                    i_sample_theta + self.half_ring, self.dim_ring)    

        trial['sample_theta'] = sample_theta = self.theta[i_sample_theta]
        trial['test_theta'] = test_theta = self.theta[i_test_theta]

        stim_sample = np.cos(self.theta - sample_theta) / 2 + 0.5
        stim_test = np.cos(self.theta - test_theta) / 2 + 0.5

        # Periods
        self.add_period(['fixation', 'sample', 'delay', 'decision'])

        self.add_ob(1, period=['fixation', 'sample', 'delay'], where='fixation')
        if trial['modality'] == 0:
            self.add_ob(stim_sample, period='sample', where='stimulus_mod1')
            self.add_ob(stim_test, period='decision', where='stimulus_mod1')
        elif trial['modality'] == 1:
            self.add_ob(stim_sample, period='sample', where='stimulus_mod2')
            self.add_ob(stim_test, period='decision', where='stimulus_mod2')

        if ((ground_truth == 'match' and self.matchgo) or
                (ground_truth == 'non-match' and not self.matchgo)):
            self.set_groundtruth(i_test_theta, period='decision', where='choice')
        else:
            self.set_groundtruth(0)

        return trial

    def _step(self, action, **kwargs):
        new_trial = False

        ob = self.ob_now
        gt = self.gt_now

        reward = 0
        if self.in_period('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards['abort']
        elif self.in_period('decision'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward = self.rewards['correct']
                    self.performance = 1
                else:
                    reward = self.rewards['fail']

        return ob, reward, False, {'new_trial': new_trial, 'gt': gt}
