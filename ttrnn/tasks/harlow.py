import numpy as np
import gym
import neurogym as ngym
from neurogym import spaces

from typing import Optional, Literal


class HarlowSimple(ngym.TrialEnv):
    """Abstract base class for non-visual Harlow tasks handling object creation"""
    def __init__(
        self,
        dt: int = 100,
        obj_dim: int = 8, 
        obj_mode: Literal["kb", "su", "sr", "sb"] = "kb",
        obj_init: Literal["normal", "uniform", "randint"] = "uniform",
        orthogonalize: bool = True,
        num_trials_before_reset: int = 6,
        r_tmax: int = 0,
    ):
        super(HarlowSimple, self).__init__(
            dt=dt, 
            num_trials_before_reset=num_trials_before_reset, 
            r_tmax=r_tmax,
        )

        self.obj_dim = obj_dim
        self.obj_mode = obj_mode
        self.orthogonalize = orthogonalize
        self.obj_init = obj_init
        self.obj1 = None
        self.obj2 = None
        self.obj_left = None

    def reset(self, reset_obj=False, **kwargs):
        if reset_obj:
            self.init_objects()
        return super().reset(**kwargs)

    def new_trial(self, **kwargs):
        """Public interface for starting a new trial.

        Returns:
            trial: dict of trial information. Available to step function as
                self.trial
        """
        if self.num_tr >= self.num_tr_exp:
            # if max trials reached
            self.reset(reset_obj=True, **kwargs)
        else:
            # Reset for next trial
            self._tmax = 0  # reset, self.tmax not reset so it can be used in step
            self._ob_built = False
            self._gt_built = False
            trial = self._new_trial(**kwargs)
            self.trial = trial
            self.num_tr += 1  # Increment trial count
            self._has_gt = self._gt_built
        return self.trial

    def init_objects(self):
        self.reward_idx = self.rng.choice([0, 1])

        if self.obj_init == "uniform":
            initializer = lambda: self.rng.uniform(low=-1.0, high=1.0, size=self.obj_dim)
        elif self.obj_init == "normal":
            initializer = lambda: self.rng.normal(loc=0.0, scale=1.0, size=self.obj_dim)
        elif self.obj_init == "randint":
            initializer = lambda: self.rng.randint(low=2., high=100., size=self.obj_dim) / 100. # TODO: just use uniform in these cases
        else:
            raise ValueError
        if self.orthogonalize and self.obj_dim > 1:
            orthogonalizer = lambda x, y: (x, y - (np.dot(x,y) / np.dot(x,x)) * x)
        else:
            orthogonalizer = lambda x, y: (x, y)

        obj1, obj2 = orthogonalizer(initializer(), initializer())
        n_retries = 0
        while np.allclose(obj1, obj2) or np.linalg.norm(obj1) < 1e-3 or np.linalg.norm(obj2) < 1e-3:
            obj1, obj2 = orthogonalizer(initializer(), initializer())
            n_retries += 1
            if n_retries > 10:
                raise AssertionError("Object generation failed")

        if self.obj_mode == "sb": # switch both
            obj_mode = np.random.choice(["su", "sr", "kb"])
        else:
            obj_mode = self.obj_mode

        if obj_mode == "kb": # keep both
            self.obj1_builder = lambda: obj1
            self.obj2_builder = lambda: obj2
        elif obj_mode == "su": # switch unrewarded
            self.obj1_builder = (lambda: obj1) if self.reward_idx == 0 else \
                (lambda: orthogonalizer(obj1, initializer())[1])
            self.obj2_builder = (lambda: obj2) if self.reward_idx == 1 else \
                (lambda: orthogonalizer(obj2, initializer())[1])
        elif obj_mode == "sr": # switch rewarded
            self.obj1_builder = (lambda: obj1) if self.reward_idx == 1 else \
                (lambda: orthogonalizer(obj1, initializer())[1])
            self.obj2_builder = (lambda: obj2) if self.reward_idx == 0 else \
                (lambda: orthogonalizer(obj2, initializer())[1])
        else:
            raise ValueError

    def get_objects(self):
        obj1, obj2 = self.obj1_builder(), self.obj2_builder()
        n_retries = 0
        while np.allclose(obj1, obj2) or np.linalg.norm(obj1) < 1e-3 or np.linalg.norm(obj2) < 1e-3:
            obj1, obj2 = self.obj1_builder(), self.obj2_builder()
            n_retries += 1
            if n_retries > 10:
                raise AssertionError("Object generation failed")
        return obj1, obj2

    def set_trial_params(self, **kwargs):
        """Hacky workaround"""
        return self._new_trial(**kwargs)


class HarlowMinimal(HarlowSimple):
    """Minimal Harlow task"""
    def __init__(
        self,
        dt: int = 100,
        obj_dim: int = 8, 
        rewards: Optional[dict] = None,
        obj_mode: Literal["kb", "su", "sr", "sb"] = "kb",
        obj_init: Literal["normal", "uniform", "randint"] = "uniform",
        orthogonalize: bool = True,
        inter_trial_interval: int = 4, # * dt
        num_trials_before_reset: int = 6,
        r_tmax: int = 0,
        max_tlen: int = 250, # * dt
    ):
        super(HarlowMinimal, self).__init__(
            dt=dt, 
            obj_dim=obj_dim, 
            obj_mode=obj_mode,
            obj_init=obj_init,
            orthogonalize=orthogonalize,
            num_trials_before_reset=num_trials_before_reset, 
            r_tmax=r_tmax,
        )

        self.rewards = {'fixation': +0.2, 'correct': +1.0, 'fail': 0.0}
        if rewards:
            self.rewards.update(rewards)

        name = {'fixation': 0, 'stimulus': range(1, obj_dim*2 + 1)}
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(1+obj_dim*2,), dtype=np.float32, name=name)

        name = {'fixation': 0, 'choice': range(1, 2 + 1)}
        self.action_space = spaces.Discrete(2+1, name=name)

        self.period = "fixation"
        self.iti = (inter_trial_interval + 1) * dt
        self.max_tlen = max_tlen * dt
        self.tmax = self.max_tlen

        self.init_objects()

    def reset(self, action=0, **kwargs):
        return super().reset(action=action, **kwargs)

    def _new_trial(self, **kwargs):
        # Trial info
        self.period = "fixation"
        self.tmax = self.max_tlen
        self.obj1, self.obj2 = self.get_objects()
        self.obj_left = self.rng.choice([0, 1])
        trial = {
            'reward_idx': self.reward_idx,
            'obj_left': self.obj_left,
        }
        trial.update(kwargs)

        return trial

    def _step(self, action):
        new_trial = False
        # rewards
        reward = 0
        # observations
        if self.period == 'fixation':
            if action == 0:  # action = 1 means fixating
                reward += self.rewards['fixation']
                self.period = 'stimulus'
        elif self.period == 'stimulus':
            if action > 0:
                if action == 2 - int(self.reward_idx == self.obj_left):
                    reward += self.rewards['correct']
                    self.performance = 1
                else:
                    reward += self.rewards['fail']
                self.period = "iti"
                self.tmax = self.t + self.iti
        elif self.period == "iti":
            if (self.t + 2 * self.dt) > self.tmax:
                new_trial = True
                self.period = "fixation"

        if self.period == 'fixation':
            ob = np.zeros(self.observation_space.shape, dtype=np.float32)
            ob[0] = 1.
        elif self.period == 'stimulus':
            ob = np.zeros(self.observation_space.shape, dtype=np.float32)
            ob[1:(1+self.obj_dim)] = self.obj1 if self.obj_left == 0 else self.obj2
            ob[(1+self.obj_dim):] = self.obj2 if self.obj_left == 0 else self.obj1
        elif self.period == "iti":
            ob = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            raise ValueError

        return ob, reward, False, {'new_trial': new_trial}
    

class Harlow1D(HarlowSimple):
    """1D Harlow task"""
    def __init__(
        self,
        dt: int = 100,
        obj_dim: int = 1, 
        obj_dist: int = 3, # steps to choose
        rewards: Optional[dict] = None,
        obj_mode: Literal["kb", "su", "sr", "sb"] = "kb",
        obj_init: Literal["normal", "uniform", "randint"] = "randint",
        orthogonalize: bool = True,
        inter_trial_interval: int = 4,
        num_trials_before_reset: int = 6,
        r_tmax: int = 0,
        max_tlen: int = 250, # * dt
    ):
        super(Harlow1D, self).__init__(
            dt=dt, 
            obj_dim=obj_dim,
            obj_mode=obj_mode,
            obj_init=obj_init,
            orthogonalize=orthogonalize,
            num_trials_before_reset=num_trials_before_reset, 
            r_tmax=r_tmax,
        )

        self.rewards = {'fixation': +0.2, 'correct': +1.0, 'fail': 0.0}
        if rewards:
            self.rewards.update(rewards)

        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=((obj_dist*2 - 1) + obj_dim*2 + obj_dist*2,), dtype=np.float32)

        name = {'noop': 0, 'left': 1, 'right': 2}
        self.action_space = spaces.Discrete(3, name=name)

        self.period = "fixation"
        self.iti = (inter_trial_interval + 1) * dt
        self.max_tlen = max_tlen * dt
        self.tmax = self.max_tlen

        self.init_objects()

        self.obs_dim = (obj_dist*2 - 1) + obj_dim*2 + obj_dist*2
        self.center = self.obs_dim // 2
        self.obj_dist = obj_dist
        self.loc = self.rng.choice([-self.obj_dist, self.obj_dist])
        self.obs = np.zeros(self.obs_dim)

    def reset(self, action=0, **kwargs):
        return super().reset(action=action, **kwargs)

    def _new_trial(self, **kwargs):
        # Trial info
        self.period = "fixation"
        self.tmax = self.max_tlen
        self.obj1, self.obj2 = self.get_objects()
        self.obj_left = self.rng.choice([0, 1])
        trial = {
            'reward_idx': self.reward_idx,
            'obj_left': self.obj_left,
        }
        trial.update(kwargs)

        self.obs[:] = 0.
        if self.loc == 0 or np.abs(self.loc) > self.obj_dist:
            self.loc = int(self.rng.randint(1, self.obj_dist) * self.rng.choice([-1, 1]))
        self.obs[self.center - self.loc] = 1.

        return trial

    def _step(self, action):
        new_trial = False
        # rewards
        reward = 0
        # roll obs
        if action == 1:
            if self.loc > -self.obj_dist:
                self.obs = np.roll(self.obs, 1)
                self.loc = int(self.loc - 1)
        elif action == 2:
            if self.loc < self.obj_dist:
                self.obs = np.roll(self.obs, -1)
                self.loc = int(self.loc + 1)
        # observations
        if self.period == 'fixation':
            if self.obs[self.center] == 1.:  # action = 1 means fixating
                reward += self.rewards['fixation']
                self.period = 'stimulus'
                self.obs[:] = 0.
                self.obs[
                    (self.center - self.obj_dist - self.obj_dim + 1):(self.center - self.obj_dist + 1)
                ] = self.obj1 if self.obj_left == 0 else self.obj2
                self.obs[
                    (self.center + self.obj_dist):(self.center + self.obj_dist + self.obj_dim)
                ] = self.obj2 if self.obj_left == 0 else self.obj1
        elif self.period == 'stimulus':
            if np.abs(self.loc) == self.obj_dist:
                if (self.loc < 0) == (self.reward_idx == self.obj_left):
                    reward += self.rewards['correct']
                    self.performance = 1
                else:
                    reward += self.rewards['fail']
                # import pdb; pdb.set_trace()
                self.period = "iti"
                self.tmax = self.t + self.iti
                self.obs[:] = 0.
        elif self.period == "iti":
            pass
            # if (self.t + 1 * self.dt) > self.tmax:
            #     new_trial = True

        return self.obs, reward, False, {'new_trial': new_trial}


class HarlowMinimalDelay(HarlowSimple):
    def __init__(
        self,
        dt: int = 100,
        obj_dim: int = 8, 
        rewards: Optional[dict] = None,
        timing: Optional[dict] = None,
        abort: bool = False,
        obj_mode: Literal["kb", "su", "sr", "sb"] = "kb",
        obj_init: Literal["normal", "uniform", "randint"] = "uniform",
        orthogonalize: bool = True,
        num_trials_before_reset: int = 6,
        r_tmax: int = 0,
        stim_to_decision: bool = False,
    ):
        super(HarlowMinimalDelay, self).__init__(
            dt=dt,
            obj_dim=obj_dim,
            obj_mode=obj_mode,
            obj_init=obj_init,
            orthogonalize=orthogonalize,
            num_trials_before_reset=num_trials_before_reset,
            r_tmax=r_tmax,
        )
        self.stim_to_decision = stim_to_decision

        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)
        
        self.timing = {
            'fixation': 500,
            'stimulus': 500,
            'delay': 0,
            'decision': 500,}
        if timing:
            self.timing.update(timing)

        self.abort = abort

        name = {'fixation': 0, 'stimulus': range(1, obj_dim*2 + 1)}
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(1+obj_dim*2,), dtype=np.float32, name=name)

        name = {'fixation': 0, 'choice': range(1, 2 + 1)}
        self.action_space = spaces.Discrete(2+1, name=name)

        self.init_objects()

    def reset(self, action=0, **kwargs):
        return super().reset(action=action, **kwargs)

    def _new_trial(self, **kwargs):
        # Trial info
        self.obj1, self.obj2 = self.get_objects()
        # self.obj_left = self.rng.choice([0, 1])
        trial = {
            'reward_idx': self.reward_idx,
            'obj_left': self.rng.choice([0, 1]),
        }
        trial.update(kwargs)
        self.obj_left = trial.get('obj_left')

        ground_truth = int(self.reward_idx != self.obj_left)

        # Periods
        periods = ['fixation', 'stimulus', 'delay', 'decision']
        self.add_period(periods)

        self.add_ob(1, period=['fixation', 'stimulus', 'delay'], where='fixation')
        stim = np.concatenate([
            self.obj1 if self.obj_left == 0 else self.obj2,
            self.obj2 if self.obj_left == 0 else self.obj1,
        ], axis=0)
        self.add_ob(stim, 'stimulus', where='stimulus')
        if self.stim_to_decision:
            self.add_ob(stim, period=['delay', 'decision'], where='stimulus')

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
        if self.in_period('stimulus'):
            if action != 0:  # action = 0 means fixating
                new_trial = self.abort
                reward += self.rewards['abort']
        if self.in_period('delay'):
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
    

class HarlowMinimalRT(HarlowSimple):
    def __init__(
        self,
        dt: int = 100,
        obj_dim: int = 8, 
        rewards: Optional[dict] = None,
        timing: Optional[dict] = None,
        abort: bool = False,
        obj_mode: Literal["kb", "su", "sr", "sb"] = "kb",
        obj_init: Literal["normal", "uniform", "randint"] = "uniform",
        orthogonalize: bool = True,
        num_trials_before_reset: int = 6,
        r_tmax: int = 0,
    ):
        super(HarlowMinimalRT, self).__init__(
            dt=dt,
            obj_dim=obj_dim,
            obj_mode=obj_mode,
            obj_init=obj_init,
            orthogonalize=orthogonalize,
            num_trials_before_reset=num_trials_before_reset,
            r_tmax=r_tmax,
        )

        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)
        
        self.timing = {
            'fixation': 500,
            'decision': 500,}
        if timing:
            self.timing.update(timing)

        self.abort = abort

        name = {'fixation': 0, 'stimulus': range(1, obj_dim*2 + 1)}
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(1+obj_dim*2,), dtype=np.float32, name=name)

        name = {'fixation': 0, 'choice': range(1, 2 + 1)}
        self.action_space = spaces.Discrete(2+1, name=name)

        self.init_objects()

    def reset(self, action=0, **kwargs):
        return super().reset(action=action, **kwargs)

    def _new_trial(self, **kwargs):
        # Trial info
        self.obj1, self.obj2 = self.get_objects()
        self.obj_left = self.rng.choice([0, 1])
        trial = {
            'reward_idx': self.reward_idx,
            'obj_left': self.obj_left,
        }
        trial.update(kwargs)

        ground_truth = int(self.reward_idx != self.obj_left)

        # Periods
        periods = ['fixation', 'decision']
        self.add_period(periods)

        self.add_ob(1, period=['fixation'], where='fixation')
        stim = np.concatenate([
            self.obj1 if self.obj_left == 0 else self.obj2,
            self.obj2 if self.obj_left == 0 else self.obj1,
        ], axis=0)
        self.add_ob(stim, 'decision', where='stimulus')

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


if __name__ == "__main__":
    # env = Harlow1D(
    #     inter_trial_interval=0,
    # )
    env = HarlowMinimalDelay(
        dt=100,
        abort=True,
        obj_dim=5,
    )
    obs, _ = env.reset()
    reward = 0

    while True:
        print(f'Obs = {obs}')
        print(f'Reward = {reward}')

        action = int(input("Fix (0) or Left (1) or Right (2): "))
        
        if action > 2 or action < 0:
            break

        obs, reward, done, trunc, info = env.step(action)