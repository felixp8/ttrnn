
import os
import shutil
import pickle
import copy
import torch
import pytorch_lightning as pl
import neurogym as ngym
from pytorch_lightning.callbacks import ModelCheckpoint
from neurogym.wrappers import PassAction, PassReward, Noise
from ttrnn.trainer import Supervised, A2C, MetaA2C
from ttrnn.dataset import NeurogymTrialEnvDataset, NeurogymDataLoader
from ttrnn.tasks.driscoll2022 import Driscoll2022, MemoryPro
from ttrnn.tasks.harlow import HarlowMinimal, Harlow1D, HarlowMinimalDelay, HarlowMinimalRT
from ttrnn.tasks.wrappers import DiscreteToBoxWrapper, RingToBoxWrapper, ParallelEnvs
from ttrnn.callbacks import TaskPerformance, TrajectoryPlot, TaskPlot

import sys
if len(sys.argv) > 1:
    seed = int(sys.argv[1]) 
else:
    seed = 0

pl.seed_everything(seed)

rnn_params = {
    'input_size': 1 + 11 * 2 + 3 + 1,
    'hidden_size': 256, 
    # 'output_size': 3, 
    # 'nonlinearity': 'relu',
    'bias': True, 
    'trainable_h0': False,
    'batch_first': True,
    # 'init_config': {
    #     'default': ('normal_', {'mean': 0.0, 'std': 1 / (32 ** 0.5)}),
    # },
    'noise_config': {'enable': False, 'noise_type': 'normal', 'noise_params': {'mean': 0.0, 'std': 0.05}},
    'dt': 10,
    'tau': 20,
    'trainable_tau': False,
}
# import pdb; pdb.set_trace()

# task = 'PerceptualDecisionMaking-v0'
# task = MemoryPro(
#     dim_ring=4,
#     dt=20,
#     timing={'fixation': 200, 'stimulus': 1000, 'delay': 1000, 'decision': 400},
# )
# task = HarlowMinimal(
#     dt=100, obj_dim=5, obj_mode="kb", obj_init="normal", orthogonalize=True,
#     # rewards={'fail': -1.0},
#     inter_trial_interval=2, num_trials_before_reset=6,
# )
# task = Harlow1D(
#     dt=100, obj_dim=1, obj_dist=3, obj_mode="kb", obj_init="randint",
#     rewards={'fail': -1.0},
#     inter_trial_interval=0, num_trials_before_reset=6,
# )
task = HarlowMinimalDelay(
    dt=100,
    obj_dim=11,
    obj_mode="kb", 
    obj_init="normal",
    orthogonalize=False,
    normalize=True,
    abort=True,
    rewards={'abort': -0.1, 'correct': 1.0, 'fail': 0.0},
    timing={'fixation': 200, 'stimulus': 400, 'delay': 200, 'decision': 200},
    num_trials_before_reset=6,
    r_tmax=-1.0,
)
# task = HarlowMinimalRT(
#     dt=100,
#     obj_dim=5,
#     obj_mode="kb", 
#     obj_init="normal",
#     orthogonalize=True,
#     abort=True,
#     rewards={'abort': -0.1, 'correct': 1.0, 'fail': -1.0},
#     timing={'fixation': 400, 'decision': 1600},
#     num_trials_before_reset=6,
# )
# env_kwargs = {'dt': 100, 'timing': {'fixation': 300, 'stimulus': 500, 'delay': 0, 'decision': 300}}
env_kwargs = {}
wrappers = [
    (Noise, {'std_noise': 0.1}),
    (PassAction, {'one_hot': True}), 
    (PassReward, {}), 
    (ParallelEnvs, {'num_envs': 8})
] # [(RingToBoxWrapper, {})] # [(DiscreteToBoxWrapper, {})]
# wrappers = []

ckpt_dir = "/home/fpei2/learning/harlow-rnn-analysis/runs/harlowdelaynotorth_rnn256/"

overwrite = True
if overwrite:
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)

loggers = [
    pl.loggers.CSVLogger(save_dir=ckpt_dir),
    # pl.loggers.WandbLogger(project='ttrnn-dev'),
]
callbacks = [
    # TaskPerformance(log_every_n_epochs=250, threshold=0.6),
    # TrajectoryPlot(log_every_n_epochs=5),
    # TaskPlot(log_every_n_epochs=5),
    ModelCheckpoint(dirpath=ckpt_dir, monitor="train/loss", save_top_k=8, every_n_epochs=2500),
]
    
if len(wrappers) > 0:
    for wrapper, wrapper_kwargs in wrappers:
        task = wrapper(task, **wrapper_kwargs)
backup = copy.deepcopy(task)

model = A2C(
    env=task,
    env_kwargs=env_kwargs,
    rnn_type='leakyRNN',
    rnn_params=rnn_params,
    actor_type='linear',
    critic_type='linear',
    encoder_type='none',
    optim_type='RMSprop',
    optim_params={'lr': 7.5e-4}, #, 'weight_decay': 1e-6},
    epoch_len=20,
    reset_state_per_episode=False,
    trials_per_episode=6,
    discount_gamma=0.91,
    critic_beta=0.4,
    entropy_beta=0.001,
    entropy_anneal_len=30000,
)

trainer = pl.Trainer(
    max_epochs=30000,
    callbacks=callbacks,
    # accelerator='gpu',
    # devices=1,
    # num_nodes=1,
    log_every_n_steps=100,
    enable_progress_bar=True,
    enable_model_summary=True,
    enable_checkpointing=True,
    logger=loggers,
    gradient_clip_val=0.5,
)

trainer.fit(model=model) # , train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# TODO: pickle `task` for later loading
# with open(os.path.join(ckpt_dir, "task.pkl"), 'wb') as f:
#     backup.obj1_builder = None
#     backup.obj2_builder = None
#     pickle.dump(backup, f)

# import pdb; pdb.set_trace()