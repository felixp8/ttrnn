import torch
import pytorch_lightning as pl
# import neurogym as ngym
from ttrnn.trainer import Supervised, A2C
from ttrnn.dataset import NeurogymTrialEnvDataset, NeurogymDataLoader
from ttrnn.tasks.driscoll2022 import Driscoll2022, MemoryPro
from ttrnn.tasks.wrappers import DiscreteToBoxWrapper, RingToBoxWrapper
from ttrnn.callbacks import TaskPerformance, TrajectoryPlot, TaskPlot

rnn_params = {
    'input_size': 3,
    'hidden_size': 64, 
    # 'output_size': 3, 
    # 'nonlinearity': 'relu',
    'bias': False, 
    'trainable_h0': False,
    'batch_first': True,
    # 'init_config': {
    #     'default': ('normal_', {'mean': 0.0, 'std': 1 / (32 ** 0.5)}),
    # },
    # 'noise_config': {'use_noise': True, 'noise_type': 'normal', 'noise_params': {'mean': 0.0, 'std': 0.05}}
}
# import pdb; pdb.set_trace()

task = 'PerceptualDecisionMaking-v0'
# task = MemoryPro(
#     dim_ring=4,
#     dt=20,
#     timing={'fixation': 200, 'stimulus': 1000, 'delay': 1000, 'decision': 400},
# )
env_kwargs = {'dt': 50, 'timing': {'fixation': 300, 'stimulus': 500, 'delay': 0, 'decision': 300}}
wrappers = [] # [(RingToBoxWrapper, {})] # [(DiscreteToBoxWrapper, {})]

loggers = [
    # pl.loggers.CSVLogger(save_dir="csv_logs"),
    # pl.loggers.WandbLogger(project='ttrnn-dev'),
]
callbacks = [
    TaskPerformance(log_every_n_epochs=100, threshold=0.6),
    # TrajectoryPlot(log_every_n_epochs=5),
    # TaskPlot(log_every_n_epochs=5),
]

model = A2C(
    env=task,
    env_kwargs=env_kwargs,
    rnn_type='RNN',
    rnn_params=rnn_params,
    optim_type='RMSprop',
    optim_params={'lr': 7e-4}, #, 'weight_decay': 1e-6},
    max_batch_len=50,
    max_batch_episodes=10,
    unroll_len=100,
)


trainer = pl.Trainer(
    max_epochs=2000,
    callbacks=callbacks,
    # accelerator='gpu',
    # devices=1,
    # num_nodes=1,
    log_every_n_steps=1,
    enable_progress_bar=True,
    enable_model_summary=True,
    enable_checkpointing=False,
    logger=loggers,
    # gradient_clip_val=0.5,
)

trainer.fit(model=model) # , train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# import pdb; pdb.set_trace()