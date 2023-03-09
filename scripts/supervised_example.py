import torch
import pytorch_lightning as pl
# import neurogym as ngym
from ttrnn.trainer import Supervised
from ttrnn.dataset import NeurogymTrialEnvDataset, NeurogymDataLoader
from ttrnn.tasks.driscoll2022 import Driscoll2022, MemoryPro
from ttrnn.tasks.wrappers import DiscreteToBoxWrapper, RingToBoxWrapper
from ttrnn.callbacks import TaskPerformance, TrajectoryPlot, TaskPlot

rnn_params = {
    'input_size': 3, 
    'hidden_size': 32, 
    'output_size': 3, 
    'nonlinearity': 'relu',
    'bias': False, 
    'trainable_h0': False,
    'batch_first': True,
    # 'init_config': {
    #     'default': ('normal_', {'mean': 0.0, 'std': 1 / (32 ** 0.5)}),
    # },
    # 'noise_config': {'use_noise': True, 'noise_type': 'normal', 'noise_params': {'mean': 0.0, 'std': 0.05}}
}

model = Supervised(
    rnn_type='RNN',
    rnn_params=rnn_params,
    optim_type='Adam',
    optim_params={'lr': 1e-3, 'weight_decay': 1e-6},
    loss_func='mse_loss',
)

# import pdb; pdb.set_trace()

task = 'PerceptualDecisionMaking-v0'
# task = MemoryPro()
env_kwargs = {'dt': 50, 'timing': {'fixation': 100, 'stimulus': 2000, 'delay': 0, 'decision': 400}}
wrappers = [(DiscreteToBoxWrapper, {})] # [(RingToBoxWrapper, {})]

train_dataloader = NeurogymDataLoader(
    NeurogymTrialEnvDataset(task, env_kwargs, wrappers=wrappers, num_trials=640, seq_len=200, batch_first=True, save_envs=False), 
    static=False,
    batch_size=64, 
    shuffle=True, 
)

val_dataloader = NeurogymDataLoader(
    NeurogymTrialEnvDataset(task, env_kwargs, wrappers=wrappers, num_trials=128, seq_len=200, batch_first=True, save_envs=True), 
    static=True,
    batch_size=64, 
    shuffle=False, 
)

loggers = [
    # pl.loggers.CSVLogger(save_dir="csv_logs"),
    # pl.loggers.WandbLogger(project='ttrnn-dev'),
]
callbacks = [
    TaskPerformance(log_every_n_epochs=10, threshold=0.6),
    # TrajectoryPlot(log_every_n_epochs=5),
    # TaskPlot(log_every_n_epochs=5),
]

trainer = pl.Trainer(
    max_epochs=100,
    callbacks=callbacks,
    accelerator='gpu',
    devices=1,
    num_nodes=1,
    log_every_n_steps=1,
    enable_progress_bar=True,
    enable_model_summary=True,
    enable_checkpointing=False,
    logger=loggers,
)

trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# import pdb; pdb.set_trace()

# import matplotlib.pyplot as plt
# from ttrnn.analysis.fixed_points.fixed_point_finder import FixedPointFinder
# from ttrnn.analysis.fixed_points.plot_utils import plot_fps

# fpf = FixedPointFinder(model.model.rnn.rnn_cell, device=model.device)

# dataloader = trainer.val_dataloaders
# dataloader = dataloader[0]
# dataloader.freeze()
# model_states = []
# model_inputs = []
# for batch in dataloader:
#     # Move data to the right device
#     inputs, target = batch
#     inputs = inputs.to(model.device)
#     # Perform the forward pass through the model
#     outputs, hs = model(inputs)
#     model_states.append(hs)
#     model_inputs.append(inputs)
# model_states = torch.cat(model_states).detach()
# model_inputs = torch.cat(model_inputs).detach()

# inputs, initial_states = fpf.sample_inputs_and_states(model_inputs, model_states, n_inits=40)
# unique_fps, _ = fpf.find_fixed_points(initial_states, inputs)

# fig = plt.figure()
# plot_fps(unique_fps, 
#     state_traj=model_states,
#     fig=fig
#     )
# plt.savefig('fps.png')

# import pdb; pdb.set_trace()