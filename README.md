# ttrnn
PyTorch package for task-trained RNNs

## Goals
**Models**
* Vanilla RNN (Elman et al. 1990)
* GRU (Cho et al. 2014)
* LSTM (Hochreiter et al. 1997)
* UGRNN (Collins et al. 2017)
* euRNN (Jing et al. 2017)
* Excitatory-Inhibitory / Dale's law (Song et al. 2016)
* Locality-masked RNNs (Khona et al. 2022)
* Low-rank RNNs (Mastrogiuseppe)
* Multi-area RNNs (hierarchical and parallel)
* Spiking neural networks?
* Reservoir models?
* etc...

**Learning Rules**
* SGD
* Adam
* SGD with clipping & scaling (Pascanu et al. 2013)
* Hessian-free (Martens et al. 2011)
* RTRL (Williams et al. 1989)
* Recursive least squares (Haykins 2002)
* FORCE (Sussillo et al. 2009)
* Hebbian (Miconi 2017)
* UORO (Tallec et al. 2017)
* KF-RTRL (Mujika et al. 2018)
* RFLO (Murray et al. 2019)
* SnAp RTRL (Menick et al. 2021)
* Genetic algorithm?
* Meta-learning?
* Multiple rules at once, varying timescales?
* etc...

**Analysis**
* Tensorboard / WandB logging during training
* JSLDS co-training
* Fixed-point finding
* RADD
* Neuron clustering / task variance
* etc...

**Misc**
* Multi-task training, mixed or sequential
* Flexible support for different loss functions
* Reinforcement learning maybe, though Stable Baselines already exists
* etc...

*Disclaimer: I don't actually know what 80% of this stuff is, so the list is very tentative and ill-informed*

## Inspirations
* Neurogym
* psychRNN
* [Maheswaranathan et al. 2019](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7416639/)
* [McMahan et al. 2021](https://proceedings.neurips.cc/paper/2021/hash/b87039703fe79778e9f140b78621d7fb-Abstract.html)
