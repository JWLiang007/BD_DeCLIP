#### 1. Installation

Install PyTorch. The code has been tested with PyTorch 2.0.0.

Install and configure [Slurm](https://slurm.schedmd.com/) in your system.

We organize the different models trained on different data through separate [experimental catalogs] (experiments/), you can check the dir for detail.

#### 2. Pre-training

You can run `run.sh` directly to train the corresponding model. We train most of our models on 4x8-gpu nodes. Check the config in the experiment directory of the corresponding model for details.

#### 3. Zero-shot Evalution

You can add a argument `--evaluate` on run script for zero-shot evalution. There are two ways to set the model file location:


+ Move the checkpoint file to the corresponding experiment directory and rename it to checkpoints/ckpt.pth.tar

+ Change the config file:  

```
...
...
saver:
    print_freq: 100
    val_freq: 2000
    save_freq: 500
    save_many: False
    pretrain:
        auto_resume: False
        path: /path/to/checkpoint
```
