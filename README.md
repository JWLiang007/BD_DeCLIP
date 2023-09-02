<!-- # DeCLIP
Supervision Exists Everywhere: A Data Efficient Contrastive Language-Image Pre-training Paradigm.

Our paper is available on [arxiv](https://arxiv.org/abs/2110.05208) -->

# Backdoor Attack against Declip

This repo is based on the project Declip.

## Installation

Please refer to [get_started.md](docs/get_started.md#installation) for installation and [dataset_prepare.md](docs/dataset_prepare.md#prepare-datasets) for dataset preparation.


## Get Started

#### 1. Backdoored Pre-training

We implement backdoor attack on the [r50 declip](experiments/declip_experiments/yfcc15m/yfcc15m_r50_declip) model pretrained on yfcc15m dataset.

Run:
```bash
cd experiments/declip_experiments/yfcc15m/yfcc15m_r50_declip
sh run.sh $PARTITION $JOB_NAME $CONFIG
# PARTITION and JOB_NAME are parameters of Slurm, CONFIG is the path to config file
# Example:
sh run.sh debug bd_train  config_poison_001.yaml
```
#### 2. Backdoor Defense
You only need to assign the corresponding file for different backdoor defense methods 

* Fine-Tuning:
```bash
sh run.sh debug ft_defense   config_ft.yaml
```
* NAD:
```bash
sh run.sh debug nad_defense   config_nad.yaml
```
* Fine-Tuning:
```bash
sh run.sh debug ours_defense   config_ours_adp.yaml
```

#### 3. Zero-shot Evalution

You can add a argument `--evaluate` on run script for zero-shot evalution.
```bash
# Append the `--evaluate` at the end of the run.sh script
...
...
python -u -m prototype.solver.declip_solver --config ${CONFIG} --evaluate
... 
```
## Results
#### 1. Image Classification

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="center">Index</th>
<th valign="center">Method</th>
<th valign="center">ASR</th>
<th valign="center">ASR Drop</th>


<tr>
<td align="center">1</td>
<td align="center">Victim</td>
<td align="center">96.7</td>
<td align="center">-</td>
</tr>

<tr>
<td align="center">2</td>
<td align="center">NAD</td>
<td align="center">82.88</td>
<td align="center">13.82</td>
</tr>
<tr>
<td align="center">3</td>
<td align="center">Ours</td>
<td align="center">78.26</td>
<td align="center">18.44(↑33%)</td>
</tr>

</tbody></table>

#### 2. Image Captioning
__TODO__



## Acknowledgement

Our repo is based on [prototype](https://github.com/ModelTC/prototype) and [Declip](https://github.com/Sense-GVT/DeCLIP).


