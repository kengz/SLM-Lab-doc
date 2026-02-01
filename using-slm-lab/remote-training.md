# Remote Training with dstack ☁️

## Overview

SLM Lab uses [dstack](https://dstack.ai/) for cloud GPU training and HuggingFace for experiment storage. This allows you to:

* Run experiments on cloud GPUs without managing infrastructure
* Automatically upload results to HuggingFace
* Pull results back for local analysis

## Setup

### 1. Install dstack

```bash
uv tool install dstack
```

### 2. Configure dstack

Create an account at [dstack.ai](https://sky.dstack.ai/) and get your token:

```bash
dstack project add --name your-project --url https://sky.dstack.ai --token $DSTACK_TOKEN -y
```

This saves configuration to `~/.dstack/config.yml`.

### 3. Set up HuggingFace credentials

Create a `.env` file in your SLM-Lab directory with your HuggingFace token:

```bash
HF_TOKEN=hf_xxxxxxxxxxxx
HF_REPO=your-username/slm-lab-results
```

Source the environment before running:

```bash
source .env
```

## Running Remote Experiments

### Basic Usage

```bash
# Run training on cloud GPU
source .env && slm-lab run-remote --gpu slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari train -n my-experiment

# Run hyperparameter search
source .env && slm-lab run-remote --gpu slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari search -n my-search
```

The `-n` flag names your dstack run for easy identification.

### Using Variable Substitution

```bash
source .env && slm-lab run-remote --gpu -s env=ALE/Breakout-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari train -n ppo-breakout
```

### Monitoring Runs

```bash
# List running jobs
dstack ps

# View logs
dstack logs my-experiment

# Check resource usage
dstack metrics my-experiment

# Stop a run
dstack stop my-experiment -y
```

## Managing Results

### Pull Results

Download completed experiments from HuggingFace:

```bash
slm-lab pull ppo_atari
```

This downloads to `data/` for local analysis.

### List Experiments

```bash
slm-lab list
```

Shows all experiments stored on your HuggingFace repo.

### Push Local Results

Upload a local experiment:

```bash
slm-lab push data/ppo_atari_2026_01_30_221924
```

## Configuration

### Fleet Setup (dstack 0.20+)

For dstack 0.20+, create a fleet before running tasks:

```bash
dstack apply -f .dstack/fleet-gpu.yml
```

### Hardware Selection

Edit `.dstack/run-gpu-train.yml` to customize hardware:

```yaml
resources:
  gpu: L4
  memory: 16GB
```

Available GPU types depend on your dstack backends (AWS, GCP, etc.).

### Fractional GPU Sharing

For hyperparameter search, run multiple trials on a single GPU using fractional allocation:

```json
"meta": {
  "search_resources": {"cpu": 1, "gpu": 0.125}
}
```

With `gpu: 0.125`, you can run **8 trials in parallel** on a single GPU. This is ideal for ASHA search where many trials run briefly before early termination.

{% hint style="info" %}
**Cost tip:** GPU instances ($0.39/hr L4) are often cheaper than equivalent CPU instances ($0.54/hr 16-CPU) due to fractional GPU sharing. Always use `--gpu` unless your algorithm is CPU-bound.
{% endhint %}

### Max Duration

All runs have a 4-hour safeguard (`max_duration: 4h`) to prevent runaway costs. Edit the dstack YAML files to adjust.

## Example Workflow

```bash
# 1. Source credentials
source .env

# 2. Launch experiment
slm-lab run-remote --gpu slm_lab/spec/benchmark/ppo/ppo_hopper.json ppo_hopper train -n ppo-hopper

# 3. Monitor progress
dstack ps
dstack logs ppo-hopper

# 4. When complete, pull results
slm-lab pull ppo_hopper

# 5. Analyze locally
ls data/ppo_hopper_*/
```

## Efficient Batch Running

For systematic benchmarking, maximize GPU utilization by running multiple experiments:

```bash
# Launch multiple runs in parallel
source .env
slm-lab run-remote --gpu spec1.json spec1 train -n run1
slm-lab run-remote --gpu spec2.json spec2 train -n run2
slm-lab run-remote --gpu spec3.json spec3 train -n run3

# Monitor all runs
dstack ps
```

For ASHA search with 8 trials per GPU, you can run **~30 concurrent trials** on 4 GPUs.

{% hint style="success" %}
**Workflow tip:** Launch runs, monitor with `dstack ps` and `dstack logs`, pull completed results with `slm-lab pull`, then immediately launch the next batch. Don't wait idle—iterate quickly on failures.
{% endhint %}

## Troubleshooting

### Run fails to start

Check dstack status and ensure your fleet is ready:

```bash
dstack ps
dstack fleet list
```

### Results not uploading

Ensure `HF_TOKEN` and `HF_REPO` are set in `.env` and sourced.

### GPU not available

Try a different GPU type in `.dstack/run-gpu-train.yml` or wait for availability.

### Low GPU utilization

Check resource usage to identify bottlenecks:

```bash
dstack metrics my-experiment
```

Low GPU utilization often indicates:
- **Environment stepping is slow** - increase `num_envs` or use GPU-accelerated environments
- **Batch size too small** - increase `minibatch_size`
- **Config mismatch** - verify spec settings match the environment category

### Comparing with Reference Implementations

If results differ significantly from expected, compare against reference implementations:
- [CleanRL](https://github.com/vwxyzjn/cleanrl) - single-file implementations
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) - production-ready RL

Check hyperparameters, normalization settings, and reward scaling.
