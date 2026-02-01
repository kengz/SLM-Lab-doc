# Remote Training ☁️

SLM Lab uses [dstack](https://dstack.ai/) for cloud GPU training and HuggingFace for experiment storage.

## Setup

### 1. Install dstack

```bash
uv tool install dstack
```

### 2. Configure dstack

Create an account at [dstack Sky](https://sky.dstack.ai/) and get your token:

```bash
dstack project add --name your-project --url https://sky.dstack.ai --token $DSTACK_TOKEN -y
```

This saves configuration to `~/.dstack/config.yml`.

### 3. Set up HuggingFace credentials

Create a `.env` file in your SLM-Lab directory:

```bash
HF_TOKEN=hf_xxxxxxxxxxxx
HF_REPO=your-username/slm-lab-results
```

Source before running:

```bash
source .env
```

## Running Remote Experiments

### Basic Commands

```bash
# Train on cloud GPU
source .env && slm-lab run-remote --gpu slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari train -n my-experiment

# Hyperparameter search
source .env && slm-lab run-remote --gpu slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari search -n my-search

# With variable substitution
source .env && slm-lab run-remote --gpu -s env=ALE/Qbert-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari train -n ppo-qbert
```

The `-n` flag names your run for easy identification.

### Monitoring Runs

```bash
dstack ps                    # List running jobs
dstack logs my-experiment    # View logs
dstack metrics my-experiment # Check GPU/CPU utilization
dstack stop my-experiment -y # Stop a run
```

### Checking Results

When a run completes, check the final score in logs:

```bash
dstack logs my-experiment | grep "trial_metrics"
# Output: trial_metrics: frame:1.00e+07 | total_reward_ma:15094 | ...
```

The `total_reward_ma` is the final moving average score.

## Managing Results

### Pull Results

Download completed experiments from HuggingFace:

```bash
slm-lab pull ppo_atari
```

### List Experiments

```bash
slm-lab list
```

### Push Local Results

```bash
slm-lab push data/ppo_atari_2026_01_30_221924
```

## Configuration

### Hardware

SLM Lab defaults to **L4 GPU** ($0.39/hr) which handles all benchmark environments. The configuration is in `.dstack/run-gpu-train.yml`.

For very large models or faster training, you can switch to V100:

```yaml
resources:
  gpu: V100
```

{% hint style="info" %}
**Cost tip:** GPU instances are often cheaper than equivalent CPU instances due to fractional GPU sharing. Always use `--gpu` unless your workload is CPU-bound.
{% endhint %}

### Fractional GPU for Search

In search mode, multiple trials share one GPU:

```json
"meta": {
  "search_resources": {"cpu": 1, "gpu": 0.125}
}
```

With `gpu: 0.125`, **8 trials run in parallel** on a single GPU—ideal for ASHA search.

### Max Duration

Runs have safeguards to prevent runaway costs: CPU runs are limited to 4 hours, GPU runs to 6 hours. Edit `.dstack/*.yml` to adjust.

### Fleet Setup (dstack 0.20+)

For dstack 0.20+, create a fleet before running:

```bash
dstack apply -f .dstack/fleet-gpu.yml
```

## Workflow Example

```bash
# 1. Source credentials
source .env

# 2. Launch experiment
slm-lab run-remote --gpu slm_lab/spec/benchmark/ppo/ppo_hopper.json ppo_hopper train -n ppo-hopper

# 3. Monitor
dstack ps
dstack logs ppo-hopper

# 4. When complete, check score
dstack logs ppo-hopper | grep "trial_metrics"

# 5. Pull results
slm-lab pull ppo_hopper

# 6. Analyze locally
ls data/ppo_hopper_*/
```

## Batch Running

Launch multiple experiments to maximize GPU utilization:

```bash
source .env
slm-lab run-remote --gpu -s env=ALE/Qbert-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari train -n qbert
slm-lab run-remote --gpu -s env=ALE/MsPacman-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari_lam85 train -n mspacman
slm-lab run-remote --gpu -s env=ALE/Breakout-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari_lam70 train -n breakout

dstack ps  # Monitor all
```

{% hint style="success" %}
**Iterate quickly:** Launch runs, monitor with `dstack ps`, pull completed results, launch the next batch. Don't wait idle.
{% endhint %}

## Troubleshooting

### Run fails to start

```bash
dstack ps
dstack fleet list
```

Check fleet status and GPU availability.

### Results not uploading

Ensure `HF_TOKEN` and `HF_REPO` are set in `.env` and sourced.

### Low GPU utilization

```bash
dstack metrics my-experiment
```

Low GPU util often means:
- **Environment stepping is slow** - increase `num_envs`
- **Batch size too small** - increase `minibatch_size`
- **Config issue** - verify spec settings

## More Resources

- [dstack Documentation](https://dstack.ai/docs/) - Full dstack reference
- [BENCHMARKS.md](https://github.com/kengz/SLM-Lab/blob/master/docs/BENCHMARKS.md) - Benchmark methodology and commands
