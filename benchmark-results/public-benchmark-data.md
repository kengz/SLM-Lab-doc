# Public Benchmark Data 📊

## Overview

All SLM Lab benchmark results are publicly available on HuggingFace:

{% embed url="https://huggingface.co/datasets/SLM-Lab/benchmark" %}

Each experiment includes:

- **Trained models** — PyTorch checkpoints (`*_ckpt-best_net_model.pt`)
- **Training curves** — Full learning history (`*_session_df_{train,eval}.csv`)
- **Specs** — Exact configurations for reproduction (`*_spec.yaml`)
- **Graphs** — Plotly visualizations (PNG and HTML)

## Algorithm Coverage

Which algorithms are benchmarked in each environment category. ✓ = benchmarked.

| Algorithm | Classic Control | Box2D | MuJoCo | Atari |
|-----------|:-:|:-:|:-:|:-:|
| **REINFORCE** | ✓ | | | |
| **SARSA** | ✓ | | | |
| **DQN** | ✓ | ✓ | | |
| **DDQN+PER** | ✓ | ✓ | | |
| **A2C** | ✓ | ✓ | | ✓ |
| **PPO** | ✓ | ✓ | ✓ | ✓ |
| **SAC** | ✓ | ✓ | ✓ | ✓ |

| Category | Environments | Algorithms |
|----------|--------------|------------|
| **Classic Control** | CartPole-v1, Acrobot-v1, Pendulum-v1 | REINFORCE, SARSA, DQN, DDQN+PER, A2C, PPO, SAC |
| **Box2D** | LunarLander-v3 (discrete & continuous) | DQN, DDQN+PER, A2C, PPO, SAC |
| **MuJoCo** | 11 environments (Hopper, HalfCheetah, Humanoid, etc.) | PPO, SAC |
| **Atari** | 57 games (6 hard-exploration skipped) | A2C, PPO, SAC |

### Detailed Results

| Benchmark | Page | Environments |
|-----------|------|--------------|
| Classic + Box2D | [Discrete Benchmark](discrete-benchmark.md) | CartPole, Acrobot, Pendulum, LunarLander |
| MuJoCo | [Continuous Benchmark](continuous-benchmark.md) | Hopper, HalfCheetah, Humanoid, etc. |
| Atari | [Atari Benchmark](atari-benchmark.md) | 57 games |

## Accessing Results

### List and Download

```bash
# Set HuggingFace repo
echo 'HF_REPO=SLM-Lab/benchmark' >> .env

# List all experiments
source .env && slm-lab list

# Download an experiment
source .env && slm-lab pull ppo_hopper
```

{% hint style="info" %}
**No token needed for read-only access.** `HF_TOKEN` is only required for uploading to your own repo.
{% endhint %}

### Replay a Trained Agent

```bash
slm-lab run SPEC_FILE SPEC_NAME enjoy@data/FOLDER/SPEC_NAME_t0_spec.yaml
```

### Browse on HuggingFace

Direct links to experiment folders (example):

- [ppo_cartpole_arc_2026_02_11](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_cartpole_arc_2026_02_11_144029)
- [ppo_hopper_arc_2026_01_31](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_hopper_2026_01_31_105438)

## Methodology

### How Scores Are Reported

1. **Trial** = 4 Sessions with different random seeds
2. **Session** = One complete training run
3. **Score** = Final 100-checkpoint moving average (`total_reward_ma`)

The trial score is the mean across 4 sessions.

### Environment Settings

Standardized settings for fair comparison:

| Category | num_envs | max_frame | log_frequency | ASHA grace_period |
|----------|----------|-----------|---------------|-------------------|
| Classic Control | 4 | 2e5-3e5 | 500 | 1e4 |
| Box2D | 8 | 3e5 | 1000 | 5e4 |
| MuJoCo | 16 | 1e6-10e6 | 1e4 | 1e5-1e6 |
| Atari | 16 | 10e6 | 10000 | 5e5 |

### Hardware Requirements

| Category | GPU Required | Typical Runtime | Recommendation |
|----------|--------------|-----------------|----------------|
| Classic Control | No | Minutes | Local CPU is fine |
| Box2D | Optional | 10-30 min | Local or remote |
| MuJoCo | Yes | 1-4 hours | Use `run-remote --gpu` |
| Atari | Yes | 2-3 hours | Use `run-remote --gpu` |

{% hint style="info" %}
**Cloud GPUs recommended for MuJoCo and Atari.** Cloud L4/A10G via [dstack](https://dstack.ai) is faster and often cheaper than local training. See [Remote Training](../using-slm-lab/remote-training.md) for setup.
{% endhint %}

## Contributing Benchmarks

Follow these steps when adding or updating benchmark results.

### 1. Audit Spec Settings

Ensure your `spec.yaml` matches the **Settings** line in the [benchmark tables](https://github.com/kengz/SLM-Lab/blob/master/docs/BENCHMARKS.md). Example: `max_frame 3e5 | num_envs 4 | max_session 4 | log_frequency 500`.

### 2. Run Benchmark and Commit Specs

```bash
# Local (Classic Control: minutes)
slm-lab run SPEC_FILE SPEC_NAME train

# Remote (MuJoCo, Atari: hours)
source .env && slm-lab run-remote --gpu SPEC_FILE SPEC_NAME train -n NAME
```

Always commit the `spec.yaml` file to the repo after a successful run.

### 3. Record Scores and Plots

- Extract `total_reward_ma` from logs (`trial_metrics`)
- Add HuggingFace folder link to the benchmark table
- Generate plots:

```bash
slm-lab plot -t "CartPole-v1" -f ppo_cartpole_2026...,dqn_cartpole_2026...
```

### Hyperparameter Search

When an algorithm fails to reach target, run search before the final validation:

```bash
slm-lab run SPEC_FILE SPEC_NAME search                                        # local
source .env && slm-lab run-remote --gpu SPEC_FILE SPEC_NAME search -n NAME    # remote
```

| Stage | Mode | Config | Purpose |
|-------|------|--------|---------|
| ASHA | `search` | `max_session=1`, `search_scheduler` enabled | Wide exploration with early stopping |
| Multi | `search` | `max_session=4`, NO `search_scheduler` | Robust validation with averaging |
| Validate | `train` | Final spec | Confirmation run |

Search budget: ~3-4 trials per dimension (8 trials = 2-3 dims, 16 = 3-4 dims).

{% hint style="warning" %}
Only use final validation runs (not search results) for benchmark tables. Search is for hyperparameter discovery; validation confirms with committed specs.
{% endhint %}

## Reproducibility

Every experiment can be exactly reproduced:

```bash
# 1. Download the experiment
source .env && slm-lab pull ppo_hopper

# 2. Check the spec for settings and git SHA
cat data/ppo_hopper_arc_*/ppo_hopper_arc_t0_spec.yaml

# 3. Replay the trained model
slm-lab run slm_lab/spec/benchmark_arc/ppo/ppo_hopper_arc.yaml ppo_hopper_arc enjoy@data/ppo_hopper_arc_*/ppo_hopper_arc_t0_spec.yaml
```

For exact code version, checkout the git SHA recorded in the spec file.

## Using Your Own HuggingFace Repo

```bash
# .env
HF_TOKEN=hf_xxxxxxxxxxxx
HF_REPO=your-username/your-repo
```

```bash
source .env && slm-lab push data/my_experiment_2026_01_30_221924
```

## Historical Data (v4)

v4 benchmarks used OpenAI Gym and Roboschool (both deprecated). Available for historical reference:

- [All v4 benchmark data (Google Drive)](https://drive.google.com/drive/folders/1fUB3jRvXr8ySZMSW5w0GPWJe3QmM7tb3?usp=sharing)

{% hint style="warning" %}
**Not directly comparable:** v4 and v5 use different environment versions with different reward scales and physics. See [Changelog](../CHANGELOG.md) for details.
{% endhint %}

## Terminology

| Abbreviation | Meaning |
|--------------|---------|
| A2C | Advantage Actor-Critic |
| DDQN | Double Deep Q-Network |
| DQN | Deep Q-Network |
| GAE | Generalized Advantage Estimation |
| MA | Moving Average |
| PER | Prioritized Experience Replay |
| PPO | Proximal Policy Optimization |
| SAC | Soft Actor-Critic |
