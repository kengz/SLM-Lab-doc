# Public Benchmark Data 📊

## Overview

All SLM Lab benchmark results are publicly available on HuggingFace for reproducibility and comparison:

{% embed url="https://huggingface.co/datasets/SLM-Lab/benchmark" %}

Each experiment includes:

- **Trained models** - PyTorch checkpoints (`*_ckpt-best.pt`)
- **Training curves** - Full learning history (`*_session_df.csv`)
- **Specs** - Exact configurations for reproduction (`*_spec.json`)
- **Graphs** - Plotly visualizations (PNG and HTML)

## Accessing Results 🔗

### List Available Experiments

```bash
slm-lab list
```

Shows all experiments on [HuggingFace](https://huggingface.co/datasets/SLM-Lab/benchmark).

### Download an Experiment

```bash
slm-lab pull ppo_hopper
```

Downloads to `data/ppo_hopper_*/` including:
- Model checkpoints (`model/*_ckpt-best.pt`)
- Training metrics (`info/*_session_df.csv`)
- Saved spec (`*_spec.json`)

### Replay a Trained Agent

```bash
slm-lab run _ _ enjoy@data/ppo_hopper_*/ppo_hopper_t0_spec.json
```

### Browse on HuggingFace

Direct links to experiment folders (example):

- [ppo_cartpole_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_cartpole_2026_01_30_221924)
- [ppo_hopper_2026_01_31](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_hopper_2026_01_31_105438)
- [ppo_atari_breakout_2026_01_07](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_breakout_2026_01_07_110559)

See the benchmark pages for complete lists:
- [Discrete Benchmark](discrete-benchmark.md) - Classic Control & Box2D
- [Continuous Benchmark](continuous-benchmark.md) - MuJoCo
- [Atari Benchmark](atari-benchmark.md) - 54 Atari games

## v5 Benchmark Coverage

### Environments Tested

| Category | Environments | Algorithms |
|----------|--------------|------------|
| **Classic Control** | CartPole-v1, Acrobot-v1, Pendulum-v1 | REINFORCE, SARSA, DQN, DDQN+PER, A2C, PPO, SAC |
| **Box2D** | LunarLander-v3 (discrete & continuous) | DQN, DDQN+PER, A2C, PPO, SAC |
| **MuJoCo** | 11 environments (Hopper, HalfCheetah, etc.) | PPO |
| **Atari** | 54 games | PPO (3 lambda variants) |

### Quick Links

| Benchmark | Page | Environments |
|-----------|------|--------------|
| Classic + Box2D | [Discrete Benchmark](discrete-benchmark.md) | CartPole, Acrobot, Pendulum, LunarLander |
| MuJoCo | [Continuous Benchmark](continuous-benchmark.md) | Hopper, HalfCheetah, Humanoid, etc. |
| Atari | [Atari Benchmark](atari-benchmark.md) | 54 games |

## Methodology

### How Scores Are Reported

Results show **Trial-level** performance:

1. **Trial** = 4 Sessions with different random seeds
2. **Session** = One complete training run
3. **Score** = Final 100-checkpoint moving average (`total_reward_ma`)

The trial score is the mean across 4 sessions, providing statistically meaningful results.

### Training Details

| Setting | Value |
|---------|-------|
| Sessions per trial | 4 (different random seeds) |
| Checkpoint frequency | Varies by env (see below) |
| Moving average window | 100 checkpoints |
| Hardware | Cloud GPUs (L4/A10G via dstack) |

### Environment Settings

Standardized settings for fair comparison across environment categories:

| Category | num_envs | max_frame | log_frequency | ASHA grace_period |
|----------|----------|-----------|---------------|-------------------|
| Classic Control | 4 | 2e5-3e5 | 500 | 1e4 |
| Box2D | 8 | 3e5 | 1000 | 5e4 |
| MuJoCo | 16 | 4e6-10e6 | 10000 | 1e5-1e6 |
| Atari | 16 | 10e6 | 10000 | 5e5 |

The `grace_period` is the minimum frames before ASHA can terminate underperforming trials. Set it high enough for meaningful learning signal (typically 5-10% of max_frame).

### Contributing Benchmark Results

When adding or updating benchmarks:

1. **Audit spec settings**: Ensure your `spec.json` matches the Settings line in the benchmark table
2. **Run and commit**: Execute the benchmark, then commit the spec file to the repo
3. **Record scores**: Extract `total_reward_ma` from logs and add HuggingFace folder link
4. **Generate plots**: Use `slm-lab plot -t "EnvName" -f folder1,folder2,...`

{% hint style="info" %}
Only use final validation runs (not search results) for benchmark tables. Search is for hyperparameter discovery; validation confirms with committed specs.
{% endhint %}

### Reproducibility

Every experiment can be exactly reproduced:

```bash
# 1. Download the experiment
slm-lab pull ppo_hopper

# 2. Check the spec for settings and git SHA
cat data/ppo_hopper_*/ppo_hopper_t0_spec.json

# 3. Run with saved spec
slm-lab run _ _ train@data/ppo_hopper_*/ppo_hopper_t0_spec.json
```

For exact code version, checkout the git SHA in the spec file.

## Historical Data

### v4 Results (Google Drive)

v4 benchmarks used OpenAI Gym and Roboschool (both deprecated). Available for historical reference:

- [All benchmark data](https://drive.google.com/drive/folders/1fUB3jRvXr8ySZMSW5w0GPWJe3QmM7tb3?usp=sharing)

{% hint style="warning" %}
**Not directly comparable:** v4 and v5 use different environment versions with different reward scales and physics. See [Changelog](../CHANGELOG.md) for migration details.
{% endhint %}

## Using Your Own HuggingFace Repo

Set up credentials in `.env`:

```bash
HF_TOKEN=hf_xxxxxxxxxxxx
HF_REPO=your-username/your-repo
```

Then push your results:

```bash
source .env
slm-lab push data/my_experiment_2026_01_30_221924
```

## Terminology

| Abbreviation | Meaning |
|--------------|---------|
| A2C | Advantage Actor-Critic |
| DDQN | Double Deep Q-Network |
| DQN | Deep Q-Network |
| GAE | Generalized Advantage Estimation |
| PER | Prioritized Experience Replay |
| PPO | Proximal Policy Optimization |
| SAC | Soft Actor-Critic |
| CER | Combined Experience Replay |
| MA | Moving Average |

## Contributing Benchmarks

To contribute new benchmark results:

1. Run experiments with `--upload-hf` flag (or `source .env` for auto-upload)
2. Ensure `HF_TOKEN` and `HF_REPO` are configured
3. Results automatically upload to your HuggingFace repo

For official SLM Lab benchmarks, see [docs/BENCHMARKS.md](https://github.com/kengz/SLM-Lab/blob/master/docs/BENCHMARKS.md) in the code repository.
