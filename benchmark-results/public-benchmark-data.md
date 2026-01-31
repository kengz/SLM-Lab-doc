# Public Benchmark Data

## Overview

SLM Lab publishes benchmark results for reproducibility and comparison. All results include:

- **Trained models** - PyTorch checkpoints you can load and evaluate
- **Training curves** - Full learning history (not just final scores)
- **Specs** - Exact configurations for reproduction
- **Git SHA** - Code version used

## Accessing Results

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

## v5 Benchmark Coverage

### Environments Tested

| Category | Environments | Algorithms |
|----------|--------------|------------|
| **Classic Control** | CartPole-v1, Acrobot-v1, Pendulum-v1 | All |
| **Box2D** | LunarLander-v3, BipedalWalker-v3 | DQN, PPO, SAC |
| **MuJoCo** | 11 environments (Hopper, HalfCheetah, etc.) | PPO |
| **Atari** | 54 games | PPO |

### Quick Links

| Benchmark | Page | Environments |
|-----------|------|--------------|
| Classic + Box2D | [Discrete Benchmark](discrete-benchmark.md) | CartPole, Acrobot, LunarLander |
| MuJoCo | [Continuous Benchmark](continuous-benchmark.md) | Hopper, HalfCheetah, Humanoid |
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
| Checkpoint frequency | Every 10,000 frames |
| Moving average window | 100 checkpoints |
| Hardware | Cloud GPUs (L4/A10G via dstack) |

### Reproducibility

Every experiment can be exactly reproduced:

```bash
# 1. Check out the exact code version (git SHA in spec file)
git checkout <sha-from-spec>

# 2. Run with saved spec
slm-lab run _ _ train@path/to/spec.json
```

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
slm-lab push data/my_experiment_2024_01_15_123456
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

## Contributing Benchmarks

To contribute new benchmark results:

1. Run experiments with `--upload-hf` flag
2. Ensure `HF_TOKEN` and `HF_REPO` are configured
3. Results automatically upload to your HuggingFace repo

For official SLM Lab benchmarks, see [docs/BENCHMARKS.md](https://github.com/kengz/SLM-Lab/blob/master/docs/BENCHMARKS.md) in the code repository.
