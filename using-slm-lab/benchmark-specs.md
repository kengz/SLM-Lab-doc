# Benchmark Specs 📋

This page lists all benchmark spec files for reproducing SLM Lab results. For benchmark scores and trained models, see [Benchmark Results](../benchmark-results/discrete-benchmark.md).

## Reproducing Benchmark Results

The authoritative source for benchmark methodology is [BENCHMARKS.md](https://github.com/kengz/SLM-Lab/blob/master/docs/BENCHMARKS.md) in the SLM-Lab repository. It contains:

- Exact commands to reproduce each result
- Standardized environment settings for fair comparison
- Links to trained models on HuggingFace
- Training curves and scores

### Quick Reproduction

```bash
# 1. Find the spec in tables below
# 2. Run with train mode
slm-lab run SPEC_FILE SPEC_NAME train

# Example: PPO on CartPole
slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole train
```

### Download Trained Models

```bash
# List available experiments
slm-lab list

# Download a specific experiment
slm-lab pull ppo_cartpole

# Replay trained model
slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole enjoy
```

## Spec File Organization

All specs are in [slm_lab/spec/benchmark/](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/spec/benchmark), organized by algorithm:

```
slm_lab/spec/benchmark/
├── reinforce/     # REINFORCE specs
├── sarsa/         # SARSA specs
├── dqn/           # DQN and DDQN+PER specs
├── a2c/           # A2C specs
├── ppo/           # PPO specs
└── sac/           # SAC specs
```

## By Algorithm

### REINFORCE / SARSA

Simple algorithms for learning fundamentals. Classic Control only.

| Algorithm | Environment | Spec File | Spec Name |
|-----------|-------------|-----------|-----------|
| REINFORCE | CartPole | [reinforce_cartpole.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/reinforce/reinforce_cartpole.json) | `reinforce_cartpole` |
| SARSA | CartPole | [sarsa_cartpole.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sarsa/sarsa_cartpole.json) | `sarsa_boltzmann_cartpole` |

### DQN Family

Value-based algorithms for discrete action spaces.

| Environment | DQN Spec | DDQN+PER Spec |
|-------------|----------|---------------|
| CartPole | [dqn_cartpole.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/dqn/dqn_cartpole.json) | (same file, `ddqn_per_boltzmann_cartpole`) |
| Acrobot | [dqn_acrobot.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/dqn/dqn_acrobot.json) | [ddqn_per_acrobot.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/dqn/ddqn_per_acrobot.json) |
| LunarLander | [dqn_lunar.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/dqn/dqn_lunar.json) | [ddqn_per_lunar.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/dqn/ddqn_per_lunar.json) |

### A2C

On-policy actor-critic with synchronized updates.

| Environment | Spec File | Spec Name |
|-------------|-----------|-----------|
| CartPole | [a2c_gae_cartpole.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/a2c/a2c_gae_cartpole.json) | `a2c_gae_cartpole` |
| Acrobot | [a2c_gae_acrobot.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/a2c/a2c_gae_acrobot.json) | `a2c_gae_acrobot` |
| Pendulum | [a2c_gae_pendulum.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/a2c/a2c_gae_pendulum.json) | `a2c_gae_pendulum` |
| LunarLander | [a2c_gae_lunar.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/a2c/a2c_gae_lunar.json) | `a2c_gae_lunar` |
| BipedalWalker | [a2c_gae_bipedalwalker.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/a2c/a2c_gae_bipedalwalker.json) | `a2c_gae_bipedalwalker` |

### PPO

Proximal Policy Optimization—robust across all environment types.

**Classic Control & Box2D:**

| Environment | Spec File | Spec Name |
|-------------|-----------|-----------|
| CartPole | [ppo_cartpole.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_cartpole.json) | `ppo_cartpole` |
| Acrobot | [ppo_acrobot.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_acrobot.json) | `ppo_acrobot` |
| Pendulum | [ppo_pendulum.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_pendulum.json) | `ppo_pendulum` |
| LunarLander | [ppo_lunar.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_lunar.json) | `ppo_lunar` |
| BipedalWalker | [ppo_bipedalwalker.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_bipedalwalker.json) | `ppo_bipedalwalker` |

**MuJoCo** (individual tuned specs):

| Environment | Spec File | Spec Name |
|-------------|-----------|-----------|
| Hopper | [ppo_hopper.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_hopper.json) | `ppo_hopper` |
| Swimmer | [ppo_swimmer.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_swimmer.json) | `ppo_swimmer` |
| Ant | [ppo_ant.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_ant.json) | `ppo_ant` |
| InvertedPendulum | [ppo_inverted_pendulum.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_inverted_pendulum.json) | `ppo_inverted_pendulum` |
| InvertedDoublePendulum | [ppo_inverted_double_pendulum.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_inverted_double_pendulum.json) | `ppo_inverted_double_pendulum` |

**MuJoCo** (template specs with `-s env=...`):

| Spec File | Spec Name | Best For |
|-----------|-----------|----------|
| [ppo_mujoco.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_mujoco.json) | `ppo_mujoco` | HalfCheetah, Walker2d, Humanoid, HumanoidStandup |
| [ppo_mujoco.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_mujoco.json) | `ppo_mujoco_longhorizon` | Reacher, Pusher |

```bash
# Example: PPO on HalfCheetah using template
slm-lab run -s env=HalfCheetah-v5 -s max_frame=10e6 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco train
```

**Atari** (template spec with `-s env=...`):

| Spec File | Spec Name | Lambda | Best For |
|-----------|-----------|--------|----------|
| [ppo_atari.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_atari.json) | `ppo_atari` | 0.95 | Strategic games (default) |
| [ppo_atari.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_atari.json) | `ppo_atari_lam85` | 0.85 | Mixed games |
| [ppo_atari.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_atari.json) | `ppo_atari_lam70` | 0.70 | Action games |

```bash
# Example: PPO on Breakout
slm-lab run -s env=ALE/Breakout-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari_lam70 train
```

See [Atari Benchmark](../benchmark-results/atari-benchmark.md) for all 54 games and optimal lambda values.

### SAC

Soft Actor-Critic—off-policy algorithm, best for continuous control.

**Classic Control & Box2D:**

| Environment | Spec File | Spec Name |
|-------------|-----------|-----------|
| CartPole | [sac_cartpole.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_cartpole.json) | `sac_cartpole` |
| Acrobot | [sac_acrobot.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_acrobot.json) | `sac_acrobot` |
| Pendulum | [sac_pendulum.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_pendulum.json) | `sac_pendulum` |
| LunarLander | [sac_lunar.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_lunar.json) | `sac_lunar` |
| BipedalWalker | [sac_bipedalwalker.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_bipedalwalker.json) | `sac_bipedalwalker` |

**MuJoCo:**

| Environment | Spec File | Spec Name |
|-------------|-----------|-----------|
| HalfCheetah | [sac_halfcheetah.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_halfcheetah.json) | `sac_halfcheetah` |
| Hopper | [sac_hopper.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_hopper.json) | `sac_hopper` |
| Ant | [sac_ant.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_ant.json) | `sac_ant` |
| Humanoid | [sac_humanoid.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_humanoid.json) | `sac_humanoid` |
| HumanoidStandup | [sac_humanoid_standup.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_humanoid_standup.json) | `sac_humanoid_standup` |
| InvertedPendulum | [sac_inverted_pendulum.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_inverted_pendulum.json) | `sac_inverted_pendulum` |
| InvertedDoublePendulum | [sac_inverted_double_pendulum.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_inverted_double_pendulum.json) | `sac_inverted_double_pendulum` |
| Pusher | [sac_pusher.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_pusher.json) | `sac_pusher` |

## By Environment Category

Quick reference for which algorithms have specs for each environment type:

| Environment | REINFORCE | SARSA | DQN | DDQN+PER | A2C | PPO | SAC |
|-------------|-----------|-------|-----|----------|-----|-----|-----|
| **Classic Control** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Box2D** | — | — | ✓ | ✓ | ✓ | ✓ | ✓ |
| **MuJoCo** | — | — | — | — | — | ✓ | ✓ |
| **Atari** | — | — | — | — | ✓ | ✓ | — |

## Template Specs

Template specs use `${env}` and `${max_frame}` placeholders for flexibility:

```javascript
"env": {
  "name": "${env}",           // Substituted with -s env=...
  "max_frame": "${max_frame}" // Substituted with -s max_frame=...
}
```

### Usage

```bash
# Substitute variables with -s flag
slm-lab run -s env=HalfCheetah-v5 -s max_frame=4e6 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco train

# Multiple substitutions
slm-lab run -s env=ALE/Pong-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari train
```

### Available Templates

| Spec | Variables | Environments |
|------|-----------|--------------|
| `ppo_mujoco.json` | `env`, `max_frame` | All 11 MuJoCo environments |
| `ppo_atari.json` | `env` | All 54 Atari games |
| `a2c_gae_mujoco.json` | `env`, `max_frame` | MuJoCo environments |
| `a2c_gae_atari.json` | `env` | Atari games |

## Performance Results

For scores, training curves, and trained models:

- [Discrete Benchmark](../benchmark-results/discrete-benchmark.md) — Classic Control, Box2D
- [Continuous Benchmark](../benchmark-results/continuous-benchmark.md) — MuJoCo
- [Atari Benchmark](../benchmark-results/atari-benchmark.md) — 54 Atari games

{% hint style="info" %}
**Authoritative source:** [BENCHMARKS.md](https://github.com/kengz/SLM-Lab/blob/master/docs/BENCHMARKS.md) in the repository contains exact reproduction commands, standardized settings, and HuggingFace links.
{% endhint %}
