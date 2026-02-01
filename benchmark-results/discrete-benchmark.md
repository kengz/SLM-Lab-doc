# Discrete Environment Benchmark 🎯

## Classic Control & Box2D Results (v5)

SLM Lab v5 validates algorithms on [Gymnasium](https://gymnasium.farama.org/) discrete environments. These benchmarks cover:

- **[Classic Control](https://gymnasium.farama.org/environments/classic_control/)**: CartPole, Acrobot, Pendulum—simple physics tasks ideal for algorithm validation
- **[Box2D](https://gymnasium.farama.org/environments/box2d/)**: LunarLander—2D physics with more complex dynamics

Results below are from January 2026 benchmark reruns using Gymnasium v5 environments.

All trained models and metrics are publicly available on [HuggingFace](https://huggingface.co/datasets/SLM-Lab/benchmark).

{% hint style="warning" %}
**v5 vs v4 Difficulty:** Gymnasium environments have stricter termination and reward handling:
- **LunarLander-v3** is notably harder than v2—stricter landing criteria, lower typical scores
- **Pendulum-v1** uses different reward scaling than v0
- Expect **5-15% lower scores** compared to OpenAI Gym benchmarks

See [Gymnasium docs](https://gymnasium.farama.org/) for environment-specific changes.
{% endhint %}

### Classic Control

#### CartPole-v1

**Target**: reward MA > 400 | **Settings**: max_frame 2e5 | num_envs 4 | max_session 4

| Algorithm | Status | MA | Spec | HuggingFace |
|-----------|--------|-----|------|-------------|
| REINFORCE | ✅ | 469.7 | [reinforce_cartpole.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/reinforce/reinforce_cartpole.json) | [reinforce_cartpole_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/reinforce_cartpole_2026_01_30_215510) |
| SARSA | ✅ | 421.6 | [sarsa_cartpole.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sarsa/sarsa_cartpole.json) | [sarsa_boltzmann_cartpole_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sarsa_boltzmann_cartpole_2026_01_30_215508) |
| DQN | ⚠️ | 188.1 | [dqn_cartpole.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/dqn/dqn_cartpole.json) | [dqn_boltzmann_cartpole_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/dqn_boltzmann_cartpole_2026_01_30_215213) |
| DDQN+PER | ✅ | 432.9 | [dqn_cartpole.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/dqn/dqn_cartpole.json) | [ddqn_per_boltzmann_cartpole_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ddqn_per_boltzmann_cartpole_2026_01_30_215454) |
| A2C | ✅ | 499.7 | [a2c_gae_cartpole.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/a2c/a2c_gae_cartpole.json) | [a2c_gae_cartpole_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_cartpole_2026_01_30_215337) |
| PPO | ✅ | 499.5 | [ppo_cartpole.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_cartpole.json) | [ppo_cartpole_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_cartpole_2026_01_30_221924) |
| SAC | ⚠️ | 359.7 | [sac_cartpole.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_cartpole.json) | [sac_cartpole_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_cartpole_2026_01_30_221934) |

![PPO CartPole Training Curve](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/data/ppo_cartpole_2026_01_30_221924/ppo_cartpole_t0_trial_graph_mean_returns_ma_vs_frames.png)

#### Acrobot-v1

**Target**: reward MA > -100 | **Settings**: max_frame 3e5 | num_envs 4 | max_session 4

| Algorithm | Status | MA | Spec | HuggingFace |
|-----------|--------|-----|------|-------------|
| DQN | ✅ | -94.8 | [dqn_acrobot.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/dqn/dqn_acrobot.json) | [dqn_boltzmann_acrobot_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/dqn_boltzmann_acrobot_2026_01_30_215429) |
| DDQN+PER | ✅ | -85.2 | [ddqn_per_acrobot.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/dqn/ddqn_per_acrobot.json) | [ddqn_per_acrobot_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ddqn_per_acrobot_2026_01_30_215436) |
| A2C | ✅ | -83.8 | [a2c_gae_acrobot.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/a2c/a2c_gae_acrobot.json) | [a2c_gae_acrobot_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_acrobot_2026_01_30_215413) |
| PPO | ✅ | -81.4 | [ppo_acrobot.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_acrobot.json) | [ppo_acrobot_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_acrobot_2026_01_30_215352) |
| SAC | ✅ | -97.1 | [sac_acrobot.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_acrobot.json) | [sac_acrobot_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_acrobot_2026_01_30_215401) |

![PPO Acrobot Training Curve](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/data/ppo_acrobot_2026_01_30_215352/ppo_acrobot_t0_trial_graph_mean_returns_ma_vs_frames.png)

#### Pendulum-v1

**Target**: reward MA > -200 | **Settings**: max_frame 3e5 | num_envs 4 | max_session 4

| Algorithm | Status | MA | Spec | HuggingFace |
|-----------|--------|-----|------|-------------|
| A2C | ❌ | -553 | [a2c_gae_pendulum.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/a2c/a2c_gae_pendulum.json) | [a2c_gae_pendulum_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_pendulum_2026_01_30_215421) |
| PPO | ✅ | -168.3 | [ppo_pendulum.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_pendulum.json) | [ppo_pendulum_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_pendulum_2026_01_30_215944) |
| SAC | ✅ | -152.3 | [sac_pendulum.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_pendulum.json) | [sac_pendulum_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_pendulum_2026_01_30_215454) |

![PPO Pendulum Training Curve](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/data/ppo_pendulum_2026_01_30_215944/ppo_pendulum_t0_trial_graph_mean_returns_ma_vs_frames.png)

### Box2D

#### LunarLander-v3 (Discrete)

**Target**: reward MA > 200 | **Settings**: max_frame 3e5 | num_envs 8 | max_session 4

| Algorithm | Status | MA | Spec | HuggingFace |
|-----------|--------|-----|------|-------------|
| DQN | ⚠️ | 183.6 | [dqn_lunar.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/dqn/dqn_lunar.json) | [dqn_concat_lunar_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/dqn_concat_lunar_2026_01_30_215529) |
| DDQN+PER | ✅ | 261.5 | [ddqn_per_lunar.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/dqn/ddqn_per_lunar.json) | [ddqn_per_concat_lunar_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ddqn_per_concat_lunar_2026_01_30_215532) |
| A2C | ❌ | 9.5 | [a2c_gae_lunar.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/a2c/a2c_gae_lunar.json) | [a2c_gae_lunar_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_lunar_2026_01_30_215529) |
| PPO | ⚠️ | 159.0 | [ppo_lunar.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_lunar.json) | [ppo_lunar_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_lunar_2026_01_30_215550) |
| SAC | ❌ | -75.4 | [sac_lunar.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_lunar.json) | [sac_lunar_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_lunar_2026_01_30_215552) |

![DDQN+PER LunarLander Training Curve](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/data/ddqn_per_concat_lunar_2026_01_30_215532/ddqn_per_concat_lunar_t0_trial_graph_mean_returns_ma_vs_frames.png)

#### LunarLander-v3 (Continuous)

**Target**: reward MA > 200 | **Settings**: max_frame 3e5 | num_envs 8 | max_session 4

| Algorithm | Status | MA | Spec | HuggingFace |
|-----------|--------|-----|------|-------------|
| A2C | ❌ | -38.2 | [a2c_gae_lunar.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/a2c/a2c_gae_lunar.json) | [a2c_gae_lunar_continuous_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_lunar_continuous_2026_01_30_215630) |
| PPO | ⚠️ | 165.5 | [ppo_lunar.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_lunar.json) | [ppo_lunar_continuous_2026_01_31](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_lunar_continuous_2026_01_31_104549) |
| SAC | ✅ | 208.6 | [sac_lunar.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_lunar.json) | [sac_lunar_continuous_2026_01_31](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_lunar_continuous_2026_01_31_104537) |

![SAC LunarLander Continuous Training Curve](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/data/sac_lunar_continuous_2026_01_31_104537/sac_lunar_continuous_t0_trial_graph_mean_returns_ma_vs_frames.png)

**Legend:** ✅ Solved | ⚠️ Close (>80%) | ❌ Failed

{% hint style="info" %}
**v5 vs v4 Environment Differences:** Gymnasium environments have stricter termination conditions and different reward scales than OpenAI Gym. LunarLander-v3 is notably harder than v2. See [Gymnasium docs](https://gymnasium.farama.org/) for details.
{% endhint %}

### Running Discrete Benchmarks

```bash
# PPO on CartPole
slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole train

# DDQN+PER on LunarLander
slm-lab run slm_lab/spec/benchmark/dqn/ddqn_per_lunar.json ddqn_per_concat_lunar train

# SAC on Pendulum
slm-lab run slm_lab/spec/benchmark/sac/sac_pendulum.json sac_pendulum train
```

### Download and Replay

```bash
# List all available experiments
slm-lab list

# Download a specific experiment
slm-lab pull ppo_cartpole

# Replay the trained agent
slm-lab run _ _ enjoy@data/ppo_cartpole_*/ppo_cartpole_t0_spec.json
```

## Historical Results (v4)

<details>
<summary><b>OpenAI Gym Results (v4)</b> - click to expand</summary>

{% hint style="info" %}
These results from SLM Lab v4 used OpenAI Gym environments (now deprecated). Environment versions differ from current Gymnasium versions. Unity environments are no longer included in the core package.
{% endhint %}

* [Upload PR #427](https://github.com/kengz/SLM-Lab/pull/427)
* [Google Drive data](https://drive.google.com/file/d/1lb9Hn22Uzb67ndotRULiUspk3TnmKp6Z/view?usp=sharing)

| Env. \ Alg. | DQN | DDQN+PER | A2C (GAE) | A2C (n-step) | PPO | SAC |
|-------------|-----|----------|-----------|--------------|-----|-----|
| Breakout | 80.88 | 182 | 377 | 398 | **443** | 3.51* |
| Pong | 18.48 | 20.5 | 19.31 | 19.56 | **20.58** | 19.87* |
| Qbert | 5494 | 11426 | 12405 | **13590** | 13460 | 923* |
| Seaquest | 1185 | **4405** | 1070 | 1684 | 1715 | 171* |
| LunarLander | 192 | 233 | 25.21 | 68.23 | 214 | **276** |

> Episode score at the end of training. Reported scores are the average over the last 100 checkpoints, averaged over 4 Sessions. Results marked with `*` used async SAC.

</details>

For the full Atari benchmark, see [Atari Benchmark](atari-benchmark.md).
