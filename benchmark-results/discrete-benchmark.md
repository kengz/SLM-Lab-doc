# Discrete Environment Benchmark 🎯

## Classic Control & Box2D Results (v5.1)

SLM Lab v5.1 validates algorithms on [Gymnasium](https://gymnasium.farama.org/) discrete environments using the TorchArc architecture. These benchmarks cover:

- **[Classic Control](https://gymnasium.farama.org/environments/classic_control/)**: CartPole, Acrobot, Pendulum—simple physics tasks ideal for algorithm validation
- **[Box2D](https://gymnasium.farama.org/environments/box2d/)**: LunarLander—2D physics with more complex dynamics

Results below are from February 2026 benchmark runs using Gymnasium v5 environments and TorchArc specs.

All trained models and metrics are publicly available on [HuggingFace](https://huggingface.co/datasets/SLM-Lab/benchmark).

### Methodology

Results show **Trial-level** performance:

1. **Trial** = 4 Sessions with different random seeds
2. **Session** = One complete training run
3. **Score** = Final 100-checkpoint moving average (`total_reward_ma`)

The trial score is the mean across 4 sessions, providing statistically meaningful results.

### Standardized Settings

| Category | num_envs | max_frame | log_frequency | ASHA grace_period |
|----------|----------|-----------|---------------|-------------------|
| Classic Control | 4 | 2e5-3e5 | 500 | 1e4 |
| Box2D | 8 | 3e5 | 1000 | 5e4 |

The `grace_period` is the minimum frames before ASHA early stopping can terminate underperforming trials.

{% hint style="warning" %}
**v5 vs v4 Difficulty:** Gymnasium environments have stricter termination and reward handling:
- **LunarLander-v3** is notably harder than v2—stricter landing criteria, lower typical scores
- **Pendulum-v1** uses different reward scaling than v0
- Expect **5-15% lower scores** compared to OpenAI Gym benchmarks

See [Gymnasium docs](https://gymnasium.farama.org/) for environment-specific changes.
{% endhint %}

### Running Benchmarks

**Local** - runs on your machine (Classic Control completes in minutes on CPU):
```bash
slm-lab run slm_lab/spec/benchmark_arc/ppo/ppo_classic_arc.yaml ppo_cartpole_arc train
slm-lab run slm_lab/spec/benchmark_arc/dqn/dqn_box2d_arc.yaml ddqn_per_concat_lunar_arc train
```

**Remote** - cloud GPU via [dstack](https://dstack.ai), auto-syncs to HuggingFace:
```bash
source .env && slm-lab run-remote slm_lab/spec/benchmark_arc/ppo/ppo_classic_arc.yaml ppo_cartpole_arc train -n cartpole
source .env && slm-lab run-remote slm_lab/spec/benchmark_arc/dqn/dqn_box2d_arc.yaml ddqn_per_concat_lunar_arc train -n lunar
```

Remote setup: `cp .env.example .env` then set `HF_TOKEN`. See [Remote Training](../using-slm-lab/remote-training.md) for dstack config.

{% hint style="info" %}
**GPU not required for Classic Control.** These environments train fast on CPU. Box2D (LunarLander) benefits from GPU but still runs fine locally.
{% endhint %}

### Download and Replay

```bash
# List all available experiments (requires HF_REPO=SLM-Lab/benchmark in .env)
source .env && slm-lab list

# Download a specific experiment
source .env && slm-lab pull ppo_cartpole_arc

# Replay the trained agent
slm-lab run slm_lab/spec/benchmark_arc/ppo/ppo_classic_arc.yaml ppo_cartpole_arc enjoy@data/ppo_cartpole_arc_*/ppo_cartpole_arc_t0_spec.yaml
```

---

## Results

### Classic Control

#### CartPole-v1

[Docs](https://gymnasium.farama.org/environments/classic_control/cart_pole/) | State: Box(4) | Action: Discrete(2) | Target: >400

**Settings**: max_frame 2e5 | num_envs 4 | max_session 4 | log_frequency 500

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| REINFORCE | ✅ | 483.31 | [slm_lab/spec/benchmark_arc/reinforce/reinforce_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/reinforce/reinforce_arc.yaml) | reinforce_cartpole_arc | [reinforce_cartpole_arc_2026_02_11_135616](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/reinforce_cartpole_arc_2026_02_11_135616) |
| SARSA | ✅ | 430.95 | [slm_lab/spec/benchmark_arc/sarsa/sarsa_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/sarsa/sarsa_arc.yaml) | sarsa_boltzmann_cartpole_arc | [sarsa_boltzmann_cartpole_arc_2026_02_11_135616](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sarsa_boltzmann_cartpole_arc_2026_02_11_135616) |
| DQN | ⚠️ | 239.94 | [slm_lab/spec/benchmark_arc/dqn/dqn_classic_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/dqn/dqn_classic_arc.yaml) | dqn_boltzmann_cartpole_arc | [dqn_boltzmann_cartpole_arc_2026_02_11_135648](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/dqn_boltzmann_cartpole_arc_2026_02_11_135648) |
| DDQN+PER | ✅ | 451.51 | [slm_lab/spec/benchmark_arc/dqn/dqn_classic_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/dqn/dqn_classic_arc.yaml) | ddqn_per_boltzmann_cartpole_arc | [ddqn_per_boltzmann_cartpole_arc_2026_02_11_140518](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ddqn_per_boltzmann_cartpole_arc_2026_02_11_140518) |
| A2C | ✅ | 496.68 | [slm_lab/spec/benchmark_arc/a2c/a2c_classic_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/a2c/a2c_classic_arc.yaml) | a2c_gae_cartpole_arc | [a2c_gae_cartpole_arc_2026_02_11_142531](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_cartpole_arc_2026_02_11_142531) |
| PPO | ✅ | 498.94 | [slm_lab/spec/benchmark_arc/ppo/ppo_classic_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/ppo/ppo_classic_arc.yaml) | ppo_cartpole_arc | [ppo_cartpole_arc_2026_02_11_144029](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_cartpole_arc_2026_02_11_144029) |
| SAC | ✅ | 406.09 | [slm_lab/spec/benchmark_arc/sac/sac_classic_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/sac/sac_classic_arc.yaml) | sac_cartpole_arc | [sac_cartpole_arc_2026_02_11_144155](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_cartpole_arc_2026_02_11_144155) |

![CartPole-v1](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/CartPole-v1_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### Acrobot-v1

[Docs](https://gymnasium.farama.org/environments/classic_control/acrobot/) | State: Box(6) | Action: Discrete(3) | Target: >-100

**Settings**: max_frame 3e5 | num_envs 4 | max_session 4 | log_frequency 500

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| DQN | ✅ | -94.17 | [slm_lab/spec/benchmark_arc/dqn/dqn_classic_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/dqn/dqn_classic_arc.yaml) | dqn_boltzmann_acrobot_arc | [dqn_boltzmann_acrobot_arc_2026_02_11_144342](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/dqn_boltzmann_acrobot_arc_2026_02_11_144342) |
| DDQN+PER | ✅ | -83.92 | [slm_lab/spec/benchmark_arc/dqn/dqn_classic_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/dqn/dqn_classic_arc.yaml) | ddqn_per_acrobot_arc | [ddqn_per_acrobot_arc_2026_02_11_153725](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ddqn_per_acrobot_arc_2026_02_11_153725) |
| A2C | ✅ | -83.99 | [slm_lab/spec/benchmark_arc/a2c/a2c_classic_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/a2c/a2c_classic_arc.yaml) | a2c_gae_acrobot_arc | [a2c_gae_acrobot_arc_2026_02_11_153806](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_acrobot_arc_2026_02_11_153806) |
| PPO | ✅ | -81.28 | [slm_lab/spec/benchmark_arc/ppo/ppo_classic_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/ppo/ppo_classic_arc.yaml) | ppo_acrobot_arc | [ppo_acrobot_arc_2026_02_11_153758](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_acrobot_arc_2026_02_11_153758) |
| SAC | ✅ | -92.60 | [slm_lab/spec/benchmark_arc/sac/sac_classic_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/sac/sac_classic_arc.yaml) | sac_acrobot_arc | [sac_acrobot_arc_2026_02_11_162211](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_acrobot_arc_2026_02_11_162211) |

![Acrobot-v1](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Acrobot-v1_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### Pendulum-v1

[Docs](https://gymnasium.farama.org/environments/classic_control/pendulum/) | State: Box(3) | Action: Box(1) | Target: >-200

**Settings**: max_frame 3e5 | num_envs 4 | max_session 4 | log_frequency 500

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| A2C | ❌ | -820.74 | [slm_lab/spec/benchmark_arc/a2c/a2c_classic_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/a2c/a2c_classic_arc.yaml) | a2c_gae_pendulum_arc | [a2c_gae_pendulum_arc_2026_02_11_162217](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_pendulum_arc_2026_02_11_162217) |
| PPO | ✅ | -174.87 | [slm_lab/spec/benchmark_arc/ppo/ppo_classic_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/ppo/ppo_classic_arc.yaml) | ppo_pendulum_arc | [ppo_pendulum_arc_2026_02_11_162156](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_pendulum_arc_2026_02_11_162156) |
| SAC | ✅ | -150.97 | [slm_lab/spec/benchmark_arc/sac/sac_classic_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/sac/sac_classic_arc.yaml) | sac_pendulum_arc | [sac_pendulum_arc_2026_02_11_162240](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_pendulum_arc_2026_02_11_162240) |

![Pendulum-v1](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Pendulum-v1_multi_trial_graph_mean_returns_ma_vs_frames.png)

### Box2D

#### LunarLander-v3 (Discrete)

[Docs](https://gymnasium.farama.org/environments/box2d/lunar_lander/) | State: Box(8) | Action: Discrete(4) | Target: >200

**Settings**: max_frame 3e5 | num_envs 8 | max_session 4 | log_frequency 1000

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| DQN | ⚠️ | 195.21 | [slm_lab/spec/benchmark_arc/dqn/dqn_box2d_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/dqn/dqn_box2d_arc.yaml) | dqn_concat_lunar_arc | [dqn_concat_lunar_arc_2026_02_11_201407](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/dqn_concat_lunar_arc_2026_02_11_201407) |
| DDQN+PER | ✅ | 265.90 | [slm_lab/spec/benchmark_arc/dqn/dqn_box2d_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/dqn/dqn_box2d_arc.yaml) | ddqn_per_concat_lunar_arc | [ddqn_per_concat_lunar_arc_2026_02_13_105115](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ddqn_per_concat_lunar_arc_2026_02_13_105115) |
| A2C | ❌ | 27.38 | [slm_lab/spec/benchmark_arc/a2c/a2c_classic_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/a2c/a2c_classic_arc.yaml) | a2c_gae_lunar_arc | [a2c_gae_lunar_arc_2026_02_11_224304](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_lunar_arc_2026_02_11_224304) |
| PPO | ⚠️ | 183.30 | [slm_lab/spec/benchmark_arc/ppo/ppo_box2d_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/ppo/ppo_box2d_arc.yaml) | ppo_lunar_arc | [ppo_lunar_arc_2026_02_11_201303](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_lunar_arc_2026_02_11_201303) |
| SAC | ⚠️ | 106.17 | [slm_lab/spec/benchmark_arc/sac/sac_box2d_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/sac/sac_box2d_arc.yaml) | sac_lunar_arc | [sac_lunar_arc_2026_02_11_201417](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_lunar_arc_2026_02_11_201417) |

![LunarLander-v3 Discrete](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/LunarLander-v3_Discrete_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### LunarLander-v3 (Continuous)

[Docs](https://gymnasium.farama.org/environments/box2d/lunar_lander/) | State: Box(8) | Action: Box(2) | Target: >200

**Settings**: max_frame 3e5 | num_envs 8 | max_session 4 | log_frequency 1000

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| A2C | ❌ | -76.81 | [slm_lab/spec/benchmark_arc/a2c/a2c_classic_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/a2c/a2c_classic_arc.yaml) | a2c_gae_lunar_continuous_arc | [a2c_gae_lunar_continuous_arc_2026_02_11_224301](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_lunar_continuous_arc_2026_02_11_224301) |
| PPO | ⚠️ | 132.58 | [slm_lab/spec/benchmark_arc/ppo/ppo_box2d_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/ppo/ppo_box2d_arc.yaml) | ppo_lunar_continuous_arc | [ppo_lunar_continuous_arc_2026_02_11_224229](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_lunar_continuous_arc_2026_02_11_224229) |
| SAC | ⚠️ | 125.00 | [slm_lab/spec/benchmark_arc/sac/sac_box2d_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/sac/sac_box2d_arc.yaml) | sac_lunar_continuous_arc | [sac_lunar_continuous_arc_2026_02_12_222203](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_lunar_continuous_arc_2026_02_12_222203) |

![LunarLander-v3 Continuous](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/LunarLander-v3_Continuous_multi_trial_graph_mean_returns_ma_vs_frames.png)

**Legend:** ✅ Solved | ⚠️ Close (>80%) | ❌ Failed

---

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
