# Continuous Environment Benchmark 🏃

## MuJoCo Benchmark Results (v5)

SLM Lab v5 validates PPO on [Gymnasium MuJoCo environments](https://gymnasium.farama.org/environments/mujoco/). MuJoCo (Multi-Joint dynamics with Contact) provides physics simulation for continuous control tasks ranging from simple pendulums to complex humanoid locomotion.

Results below are from January 2026 benchmark reruns using MuJoCo v5 environments.

All trained models and metrics are publicly available on [HuggingFace](https://huggingface.co/datasets/SLM-Lab/benchmark).

### Methodology

Results show **Trial-level** performance:

1. **Trial** = 4 Sessions with different random seeds
2. **Session** = One complete training run
3. **Score** = Final 100-checkpoint moving average (`total_reward_ma`)

The trial score is the mean across 4 sessions, providing statistically meaningful results.

### Standardized Settings

| Setting | Value |
|---------|-------|
| num_envs | 16 |
| max_frame | 4e6-10e6 (varies by env) |
| log_frequency | 10000 |
| ASHA grace_period | 1e5-1e6 |

The `grace_period` is the minimum frames before ASHA early stopping can terminate underperforming trials.

{% hint style="warning" %}
**v5 vs v4 Difficulty:** Gymnasium MuJoCo v5 environments are significantly harder than v4. Key changes include:
- Updated physics engine with more accurate contact dynamics
- Revised reward functions with stricter success criteria
- Termination conditions more closely match real-world failure modes

Expect **10-30% lower scores** compared to v4 benchmarks. See [Gymnasium Migration Guide](https://gymnasium.farama.org/content/migration-guide/) for details.
{% endhint %}

{% hint style="info" %}
**January 2026 Rerun:** SAC benchmarks are omitted in this rerun due to compute constraints (off-policy algorithms require significantly more resources for systematic benchmarking). PPO results cover all 11 MuJoCo environments.
{% endhint %}

### PPO MuJoCo Configuration

Standard configuration used across MuJoCo environments:

* **Network:** `[256, 256]` hidden layers with tanh activation, orthogonal init
* **Normalization:** `normalize_obs=true`, `normalize_reward=true`, `normalize_v_targets=true`
* **Training:** num_envs=16, max_frame varies by difficulty (4M-10M)

### Spec Variants

Two unified specs in [ppo_mujoco.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_mujoco.json), plus individual specs for tuned hyperparameters:

| SPEC_NAME | Envs | Key Config |
|-----------|------|------------|
| ppo_mujoco | HalfCheetah, Walker, Humanoid, HumanoidStandup | gamma=0.99, lam=0.95 |
| ppo_mujoco_longhorizon | Reacher, Pusher | gamma=0.997, lam=0.97 |
| Individual specs | Hopper, Swimmer, Ant, IP, IDP | See spec files |

**Quick Reference**: Copy `ENV`, `SPEC_FILE`, `SPEC_NAME` from the table below for reproduction.

| ENV | State | Action | MAX_FRAME | SPEC_FILE | SPEC_NAME |
|-----|-------|--------|-----------|-----------|-----------|
| [Hopper-v5](https://gymnasium.farama.org/environments/mujoco/hopper/) | Box(11) | Box(3) | 4e6 | ppo_hopper.json | ppo_hopper |
| [HalfCheetah-v5](https://gymnasium.farama.org/environments/mujoco/half_cheetah/) | Box(17) | Box(6) | 10e6 | ppo_mujoco.json | ppo_mujoco |
| [Walker2d-v5](https://gymnasium.farama.org/environments/mujoco/walker2d/) | Box(17) | Box(6) | 10e6 | ppo_mujoco.json | ppo_mujoco |
| [Ant-v5](https://gymnasium.farama.org/environments/mujoco/ant/) | Box(105) | Box(8) | 10e6 | ppo_ant.json | ppo_ant |
| [Swimmer-v5](https://gymnasium.farama.org/environments/mujoco/swimmer/) | Box(8) | Box(2) | 4e6 | ppo_swimmer.json | ppo_swimmer |
| [Reacher-v5](https://gymnasium.farama.org/environments/mujoco/reacher/) | Box(11) | Box(2) | 4e6 | ppo_mujoco.json | ppo_mujoco_longhorizon |
| [Pusher-v5](https://gymnasium.farama.org/environments/mujoco/pusher/) | Box(23) | Box(7) | 4e6 | ppo_mujoco.json | ppo_mujoco_longhorizon |
| [InvertedPendulum-v5](https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/) | Box(4) | Box(1) | 4e6 | ppo_inverted_pendulum.json | ppo_inverted_pendulum |
| [InvertedDoublePendulum-v5](https://gymnasium.farama.org/environments/mujoco/inverted_double_pendulum/) | Box(11) | Box(1) | 10e6 | ppo_inverted_double_pendulum.json | ppo_inverted_double_pendulum |
| [Humanoid-v5](https://gymnasium.farama.org/environments/mujoco/humanoid/) | Box(376) | Box(17) | 10e6 | ppo_mujoco.json | ppo_mujoco |
| [HumanoidStandup-v5](https://gymnasium.farama.org/environments/mujoco/humanoid_standup/) | Box(376) | Box(17) | 4e6 | ppo_mujoco.json | ppo_mujoco |

### Results

**Settings**: max_frame 4e6-10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Environment | Target | PPO | Status | Spec | HuggingFace |
|-------------|--------|-----|--------|------|-------------|
| Hopper-v5 | ~2000 | 1972 | ✅ | [ppo_hopper.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_hopper.json) | [ppo_hopper_2026_01_31](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_hopper_2026_01_31_105438) |
| HalfCheetah-v5 | >5000 | 5852 | ✅ | [ppo_mujoco.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_mujoco.json) | [ppo_mujoco_halfcheetah_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_mujoco_halfcheetah_2026_01_30_230302) |
| Walker2d-v5 | >3500 | 4042 | ✅ | [ppo_mujoco.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_mujoco.json) | [ppo_mujoco_walker2d_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_mujoco_walker2d_2026_01_30_222124) |
| Ant-v5 | >2000 | 2515 | ✅ | [ppo_ant.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_ant.json) | [ppo_ant_2026_01_31](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_ant_2026_01_31_042006) |
| Swimmer-v5 | >200 | 229 | ✅ | [ppo_swimmer.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_swimmer.json) | [ppo_swimmer_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_swimmer_2026_01_30_215922) |
| Reacher-v5 | >-10 | -5.08 | ✅ | [ppo_mujoco.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_mujoco.json) | [ppo_mujoco_longhorizon_reacher_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_mujoco_longhorizon_reacher_2026_01_30_215805) |
| Pusher-v5 | >-50 | -49.1 | ✅ | [ppo_mujoco.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_mujoco.json) | [ppo_mujoco_longhorizon_pusher_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_mujoco_longhorizon_pusher_2026_01_30_215824) |
| InvertedPendulum-v5 | ~1000 | 945 | ✅ | [ppo_inverted_pendulum.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_inverted_pendulum.json) | [ppo_inverted_pendulum_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_inverted_pendulum_2026_01_30_230211) |
| InvertedDoublePendulum-v5 | ~8000 | 7622 | ✅ | [ppo_inverted_double_pendulum.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_inverted_double_pendulum.json) | [ppo_inverted_double_pendulum_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_inverted_double_pendulum_2026_01_30_220651) |
| Humanoid-v5 | >1000 | 3774 | ✅ | [ppo_mujoco.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_mujoco.json) | [ppo_mujoco_humanoid_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_mujoco_humanoid_2026_01_30_222339) |
| HumanoidStandup-v5 | >100k | 165841 | ✅ | [ppo_mujoco.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_mujoco.json) | [ppo_mujoco_humanoidstandup_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_mujoco_humanoidstandup_2026_01_30_215802) |

**Legend:** ✅ Solved | ⚠️ Close (>80%) | ❌ Failed

### Training Curves

Multi-trial comparison plots showing mean returns (moving average) vs training frames. Shaded regions show standard deviation across 4 sessions.

![Hopper-v5](https://raw.githubusercontent.com/kengz/SLM-Lab/master/docs/plots/Hopper-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

![HalfCheetah-v5](https://raw.githubusercontent.com/kengz/SLM-Lab/master/docs/plots/HalfCheetah-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

![Walker2d-v5](https://raw.githubusercontent.com/kengz/SLM-Lab/master/docs/plots/Walker2d-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

![Ant-v5](https://raw.githubusercontent.com/kengz/SLM-Lab/master/docs/plots/Ant-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

![Swimmer-v5](https://raw.githubusercontent.com/kengz/SLM-Lab/master/docs/plots/Swimmer-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

![Reacher-v5](https://raw.githubusercontent.com/kengz/SLM-Lab/master/docs/plots/Reacher-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

![Pusher-v5](https://raw.githubusercontent.com/kengz/SLM-Lab/master/docs/plots/Pusher-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

![InvertedPendulum-v5](https://raw.githubusercontent.com/kengz/SLM-Lab/master/docs/plots/InvertedPendulum-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

![InvertedDoublePendulum-v5](https://raw.githubusercontent.com/kengz/SLM-Lab/master/docs/plots/InvertedDoublePendulum-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

![Humanoid-v5](https://raw.githubusercontent.com/kengz/SLM-Lab/master/docs/plots/Humanoid-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

![HumanoidStandup-v5](https://raw.githubusercontent.com/kengz/SLM-Lab/master/docs/plots/HumanoidStandup-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

### Running MuJoCo Benchmarks

```bash
# Local training - individual spec
slm-lab run slm_lab/spec/benchmark/ppo/ppo_hopper.json ppo_hopper train

# Local training - unified spec with variable substitution
slm-lab run -s env=Humanoid-v5 -s max_frame=10e6 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco train

# Long-horizon spec for Reacher/Pusher
slm-lab run -s env=Reacher-v5 -s max_frame=4e6 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco_longhorizon train

# Remote training with GPU (recommended for MuJoCo)
source .env && slm-lab run-remote --gpu -s env=Humanoid-v5 -s max_frame=10e6 \
  slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco train -n humanoid
```

### Download and Replay

```bash
# List all available experiments (requires HF_REPO=SLM-Lab/benchmark in .env)
source .env && slm-lab list

# Download a specific experiment
source .env && slm-lab pull ppo_hopper

# Replay the trained agent
slm-lab run slm_lab/spec/benchmark/ppo/ppo_hopper.json ppo_hopper enjoy@data/ppo_hopper_2026_01_31_105438/ppo_hopper_t0_spec.json
```

## Historical Results

<details>
<summary><b>Roboschool Results (v4)</b> - click to expand</summary>

{% hint style="warning" %}
**Deprecated:** Roboschool is abandoned (MuJoCo became free in 2022). These v4 results are preserved for historical reference only. Use Gymnasium MuJoCo environments for new work.

Environment mapping: `RoboschoolHopper-v1` → `Hopper-v5`, `RoboschoolHalfCheetah-v1` → `HalfCheetah-v5`, etc.
{% endhint %}

* [Upload PR #427](https://github.com/kengz/SLM-Lab/pull/427)
* [Google Drive data](https://drive.google.com/file/d/1_rVeXPuZoifXuJmyzY8vC5yregH4DTXo/view?usp=sharing)

| Env. \ Alg. | A2C (GAE) | A2C (n-step) | PPO | SAC |
|-------------|-----------|--------------|-----|-----|
| RoboschoolAnt | 787 | 1396 | 1843 | **2915** |
| RoboschoolHalfCheetah | 712 | 439 | 1960 | **2497** |
| RoboschoolHopper | 710 | 285 | 2042 | **2045** |
| RoboschoolInvertedDoublePendulum | 996 | 4410 | 8076 | **8085** |
| RoboschoolInvertedPendulum | **995** | 978 | 986 | 941 |
| RoboschoolReacher | 12.9 | 10.16 | 19.51 | **19.99** |
| RoboschoolWalker2d | 280 | 220 | 1660 | **1894** |
| RoboschoolHumanoid | 99.31 | 54.58 | 2388 | **2621*** |

> Episode score at the end of training. Reported scores are the average over the last 100 checkpoints, averaged over 4 Sessions. Results marked with `*` required 50M-100M frames using async SAC.

</details>
