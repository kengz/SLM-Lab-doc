# Continuous Environment Benchmark 🏃

## MuJoCo Benchmark Results

SLM Lab v5 validates PPO and SAC on [Gymnasium MuJoCo environments](https://gymnasium.farama.org/environments/mujoco/). MuJoCo (Multi-Joint dynamics with Contact) provides physics simulation for continuous control tasks ranging from simple pendulums to complex humanoid locomotion.

Results below are from January-February 2026 benchmark runs using MuJoCo v5 environments.

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
| MuJoCo | 16 | 4e6-10e6 | 10000 | 1e5-1e6 |

The `grace_period` is the minimum frames before ASHA early stopping can terminate underperforming trials.

**Algorithms**: PPO and SAC. Network: MLP [256,256], orthogonal init. PPO uses tanh activation; SAC uses relu.

**Note on SAC frame budgets**: SAC uses higher update-to-data ratios (more gradient updates per step), making it more sample-efficient but slower per frame than PPO. SAC benchmarks use 1-4M frames (vs PPO's 4-10M) to fit within practical GPU wall-time limits (~6h). Scores may still be improving at cutoff.

{% hint style="warning" %}
**v5 vs v4 Difficulty:** Gymnasium MuJoCo v5 environments are significantly harder than v4. Key changes include:
- Updated physics engine with more accurate contact dynamics
- Revised reward functions with stricter success criteria
- Termination conditions more closely match real-world failure modes

Expect **10-30% lower scores** compared to v4 benchmarks. See [Gymnasium Migration Guide](https://gymnasium.farama.org/content/migration-guide/) for details.
{% endhint %}

### Spec Files

**Spec Files** (one file per algorithm, all envs via YAML anchors):
- **PPO**: [ppo_mujoco_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml)
- **SAC**: [sac_mujoco_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml)

**Spec Variants**: Each file has a base config (shared via YAML anchors) with per-env overrides:

| SPEC_NAME | Envs | Key Config |
|-----------|------|------------|
| ppo_mujoco_arc | HalfCheetah, Walker, Humanoid, HumanoidStandup | Base: gamma=0.99, lam=0.95, lr=3e-4 |
| ppo_mujoco_longhorizon_arc | Reacher, Pusher | gamma=0.997, lam=0.97, lr=2e-4, entropy=0.001 |
| ppo_{env}_arc | Ant, Hopper, Swimmer, IP, IDP | Per-env tuned (gamma, lam, lr) |
| sac_mujoco_arc | (generic, use with -s flags) | Base: gamma=0.99, iter=4, lr=3e-4, [256,256] |
| sac_{env}_arc | All 11 envs | Per-env tuned (iter, gamma, lr, net size) |

### Running Benchmarks

**Reproduce**: Copy `SPEC_NAME` and `MAX_FRAME` from the table below.

```bash
# PPO: env and max_frame are parameterized via -s flags
source .env && slm-lab run-remote --gpu -s env=ENV -s max_frame=MAX_FRAME \
  slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml SPEC_NAME train -n NAME

# SAC: env and max_frame are hardcoded per spec — no -s flags needed
source .env && slm-lab run-remote --gpu \
  slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml SPEC_NAME train -n NAME
```

| ENV | SPEC_NAME | MAX_FRAME |
|-----|-----------|-----------|
| Ant-v5 | ppo_ant_arc | 10e6 |
| | sac_ant_arc | 2e6 |
| HalfCheetah-v5 | ppo_mujoco_arc | 10e6 |
| | sac_halfcheetah_arc | 4e6 |
| Hopper-v5 | ppo_hopper_arc | 4e6 |
| | sac_hopper_arc | 3e6 |
| Humanoid-v5 | ppo_mujoco_arc | 10e6 |
| | sac_humanoid_arc | 1e6 |
| HumanoidStandup-v5 | ppo_mujoco_arc | 4e6 |
| | sac_humanoid_standup_arc | 1e6 |
| InvertedDoublePendulum-v5 | ppo_inverted_double_pendulum_arc | 10e6 |
| | sac_inverted_double_pendulum_arc | 2e6 |
| InvertedPendulum-v5 | ppo_inverted_pendulum_arc | 4e6 |
| | sac_inverted_pendulum_arc | 2e6 |
| Pusher-v5 | ppo_mujoco_longhorizon_arc | 4e6 |
| | sac_pusher_arc | 1e6 |
| Reacher-v5 | ppo_mujoco_longhorizon_arc | 4e6 |
| | sac_reacher_arc | 1e6 |
| Swimmer-v5 | ppo_swimmer_arc | 4e6 |
| | sac_swimmer_arc | 2e6 |
| Walker2d-v5 | ppo_mujoco_arc | 10e6 |
| | sac_walker2d_arc | 3e6 |

Remote setup: `cp .env.example .env` then set `HF_TOKEN`. See [Remote Training](../using-slm-lab/remote-training.md) for dstack config.

{% hint style="warning" %}
**GPU strongly recommended for MuJoCo.** These benchmarks run 4M-10M frames and take 1-4 hours on cloud GPU (L4/A10G). Local CPU training is not practical. Cloud GPUs via dstack are faster and often cheaper than running on local hardware.
{% endhint %}

### Download and Replay

```bash
# List all available experiments (requires HF_REPO=SLM-Lab/benchmark in .env)
source .env && slm-lab list

# Download a specific experiment
source .env && slm-lab pull ppo_ant_arc

# Replay the trained agent
slm-lab run slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml ppo_ant_arc enjoy@data/ppo_ant_arc_ant_2026_02_12_190644/ppo_ant_arc_t0_spec.json
```

---

## Results

### Ant-v5

[Docs](https://gymnasium.farama.org/environments/mujoco/ant/) | State: Box(105) | Action: Box(8) | Target: >2000

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | 2138.28 | [slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml) | ppo_ant_arc | [ppo_ant_arc_ant_2026_02_12_190644](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_ant_arc_ant_2026_02_12_190644) |
| SAC | ✅ | 4942.91 | [slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml) | sac_ant_arc | [sac_ant_arc_2026_02_11_225529](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_ant_arc_2026_02_11_225529) |

![Ant-v5](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Ant-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

### HalfCheetah-v5

[Docs](https://gymnasium.farama.org/environments/mujoco/half_cheetah/) | State: Box(17) | Action: Box(6) | Target: >5000

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | 6240.68 | [slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml) | ppo_mujoco_arc | [ppo_mujoco_arc_halfcheetah_2026_02_12_195553](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_mujoco_arc_halfcheetah_2026_02_12_195553) |
| SAC | ✅ | 9815.16 | [slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml) | sac_halfcheetah_arc | [sac_halfcheetah_4m_i2_arc_2026_02_14_185522](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_halfcheetah_4m_i2_arc_2026_02_14_185522) |

![HalfCheetah-v5](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/HalfCheetah-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

### Hopper-v5

[Docs](https://gymnasium.farama.org/environments/mujoco/hopper/) | State: Box(11) | Action: Box(3) | Target: ~2000

**Settings**: max_frame 4e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ⚠️ | 1653.74 | [slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml) | ppo_hopper_arc | [ppo_hopper_arc_hopper_2026_02_12_222206](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_hopper_arc_hopper_2026_02_12_222206) |
| SAC | ⚠️ | 1416.52 | [slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml) | sac_hopper_arc | [sac_hopper_3m_i4_arc_2026_02_14_185434](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_hopper_3m_i4_arc_2026_02_14_185434) |

![Hopper-v5](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Hopper-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

### Humanoid-v5

[Docs](https://gymnasium.farama.org/environments/mujoco/humanoid/) | State: Box(348) | Action: Box(17) | Target: >1000

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | 2661.26 | [slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml) | ppo_mujoco_arc | [ppo_mujoco_arc_humanoid_2026_02_12_185439](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_mujoco_arc_humanoid_2026_02_12_185439) |
| SAC | ✅ | 1989.65 | [slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml) | sac_humanoid_arc | [sac_humanoid_arc_2026_02_12_020016](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_humanoid_arc_2026_02_12_020016) |

![Humanoid-v5](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Humanoid-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

### HumanoidStandup-v5

[Docs](https://gymnasium.farama.org/environments/mujoco/humanoid_standup/) | State: Box(348) | Action: Box(17) | Target: >100k

**Settings**: max_frame 4e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | 150104.59 | [slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml) | ppo_mujoco_arc | [ppo_mujoco_arc_humanoidstandup_2026_02_12_115050](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_mujoco_arc_humanoidstandup_2026_02_12_115050) |
| SAC | ✅ | 137357.00 | [slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml) | sac_humanoid_standup_arc | [sac_humanoid_standup_arc_2026_02_12_225150](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_humanoid_standup_arc_2026_02_12_225150) |

![HumanoidStandup-v5](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/HumanoidStandup-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

### InvertedDoublePendulum-v5

[Docs](https://gymnasium.farama.org/environments/mujoco/inverted_double_pendulum/) | State: Box(9) | Action: Box(1) | Target: ~8000

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | 8383.76 | [slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml) | ppo_inverted_double_pendulum_arc | [ppo_inverted_double_pendulum_arc_inverteddoublependulum_2026_02_12_225231](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_inverted_double_pendulum_arc_inverteddoublependulum_2026_02_12_225231) |
| SAC | ✅ | 9032.67 | [slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml) | sac_inverted_double_pendulum_arc | [sac_inverted_double_pendulum_arc_2026_02_12_025206](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_inverted_double_pendulum_arc_2026_02_12_025206) |

![InvertedDoublePendulum-v5](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/InvertedDoublePendulum-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

### InvertedPendulum-v5

[Docs](https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/) | State: Box(4) | Action: Box(1) | Target: ~1000

**Settings**: max_frame 4e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | 949.94 | [slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml) | ppo_inverted_pendulum_arc | [ppo_inverted_pendulum_arc_invertedpendulum_2026_02_12_062037](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_inverted_pendulum_arc_invertedpendulum_2026_02_12_062037) |
| SAC | ✅ | 928.43 | [slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml) | sac_inverted_pendulum_arc | [sac_inverted_pendulum_arc_2026_02_12_225503](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_inverted_pendulum_arc_2026_02_12_225503) |

![InvertedPendulum-v5](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/InvertedPendulum-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

### Pusher-v5

[Docs](https://gymnasium.farama.org/environments/mujoco/pusher/) | State: Box(23) | Action: Box(7) | Target: >-50

**Settings**: max_frame 4e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | -49.59 | [slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml) | ppo_mujoco_longhorizon_arc | [ppo_mujoco_longhorizon_arc_pusher_2026_02_12_222228](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_mujoco_longhorizon_arc_pusher_2026_02_12_222228) |
| SAC | ✅ | -43.00 | [slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml) | sac_pusher_arc | [sac_pusher_arc_2026_02_12_053603](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_pusher_arc_2026_02_12_053603) |

![Pusher-v5](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Pusher-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

### Reacher-v5

[Docs](https://gymnasium.farama.org/environments/mujoco/reacher/) | State: Box(10) | Action: Box(2) | Target: >-10

**Settings**: max_frame 4e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | -5.03 | [slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml) | ppo_mujoco_longhorizon_arc | [ppo_mujoco_longhorizon_arc_reacher_2026_02_12_115033](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_mujoco_longhorizon_arc_reacher_2026_02_12_115033) |
| SAC | ✅ | -6.31 | [slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml) | sac_reacher_arc | [sac_reacher_arc_2026_02_12_055200](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_reacher_arc_2026_02_12_055200) |

![Reacher-v5](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Reacher-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

### Swimmer-v5

[Docs](https://gymnasium.farama.org/environments/mujoco/swimmer/) | State: Box(8) | Action: Box(2) | Target: >200

**Settings**: max_frame 4e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | 282.44 | [slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml) | ppo_swimmer_arc | [ppo_swimmer_arc_swimmer_2026_02_12_100445](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_swimmer_arc_swimmer_2026_02_12_100445) |
| SAC | ✅ | 301.34 | [slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml) | sac_swimmer_arc | [sac_swimmer_arc_2026_02_12_054349](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_swimmer_arc_2026_02_12_054349) |

![Swimmer-v5](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Swimmer-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

### Walker2d-v5

[Docs](https://gymnasium.farama.org/environments/mujoco/walker2d/) | State: Box(17) | Action: Box(6) | Target: >3500

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | 4378.62 | [slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml) | ppo_mujoco_arc | [ppo_mujoco_arc_walker2d_2026_02_12_190312](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_mujoco_arc_walker2d_2026_02_12_190312) |
| SAC | ⚠️ | 3123.66 | [slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml) | sac_walker2d_arc | [sac_walker2d_3m_i4_arc_2026_02_14_185550](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_walker2d_3m_i4_arc_2026_02_14_185550) |

![Walker2d-v5](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Walker2d-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

**Legend:** ✅ Solved | ⚠️ Close (>80%) | ❌ Failed

---

## Historical Results (v4)

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
