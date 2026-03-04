# Continuous Environment Benchmark 🏃

## MuJoCo Benchmark Results (v5)

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

**Note on SAC frame budgets**: SAC uses higher update-to-data ratios (more gradient updates per step), making it more sample-efficient but slower per frame than PPO. SAC benchmarks use 1-2M frames (vs PPO's 4-10M) to fit within practical GPU wall-time limits (~6h). Scores may still be improving at cutoff.

{% hint style="warning" %}
**v5 vs v4 Difficulty:** Gymnasium MuJoCo v5 environments are significantly harder than v4. Key changes include:
- Updated physics engine with more accurate contact dynamics
- Revised reward functions with stricter success criteria
- Termination conditions more closely match real-world failure modes

Expect **10-30% lower scores** compared to v4 benchmarks. See [Gymnasium Migration Guide](https://gymnasium.farama.org/content/migration-guide/) for details.
{% endhint %}

### Running Benchmarks

**Remote (recommended)** - cloud GPU via [dstack](https://dstack.ai), auto-syncs to HuggingFace:
```bash
# PPO - individual spec
source .env && slm-lab run-remote --gpu slm_lab/spec/benchmark/ppo/ppo_hopper.json ppo_hopper train -n hopper

# PPO - unified spec with variable substitution
source .env && slm-lab run-remote --gpu -s env=Humanoid-v5 -s max_frame=10e6 \
  slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco train -n humanoid

# SAC - individual specs (recommended)
source .env && slm-lab run-remote --gpu \
  slm_lab/spec/benchmark/sac/sac_hopper.json sac_hopper train -n sac-hopper
```

Remote setup: `cp .env.example .env` then set `HF_TOKEN`. See [Remote Training](../using-slm-lab/remote-training.md) for dstack config.

**Local** - runs on your machine (requires decent GPU, runs 1-4 hours):
```bash
slm-lab run slm_lab/spec/benchmark/ppo/ppo_hopper.json ppo_hopper train
slm-lab run slm_lab/spec/benchmark/sac/sac_hopper.json sac_hopper train
```

{% hint style="warning" %}
**GPU strongly recommended for MuJoCo.** These benchmarks run 4M-10M frames and take 1-4 hours on cloud GPU (L4/A10G). Local CPU training is not practical. Cloud GPUs via dstack are faster and often cheaper than running on local hardware.
{% endhint %}

### Download and Replay

```bash
# List all available experiments (requires HF_REPO=SLM-Lab/benchmark in .env)
source .env && slm-lab list

# Download a specific experiment
source .env && slm-lab pull ppo_hopper

# Replay the trained agent
slm-lab run slm_lab/spec/benchmark/ppo/ppo_hopper.json ppo_hopper enjoy@data/ppo_hopper_2026_01_31_105438/ppo_hopper_t0_spec.json
```

---

## Results

### Ant-v5

[Docs](https://gymnasium.farama.org/environments/mujoco/ant/) | State: Box(105) | Action: Box(8) | Target: >2000

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | Spec | HuggingFace |
|-----------|--------|-----|------|-------------|
| PPO | ✅ | 2515 | [ppo_ant.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_ant.json) | [ppo_ant_2026_01_31](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_ant_2026_01_31_042006) |
| SAC | ✅ | 4844 | [sac_ant.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_ant.json) | [sac_ant_2026_02_09](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_ant_2026_02_09_093821) |

![Ant-v5](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/v5.0.0/docs/plots/Ant-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260210)

### HalfCheetah-v5

[Docs](https://gymnasium.farama.org/environments/mujoco/half_cheetah/) | State: Box(17) | Action: Box(6) | Target: >5000

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | Spec | HuggingFace |
|-----------|--------|-----|------|-------------|
| PPO | ✅ | 5852 | [ppo_mujoco.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_mujoco.json) | [ppo_mujoco_halfcheetah_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_mujoco_halfcheetah_2026_01_30_230302) |
| SAC | ✅ | 7255 | [sac_halfcheetah.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_halfcheetah.json) | [sac_halfcheetah_2026_02_08](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_halfcheetah_2026_02_08_115456) |

![HalfCheetah-v5](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/v5.0.0/docs/plots/HalfCheetah-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260210)

### Hopper-v5

[Docs](https://gymnasium.farama.org/environments/mujoco/hopper/) | State: Box(11) | Action: Box(3) | Target: ~2000

**Settings**: max_frame 4e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | Spec | HuggingFace |
|-----------|--------|-----|------|-------------|
| PPO | ✅ | 1972 | [ppo_hopper.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_hopper.json) | [ppo_hopper_2026_01_31](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_hopper_2026_01_31_105438) |
| SAC | ⚠️ | 1511 | [sac_hopper.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_hopper.json) | [sac_hopper_2026_02_10](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_hopper_2026_02_10_031009) |

![Hopper-v5](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/v5.0.0/docs/plots/Hopper-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260210)

### Humanoid-v5

[Docs](https://gymnasium.farama.org/environments/mujoco/humanoid/) | State: Box(348) | Action: Box(17) | Target: >1000

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | Spec | HuggingFace |
|-----------|--------|-----|------|-------------|
| PPO | ✅ | 3774 | [ppo_mujoco.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_mujoco.json) | [ppo_mujoco_humanoid_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_mujoco_humanoid_2026_01_30_222339) |
| SAC | ✅ | 2601 | [sac_humanoid.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_humanoid.json) | [sac_humanoid_2026_02_09](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_humanoid_2026_02_09_223653) |

![Humanoid-v5](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/v5.0.0/docs/plots/Humanoid-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260210)

### HumanoidStandup-v5

[Docs](https://gymnasium.farama.org/environments/mujoco/humanoid_standup/) | State: Box(348) | Action: Box(17) | Target: >100k

**Settings**: max_frame 4e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | Spec | HuggingFace |
|-----------|--------|-----|------|-------------|
| PPO | ✅ | 165841 | [ppo_mujoco.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_mujoco.json) | [ppo_mujoco_humanoidstandup_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_mujoco_humanoidstandup_2026_01_30_215802) |
| SAC | ✅ | 138222 | [sac_humanoid_standup.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_humanoid_standup.json) | [sac_humanoid_standup_2026_02_09](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_humanoid_standup_2026_02_09_213409) |

![HumanoidStandup-v5](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/v5.0.0/docs/plots/HumanoidStandup-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260210)

### InvertedDoublePendulum-v5

[Docs](https://gymnasium.farama.org/environments/mujoco/inverted_double_pendulum/) | State: Box(9) | Action: Box(1) | Target: ~8000

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | Spec | HuggingFace |
|-----------|--------|-----|------|-------------|
| PPO | ✅ | 7622 | [ppo_inverted_double_pendulum.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_inverted_double_pendulum.json) | [ppo_inverted_double_pendulum_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_inverted_double_pendulum_2026_01_30_220651) |
| SAC | ✅ | 9000 | [sac_inverted_double_pendulum.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_inverted_double_pendulum.json) | [sac_inverted_double_pendulum_2026_02_08](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_inverted_double_pendulum_2026_02_08_115548) |

![InvertedDoublePendulum-v5](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/v5.0.0/docs/plots/InvertedDoublePendulum-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260210)

### InvertedPendulum-v5

[Docs](https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/) | State: Box(4) | Action: Box(1) | Target: ~1000

**Settings**: max_frame 4e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | Spec | HuggingFace |
|-----------|--------|-----|------|-------------|
| PPO | ✅ | 945 | [ppo_inverted_pendulum.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_inverted_pendulum.json) | [ppo_inverted_pendulum_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_inverted_pendulum_2026_01_30_230211) |
| SAC | ✅ | 928 | [sac_inverted_pendulum.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_inverted_pendulum.json) | [sac_inverted_pendulum_2026_02_08](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_inverted_pendulum_2026_02_08_132433) |

![InvertedPendulum-v5](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/v5.0.0/docs/plots/InvertedPendulum-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260210)

### Pusher-v5

[Docs](https://gymnasium.farama.org/environments/mujoco/pusher/) | State: Box(23) | Action: Box(7) | Target: >-50

**Settings**: max_frame 4e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | Spec | HuggingFace |
|-----------|--------|-----|------|-------------|
| PPO | ✅ | -49.1 | [ppo_mujoco.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_mujoco.json) | [ppo_mujoco_longhorizon_pusher_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_mujoco_longhorizon_pusher_2026_01_30_215824) |
| SAC | ✅ | -42.6 | [sac_pusher.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_pusher.json) | [sac_pusher_2026_02_08](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_pusher_2026_02_08_115643) |

![Pusher-v5](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/v5.0.0/docs/plots/Pusher-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260210)

### Reacher-v5

[Docs](https://gymnasium.farama.org/environments/mujoco/reacher/) | State: Box(10) | Action: Box(2) | Target: >-10

**Settings**: max_frame 4e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | Spec | HuggingFace |
|-----------|--------|-----|------|-------------|
| PPO | ✅ | -5.08 | [ppo_mujoco.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_mujoco.json) | [ppo_mujoco_longhorizon_reacher_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_mujoco_longhorizon_reacher_2026_01_30_215805) |
| SAC | ✅ | -6.36 | [sac_reacher.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_reacher.json) | [sac_reacher_2026_02_08](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_reacher_2026_02_08_115637) |

![Reacher-v5](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/v5.0.0/docs/plots/Reacher-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260210)

### Swimmer-v5

[Docs](https://gymnasium.farama.org/environments/mujoco/swimmer/) | State: Box(8) | Action: Box(2) | Target: >200

**Settings**: max_frame 4e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | Spec | HuggingFace |
|-----------|--------|-----|------|-------------|
| PPO | ✅ | 229 | [ppo_swimmer.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_swimmer.json) | [ppo_swimmer_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_swimmer_2026_01_30_215922) |
| SAC | ✅ | 265 | [sac_swimmer.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_swimmer.json) | [sac_swimmer_2026_02_08](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_swimmer_2026_02_08_115455) |

![Swimmer-v5](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/v5.0.0/docs/plots/Swimmer-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260210)

### Walker2d-v5

[Docs](https://gymnasium.farama.org/environments/mujoco/walker2d/) | State: Box(17) | Action: Box(6) | Target: >3500

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | Spec | HuggingFace |
|-----------|--------|-----|------|-------------|
| PPO | ✅ | 4042 | [ppo_mujoco.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_mujoco.json) | [ppo_mujoco_walker2d_2026_01_30](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_mujoco_walker2d_2026_01_30_222124) |
| SAC | ⚠️ | 2288 | [sac_walker2d.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_walker2d.json) | [sac_walker2d_2026_02_08](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_walker2d_2026_02_08_221549) |

![Walker2d-v5](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/v5.0.0/docs/plots/Walker2d-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260210)

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
