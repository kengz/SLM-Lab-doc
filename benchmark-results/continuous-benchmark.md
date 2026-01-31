# Continuous Environment Benchmark

## MuJoCo Benchmark Results (v5)

SLM Lab v5 validates PPO on Gymnasium MuJoCo environments. Results below are from January 2026 benchmark reruns using MuJoCo v5 environments.

Full methodology and HuggingFace data links in [docs/BENCHMARKS.md](https://github.com/kengz/SLM-Lab/blob/master/docs/BENCHMARKS.md).

{% hint style="info" %}
**January 2026 Rerun:** SAC benchmarks are omitted in this rerun due to compute constraints (off-policy algorithms require significantly more resources for systematic benchmarking). PPO results cover all 11 MuJoCo environments.
{% endhint %}

| Environment | Target | PPO | Notes |
|-------------|--------|-----|-------|
| Hopper-v5 | ~2000 | 1972 ✅ | |
| HalfCheetah-v5 | >5000 | 5852 ✅ | |
| Walker2d-v5 | >3500 | 4042 ✅ | |
| Ant-v5 | >2000 | 2515 ✅ | |
| Swimmer-v5 | >200 | 229 ✅ | |
| Reacher-v5 | >-10 | -5.08 ✅ | |
| Pusher-v5 | >-50 | -49.1 ✅ | |
| InvertedPendulum-v5 | ~1000 | 945 ✅ | |
| InvertedDoublePendulum-v5 | ~8000 | 7622 ✅ | |
| Humanoid-v5 | >1000 | 3774 ✅ | |
| HumanoidStandup-v5 | >100k | 165841 ✅ | |

**Legend:** ✅ Solved | ⚠️ Close (>80%) | ❌ Failed

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

### Running MuJoCo Benchmarks

```bash
# PPO on Hopper (individual spec)
slm-lab run slm_lab/spec/benchmark/ppo/ppo_hopper.json ppo_hopper train

# Generic MuJoCo spec with variable substitution
slm-lab run -s env=Humanoid-v5 -s max_frame=10e6 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco train

# Long-horizon spec for Reacher/Pusher
slm-lab run -s env=Reacher-v5 -s max_frame=4e6 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco_longhorizon train
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
