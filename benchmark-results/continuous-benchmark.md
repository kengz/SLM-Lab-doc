# Continuous Environment Benchmark

## MuJoCo Benchmark Results (v5)

SLM Lab v5 validates PPO and SAC on Gymnasium MuJoCo environments. Full methodology and active runs in [docs/BENCHMARKS.md](https://github.com/kengz/SLM-Lab/blob/master/docs/BENCHMARKS.md).

| Environment | Target | PPO | SAC | Notes |
|-------------|--------|-----|-----|-------|
| Hopper-v5 | 2500 | 2914 ✅ | 2719 ✅ | |
| HalfCheetah-v5 | 5000 | 6383 ✅ | 7410 ✅ | |
| Walker2d-v5 | 3500 | 5700 ✅ | 3824 ✅ | |
| Ant-v5 | 2000 | 2190 ✅ | 3131 ✅ | |
| Swimmer-v5 | 300 | 349 ✅ | 333 ✅ | |
| Reacher-v5 | -5 | -5.29 ✅ | -5.18 ⚠️ | |
| Pusher-v5 | -40 | -40.46 ✅ | -37.7 ✅ | |
| InvertedPendulum-v5 | 1000 | 982 ✅ | 1000 ✅ | |
| InvertedDoublePendulum-v5 | 9000 | 9059 ✅ | 9347 ✅ | |
| Humanoid-v5 | 700 | 1573 ✅ | 4860 ✅ | |
| HumanoidStandup-v5 | 100k | 103k ✅ | 154k ✅ | |

### PPO MuJoCo Configuration

Standard configuration used across MuJoCo environments:

* **Network:** `[256, 256]` hidden layers with tanh activation, orthogonal init
* **Normalization:** `normalize_obs=true`, `normalize_reward=true`, `normalize_v_targets=true`
* **Training:** num_envs=16, max_frame varies by difficulty (1M-10M)

### Running MuJoCo Benchmarks

```bash
# PPO on Hopper
slm-lab run slm_lab/spec/benchmark/ppo/ppo_hopper.json ppo_hopper train

# SAC on HalfCheetah
slm-lab run slm_lab/spec/benchmark/sac/sac_halfcheetah.json sac_halfcheetah train

# Generic MuJoCo spec with variable substitution
slm-lab run -s env=Humanoid-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco train
```

## Historical Roboschool Results (v4)

{% hint style="warning" %}
Roboschool is deprecated. These v4 results are preserved for reference. Use Gymnasium MuJoCo environments for new work.
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
