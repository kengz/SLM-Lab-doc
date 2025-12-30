# Discrete Environment Benchmark

## Classic Control & Box2D Results (v5)

SLM Lab v5 validates algorithms on Gymnasium discrete environments. Full methodology in [docs/BENCHMARKS.md](https://github.com/kengz/SLM-Lab/blob/master/docs/BENCHMARKS.md).

### Classic Control

| Environment | Target | PPO | A2C | DQN | DDQN+PER | SAC |
|-------------|--------|-----|-----|-----|----------|-----|
| CartPole-v1 | 400 | 499.7 ✅ | 488.7 ✅ | 437.8 ✅ | 430.4 ✅ | 431.1 ✅ |
| Acrobot-v1 | -100 | -80.8 ✅ | -84.2 ✅ | -96.2 ✅ | -83.0 ✅ | -97 ✅ |
| Pendulum-v1 | -200 | -178 ✅ | — | — | — | -150 ✅ |

### Box2D

| Environment | Target | PPO | DQN | DDQN+PER | A2C | SAC |
|-------------|--------|-----|-----|----------|-----|-----|
| LunarLander-v3 (discrete) | 200 | 229.9 ✅ | 203.9 ✅ | 230.0 ✅ | 41 📊 | — |
| LunarLander-v3 (continuous) | 200 | 245.7 ✅ | — | — | — | 241.6 ✅ |

**Legend:** ✅ Solved | ⚠️ Close (>80%) | 📊 Acceptable | ❌ Failed

### Running Discrete Benchmarks

```bash
# PPO on CartPole
slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole train

# DDQN+PER on LunarLander
slm-lab run slm_lab/spec/benchmark/dqn/ddqn_per_lunar.json ddqn_per_concat_lunar train

# SAC on Pendulum
slm-lab run slm_lab/spec/benchmark/sac/sac_pendulum.json sac_pendulum train
```

## Historical Results (v4)

{% hint style="info" %}
These results from SLM Lab v4 are preserved for reference. Unity environments are no longer included in the core package.
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

For the full Atari benchmark, see [Atari Benchmark](atari-benchmark.md).
