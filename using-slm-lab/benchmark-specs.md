# Benchmark Specs 📋

All benchmark specs are in [slm_lab/spec/benchmark/](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/spec/benchmark), organized by algorithm.

## By Algorithm

### REINFORCE / SARSA

Simple algorithms for learning fundamentals. CartPole only.

| Algorithm | Spec |
|-----------|------|
| REINFORCE | [reinforce_cartpole.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/reinforce/reinforce_cartpole.json) |
| SARSA | [sarsa_cartpole.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sarsa/sarsa_cartpole.json) |

### DQN Family

Value-based algorithms for discrete action spaces.

| Environment | DQN | DDQN+PER |
|-------------|-----|----------|
| CartPole | [dqn_cartpole.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/dqn/dqn_cartpole.json) | — |
| Acrobot | [dqn_acrobot.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/dqn/dqn_acrobot.json) | [ddqn_per_acrobot.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/dqn/ddqn_per_acrobot.json) |
| LunarLander | [dqn_lunar.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/dqn/dqn_lunar.json) | [ddqn_per_lunar.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/dqn/ddqn_per_lunar.json) |

### A2C

On-policy actor-critic with synchronized updates.

| Environment | Spec |
|-------------|------|
| CartPole | [a2c_gae_cartpole.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/a2c/a2c_gae_cartpole.json) |
| Acrobot | [a2c_gae_acrobot.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/a2c/a2c_gae_acrobot.json) |
| Pendulum | [a2c_gae_pendulum.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/a2c/a2c_gae_pendulum.json) |
| LunarLander | [a2c_gae_lunar.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/a2c/a2c_gae_lunar.json) |
| BipedalWalker | [a2c_gae_bipedalwalker.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/a2c/a2c_gae_bipedalwalker.json) |
| MuJoCo | [a2c_gae_mujoco.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json) (template) |
| Atari | [a2c_gae_atari.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/a2c/a2c_gae_atari.json) (template) |

### PPO

Proximal Policy Optimization—robust across all environment types.

| Environment | Spec |
|-------------|------|
| CartPole | [ppo_cartpole.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_cartpole.json) |
| Acrobot | [ppo_acrobot.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_acrobot.json) |
| Pendulum | [ppo_pendulum.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_pendulum.json) |
| LunarLander | [ppo_lunar.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_lunar.json) |
| BipedalWalker | [ppo_bipedalwalker.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_bipedalwalker.json) |
| MuJoCo | [ppo_mujoco.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_mujoco.json) (template) |
| Atari | [ppo_atari.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_atari.json) (template) |

### SAC

Soft Actor-Critic—best for continuous control.

| Environment | Spec |
|-------------|------|
| CartPole | [sac_cartpole.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_cartpole.json) |
| Acrobot | [sac_acrobot.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_acrobot.json) |
| Pendulum | [sac_pendulum.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_pendulum.json) |
| LunarLander | [sac_lunar.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_lunar.json) |
| BipedalWalker | [sac_bipedalwalker.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_bipedalwalker.json) |
| HalfCheetah | [sac_halfcheetah.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_halfcheetah.json) |
| Hopper | [sac_hopper.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_hopper.json) |

## Template Specs

Template specs use `${env}` and `${max_frame}` placeholders. Substitute with `-s`:

```bash
# MuJoCo
slm-lab run -s env=Hopper-v5 -s max_frame=4e6 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco train

# Atari
slm-lab run -s env=ALE/Breakout-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari train
```

### MuJoCo Environments

All 11 MuJoCo environments work with `ppo_mujoco.json`:

| Environment | Typical max_frame |
|-------------|-------------------|
| InvertedPendulum-v5 | 1e6 |
| InvertedDoublePendulum-v5 | 2e6 |
| Reacher-v5 | 2e6 |
| Hopper-v5 | 4e6 |
| Walker2d-v5 | 4e6 |
| HalfCheetah-v5 | 4e6 |
| Swimmer-v5 | 4e6 |
| Pusher-v5 | 4e6 |
| Ant-v5 | 10e6 |
| Humanoid-v5 | 10e6 |
| HumanoidStandup-v5 | 10e6 |

### Atari Games

All 54 Atari games work with `ppo_atari.json`. Use `ALE/{Game}-v5` format:

```bash
slm-lab run -s env=ALE/Pong-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari train
slm-lab run -s env=ALE/Breakout-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari train
slm-lab run -s env=ALE/MsPacman-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari train
```

See [Atari Benchmark](../benchmark-results/atari-benchmark.md) for the full game list and optimal lambda values.

## Performance Results

- [Discrete Benchmark](../benchmark-results/discrete-benchmark.md) — Classic Control, Box2D
- [Continuous Benchmark](../benchmark-results/continuous-benchmark.md) — MuJoCo
- [Atari Benchmark](../benchmark-results/atari-benchmark.md) — 54 Atari games
