# Running Benchmarks 📋

This guide covers how to run reproducible benchmarks with SLM Lab, including hyperparameter search methodology and best practices.

## Quick Start

```bash
# Run a benchmark locally
slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole train

# Run on cloud GPU (faster, auto-syncs results)
source .env && slm-lab run-remote --gpu slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole train -n ppo-cartpole

# Download trained models
slm-lab pull ppo_cartpole

# Replay trained model
slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole enjoy
```

{% hint style="info" %}
**Authoritative source:** [BENCHMARKS.md](https://github.com/kengz/SLM-Lab/blob/master/docs/BENCHMARKS.md) in the repository contains exact reproduction commands, current results, and HuggingFace links.
{% endhint %}

## Standardized Settings

Fair comparison requires consistent configurations across environment categories:

| Category | `num_envs` | `max_frame` | `log_frequency` | `max_session` |
|----------|------------|-------------|-----------------|---------------|
| Classic Control | 4 | 2e5-3e5 | 500 | 4 |
| Box2D | 8 | 3e5 | 1000 | 4 |
| MuJoCo | 16 | 1e6-10e6 | 10000 | 4 |
| Atari | 16 | 10e6 | 10000 | 4 |

{% hint style="warning" %}
**Before running:** Verify spec settings match the table above. Inconsistent settings make results incomparable.
{% endhint %}

## Three-Stage Search Process

When tuning hyperparameters or adding new environments, use this systematic approach:

| Stage | Mode | Config | Purpose |
|-------|------|--------|---------|
| **1. ASHA** | `search` | `max_session=1`, `search_scheduler` enabled | Wide exploration with early termination |
| **2. Multi** | `search` | `max_session=4`, NO `search_scheduler` | Validate top configs with multiple seeds |
| **3. Final** | `train` | Best hyperparameters committed to spec | Confirmation run for benchmark table |

### Stage 1: ASHA Search

ASHA (Asynchronous Successive Halving) terminates unpromising trials early, focusing compute on promising configurations.

```json
{
  "meta": {
    "max_session": 1,
    "max_trial": 16,
    "search_resources": {"cpu": 1, "gpu": 0.125},
    "search_scheduler": {
      "grace_period": 100000,
      "reduction_factor": 3
    }
  },
  "search": {
    "agent.algorithm.gamma__uniform": [0.98, 0.999],
    "agent.algorithm.lam__uniform": [0.9, 0.98],
    "agent.net.optim_spec.lr__loguniform": [1e-4, 1e-3]
  }
}
```

```bash
slm-lab run spec.json spec_name search
```

### Stage 2: Multi-Seed Validation

After ASHA, validate top 3-5 configurations with multiple seeds (no early stopping):

```json
{
  "meta": {
    "max_session": 4,
    "max_trial": 5
  }
}
```

Single runs can be lucky—averaging 4 independent runs reveals true performance.

### Stage 3: Final Validation

Update spec defaults with best hyperparameters, then run in train mode:

```bash
slm-lab run spec.json spec_name train
```

{% hint style="warning" %}
**Never use raw search results in benchmark tables.** Always run a final validation with committed spec file.
{% endhint %}

## Search Space Sizing

**Rule: ~3-4 trials per search dimension minimum.**

| `max_trial` | Max Dimensions | Use Case |
|-------------|----------------|----------|
| 8 | 2-3 | Focused refinement |
| 12-16 | 3-4 | Typical search |
| 20 | 5 | Wide exploration |
| 30 | 6-7 | Broad ASHA search |

### High-Impact Hyperparameters

Focus on these first—they have the largest effect on performance:

| Priority | Parameter | Path | Typical Range |
|----------|-----------|------|---------------|
| **1** | Learning rate | `agent.net.optim_spec.lr` | 1e-5 to 1e-3 |
| **2** | Discount factor | `agent.algorithm.gamma` | 0.98-0.999 |
| **3** | GAE lambda | `agent.algorithm.lam` | 0.9-0.99 |
| 4 | Entropy coefficient | `agent.algorithm.entropy_coef` | 0.001-0.1 |
| 5 | Clip epsilon | `agent.algorithm.clip_eps` | 0.1-0.3 |

**Less impactful** (fix based on successful runs): `minibatch_size`, `training_epoch`, network architecture.

{% hint style="info" %}
**Iterative narrowing:** After finding good ranges, narrow the search space and re-run rather than continuing broad exploration.
{% endhint %}

## Grace Period by Environment

The `grace_period` determines minimum frames before ASHA can terminate trials:

| Environment | `grace_period` | Reasoning |
|-------------|----------------|-----------|
| Classic Control | 10000-50000 | Fast learning, quick signal |
| Box2D | 50000-100000 | Medium complexity |
| MuJoCo | 100000-1000000 | Slower learning curves |
| Atari | 500000-1000000 | Need significant training for signal |

## Template Specs

Template specs use `${var}` placeholders for flexibility across similar environments:

```bash
# MuJoCo template
slm-lab run -s env=HalfCheetah-v5 -s max_frame=10e6 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco train

# Atari template
slm-lab run -s env=ALE/Qbert-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari train
```

| Template | Variables | Environments |
|----------|-----------|--------------|
| `ppo_mujoco.json` | `env`, `max_frame` | All 11 MuJoCo |
| `ppo_atari.json` | `env` | All 54 Atari games |

## MuJoCo Tips

### Unified vs Individual Specs

- **`ppo_mujoco`**: HalfCheetah, Walker2d, Humanoid, HumanoidStandup (gamma=0.99, lam=0.95)
- **`ppo_mujoco_longhorizon`**: Reacher, Pusher (gamma=0.997, lam=0.97)
- **Individual specs**: Hopper, Swimmer, Ant—each has environment-specific tuning

### Common Issues

| Problem | Solution |
|---------|----------|
| Reward not improving | Try higher `training_iter` (8-16) for more gradient updates |
| Unstable learning | Try lower learning rate or enable `clip_vloss: true` |
| Large reward variance | Enable `normalize_v_targets: true` for value normalization |

## Atari Tips

### Lambda Variants

Different games benefit from different lambda values:

| Spec Name | Lambda | Best For |
|-----------|--------|----------|
| `ppo_atari` | 0.95 | Strategic games (Qbert, Seaquest) |
| `ppo_atari_lam85` | 0.85 | Mixed games (MsPacman) |
| `ppo_atari_lam70` | 0.70 | Action games (Breakout, Pong) |

**Best practice:** Test all three variants per game; use the best result.

### v5 Environment Difficulty

Gymnasium ALE v5 uses sticky actions (25% repeat probability) per Machado et al. 2018. This makes environments harder than OpenAI Gym v4—expect 10-40% lower scores.

## Troubleshooting

### When Progress Stalls

1. **Check GPU metrics** (`dstack metrics <run-name>`)—low GPU util means bottleneck in env stepping or config issue
2. **Compare with successful specs**—review what worked for similar environments
3. **Look for patterns**—same failure across runs suggests framework issue, not hyperparameters
4. **Research reference implementations**—check [CleanRL](https://github.com/vwxyzjn/cleanrl) or [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) configs
5. **Kill unpromising runs early**—iterate faster with new approaches

### Common Mistakes

| Mistake | Fix |
|---------|-----|
| Too many search dimensions | Focus on 2-3 high-impact parameters per search |
| Skipping multi-seed validation | Always run `max_session=4` before finalizing |
| Using search results directly | Always run final `train` mode with committed spec |
| Inconsistent settings | Verify spec matches standardized settings table |

## Recording Results

After a successful run:

1. **Extract final score** from logs:
   ```bash
   dstack logs my-experiment | grep "trial_metrics"
   # Output: trial_metrics: frame:1.00e+07 | total_reward_ma:15094 | ...
   ```

2. **Pull results**:
   ```bash
   slm-lab pull spec_name
   ```

3. **Update spec defaults** with best hyperparameters

4. **Commit spec file** for reproducibility

## Benchmark Spec Reference

All benchmark specs are in [slm_lab/spec/benchmark/](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/spec/benchmark), organized by algorithm.

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

### A3C (Async)

Asynchronous Advantage Actor-Critic using Hogwild!. See [Async Training](async-training-a3c-hogwild.md).

| Environment | Spec |
|-------------|------|
| Pong | [a3c_gae_pong.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/a3c/a3c_gae_pong.json) |

### Async SAC

SAC with Hogwild! for parallel training. See [Async Training](async-training-a3c-hogwild.md).

| Environment | Spec |
|-------------|------|
| MuJoCo | [async_sac_mujoco.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/async_sac/async_sac_mujoco.json) (template) |

## Performance Results

For scores, training curves, and trained models:

- [Discrete Benchmark](../benchmark-results/discrete-benchmark.md) — Classic Control, Box2D
- [Continuous Benchmark](../benchmark-results/continuous-benchmark.md) — MuJoCo
- [Atari Benchmark](../benchmark-results/atari-benchmark.md) — 54 Atari games
- [Public Benchmark Data](../benchmark-results/public-benchmark-data.md) — HuggingFace download links
