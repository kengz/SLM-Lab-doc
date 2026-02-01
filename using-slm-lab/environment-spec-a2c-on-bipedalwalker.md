# Env Spec 🌍

This tutorial shows how to configure the **env spec** for continuous control environments. We'll train PPO on HalfCheetah—a MuJoCo locomotion task.

## The Env Spec

The environment is specified using the **env** key in a spec file:

```javascript
{
  "spec_name": {
    "agent": {...},
    "env": {
      // Environment name (must be in gymnasium registry)
      "name": str,

      // Number of parallel environment instances
      "num_envs": int,

      // Maximum timesteps per episode (null = use environment default)
      "max_t": int|null,

      // Total training frames
      "max_frame": int,

      // Optional: Online state normalization (recommended for MuJoCo)
      "normalize_obs": bool,

      // Optional: Online reward normalization (recommended for MuJoCo)
      "normalize_reward": bool
    },
    ...
  }
}
```

## Supported Environments

SLM Lab works with any [Gymnasium](https://gymnasium.farama.org/) environment. Common categories:

| Category | Examples | Action Space | Notes |
|----------|----------|--------------|-------|
| **Classic Control** | CartPole-v1, Acrobot-v1 | Discrete | Fast training, good for testing |
| **Box2D** | LunarLander-v3, BipedalWalker-v3 | Discrete/Continuous | Medium complexity |
| **MuJoCo** | HalfCheetah-v5, Humanoid-v5 | Continuous | Physics simulation, use normalization |
| **Atari** | ALE/Qbert-v5, ALE/MsPacman-v5 | Discrete | Image observations, use GPU |

**Environment-specific settings:**

| Environment Type | Recommended Settings |
|-----------------|---------------------|
| Classic/Box2D | `num_envs: 8`, `gpu: auto` |
| MuJoCo | `num_envs: 16`, `normalize_obs: true`, `normalize_reward: true` |
| Atari | `num_envs: 16`, `gpu: auto` (ConvNet benefits from GPU) |

See [Benchmark Specs](benchmark-specs.md) for complete spec files for each environment.

## Example: PPO on HalfCheetah

[**HalfCheetah-v5**](https://gymnasium.farama.org/environments/mujoco/half_cheetah/) is a classic MuJoCo benchmark—a 2D cheetah robot that learns to run forward. It has a 17-dimensional observation space and 6-dimensional continuous action space.

The PPO MuJoCo spec from [slm\_lab/spec/benchmark/ppo/ppo\_mujoco.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_mujoco.json):

{% code title="slm_lab/spec/benchmark/ppo/ppo_mujoco.json (excerpt)" %}
```javascript
{
  "ppo_mujoco": {
    "agent": {
      "name": "PPO",
      "algorithm": {
        "name": "PPO",
        "gamma": 0.99,
        "lam": 0.95,
        "time_horizon": 2048,
        "minibatch_size": 64,
        "training_epoch": 10,
        "normalize_v_targets": true
      },
      "memory": {"name": "OnPolicyBatchReplay"},
      "net": {
        "type": "MLPNet",
        "hid_layers": [256, 256],
        "hid_layers_activation": "tanh",
        "init_fn": "orthogonal_",
        "gpu": "auto"
      }
    },
    "env": {
      "name": "${env}",
      "num_envs": 16,
      "max_frame": "${max_frame}",
      "normalize_obs": true,
      "normalize_reward": true
    },
    "meta": {
      "max_session": 4,
      "max_trial": 1,
      "log_frequency": 10000
    }
  }
}
```
{% endcode %}

Key env settings for MuJoCo:

| Parameter | Value | Why |
|-----------|-------|-----|
| `num_envs: 16` | 16 parallel environments | Faster data collection for on-policy learning |
| `normalize_obs: true` | Normalize observations | MuJoCo observations have varying scales |
| `normalize_reward: true` | Normalize rewards | Stabilizes value function learning |

{% hint style="info" %}
**MuJoCo became free in 2022.** No license needed—Gymnasium includes MuJoCo out of the box.
{% endhint %}

## Running PPO on HalfCheetah

```bash
# Dev mode (quick test with rendering)
slm-lab run -s env=HalfCheetah-v5 -s max_frame=1e5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco dev

# Full training (4M frames)
slm-lab run -s env=HalfCheetah-v5 -s max_frame=4e6 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco train
```

The variable substitution (`-s env=...`) lets you use the same spec for different MuJoCo environments.

### Results

PPO achieves **5852** MA on HalfCheetah-v5 with this configuration.

**Training curves** (average of 4 sessions):

![HalfCheetah Training Curve](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/data/ppo_mujoco_halfcheetah_2026_01_30_230302/ppo_mujoco_halfcheetah_t0_trial_graph_mean_returns_vs_frames.png)

![HalfCheetah Moving Average](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/data/ppo_mujoco_halfcheetah_2026_01_30_230302/ppo_mujoco_halfcheetah_t0_trial_graph_mean_returns_ma_vs_frames.png)

Trained models available on [HuggingFace](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_mujoco_halfcheetah_2026_01_30_230302).

## Using Other Environments

To use a different environment, find a spec for that environment category and modify `env.name`. Spec files are organized by algorithm in `slm_lab/spec/benchmark/`:

### Environment Categories

| Category | Environments | Spec Examples |
|----------|--------------|---------------|
| **Classic Control** | CartPole-v1, Acrobot-v1, Pendulum-v1, MountainCar-v0 | `ppo_cartpole.json`, `dqn_cartpole.json` |
| **Box2D** | LunarLander-v3, BipedalWalker-v3 | `ppo_lunar.json`, `ddqn_per_lunar.json` |
| **MuJoCo** | Hopper-v5, HalfCheetah-v5, Walker2d-v5, Ant-v5, Humanoid-v5, Swimmer-v5, etc. | `ppo_mujoco.json`, `sac_mujoco.json` |
| **Atari** | 54 games (ALE/Qbert-v5, ALE/MsPacman-v5, etc.) | `ppo_atari.json`, `dqn_atari.json` |

### Switching Environments

1. **Find a spec** for your target environment category
2. **Change `env.name`** to the Gymnasium environment name
3. **Adjust settings** as needed (num_envs, max_frame, normalization)

Example—use the same algorithm on different environments:

```bash
# PPO on CartPole (Classic Control)
slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole train

# PPO on LunarLander (Box2D)
slm-lab run slm_lab/spec/benchmark/ppo/ppo_lunar.json ppo_lunar train

# PPO on HalfCheetah (MuJoCo) - uses variable substitution
slm-lab run -s env=HalfCheetah-v5 -s max_frame=4e6 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco train

# PPO on Breakout (Atari) - uses variable substitution
slm-lab run -s env=ALE/Breakout-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari train
```

### Template Specs with Variable Substitution

Some specs use `${var}` placeholders for flexibility. Use `-s var=value` to substitute:

```bash
# MuJoCo template - works for any MuJoCo environment
slm-lab run -s env=Hopper-v5 -s max_frame=2e6 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco train
slm-lab run -s env=Walker2d-v5 -s max_frame=5e6 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco train

# Atari template - works for any ALE game
slm-lab run -s env=ALE/Qbert-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari train
slm-lab run -s env=ALE/MsPacman-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari train
```

### Finding Environment Specs

```bash
# List all benchmark specs
ls slm_lab/spec/benchmark/

# Find specs for a specific environment
grep -r "CartPole" slm_lab/spec/benchmark/
grep -r "LunarLander" slm_lab/spec/benchmark/
grep -r "HalfCheetah" slm_lab/spec/benchmark/
```

{% hint style="info" %}
**Any Gymnasium environment works.** Just set `env.name` to a valid [Gymnasium](https://gymnasium.farama.org/) environment ID. Use the benchmark specs as starting points for hyperparameters.
{% endhint %}

## Standard Settings for Fair Comparison

When comparing algorithms, use consistent environment settings. Different `num_envs` or `max_frame` values make comparisons invalid.

### Recommended Settings by Category

| Category | num_envs | max_frame | log_frequency | Notes |
|----------|----------|-----------|---------------|-------|
| **Classic Control** | 4 | 2e5-3e5 | 500 | Fast training |
| **Box2D** | 8 | 3e5 | 1000 | Medium complexity |
| **MuJoCo** | 16 | 4e6-10e6 | 10000 | Use normalization |
| **Atari** | 16 | 10e6 | 10000 | GPU recommended |

### What to Keep Consistent

When comparing algorithms on the same environment:

| Parameter | Keep Same? | Why |
|-----------|------------|-----|
| `num_envs` | **Yes** | Affects data collection rate and batch statistics |
| `max_frame` | **Yes** | Total training budget must match |
| `max_t` | **Yes** | Episode length affects learning signal |
| `normalize_obs` | **Yes** | Changes observation distribution |
| `normalize_reward` | **Yes** | Changes reward scale |

### Example: Fair Algorithm Comparison

To compare DQN vs PPO on LunarLander fairly:

```bash
# Both use: num_envs=8, max_frame=3e5, max_session=4
slm-lab run slm_lab/spec/benchmark/dqn/dqn_lunar.json dqn_concat_lunar train
slm-lab run slm_lab/spec/benchmark/ppo/ppo_lunar.json ppo_lunar train
```

Check that both specs have matching env settings before comparing results.

{% hint style="warning" %}
**Benchmark specs are pre-configured.** The specs in `slm_lab/spec/benchmark/` use standardized settings for each environment category. When creating custom specs, match these settings for comparable results.
{% endhint %}

## Advanced Env Options

### Environment Kwargs

Any additional keys in the env spec are passed to `gymnasium.make()`:

```javascript
"env": {
  "name": "HalfCheetah-v5",
  "num_envs": 16,
  "max_frame": 4e6,
  "exclude_current_positions_from_observation": false  // Passed to MuJoCo
}
```

### Normalization Details

The normalization wrappers maintain running statistics:

| Option | What It Does | When to Use |
|--------|--------------|-------------|
| `normalize_obs` | Centers observations, scales to unit variance | MuJoCo, continuous control |
| `normalize_reward` | Scales rewards using running std | Environments with varying reward scales |

{% hint style="success" %}
**Gymnasium API:** SLM Lab v5 uses Gymnasium's `(obs, reward, terminated, truncated, info)` return format. This correctly distinguishes task completion (terminated) from time limits (truncated)—important for proper value estimation.
{% endhint %}

Next, we'll use GPU to train on Atari games where image processing is the bottleneck.
