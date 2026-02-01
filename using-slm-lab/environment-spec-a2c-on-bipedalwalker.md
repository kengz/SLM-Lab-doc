# Env Spec: PPO on HalfCheetah 🏃

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

## PPO on HalfCheetah

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

## Other MuJoCo Environments

The same spec works for other MuJoCo tasks:

```bash
# Simple locomotion
slm-lab run -s env=Walker2d-v5 -s max_frame=4e6 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco train

# Complex locomotion (needs more frames)
slm-lab run -s env=Humanoid-v5 -s max_frame=10e6 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco train
```

See [Continuous Benchmark](../benchmark-results/continuous-benchmark.md) for results across all 11 MuJoCo environments.

## Env Spec for Other Environment Types

### Atari (Discrete, Image-Based)

```javascript
"env": {
  "name": "ALE/Pong-v5",
  "num_envs": 16,
  "max_frame": 1e7,
  "life_loss_info": true  // Continue after life loss
}
```

Gymnasium's ALE wrapper handles frame preprocessing automatically (grayscale, 84x84 resize, frame stacking).

{% hint style="warning" %}
**Atari requires GPU** for reasonable training speed due to the ConvNet. See [GPU Training](gpu-usage-ppo-on-pong.md).
{% endhint %}

### Classic Control (Discrete, Vector)

```javascript
"env": {
  "name": "CartPole-v1",
  "num_envs": 4,
  "max_frame": 200000
}
```

Simple environments need fewer parallel envs and frames.

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
