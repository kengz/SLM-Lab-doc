# Async Training: Hogwild! ⚡

This tutorial covers asynchronous training using Hogwild!—a technique for parallelizing network training across multiple processes with shared parameters.

{% hint style="info" %}
**Educational Purpose:** Hogwild! is included primarily for learning about async RL architectures. For production training, use **PPO with vectorized environments** (`num_envs`)—it's simpler and more efficient.
{% endhint %}

## How Hogwild! Works

[Hogwild!](https://arxiv.org/abs/1106.5730) enables lock-free parallel training by having multiple workers update shared network parameters simultaneously. SLM Lab implements this using [PyTorch multiprocessing](https://pytorch.org/docs/stable/notes/multiprocessing.html) with shared memory.

```
Worker 1 ─┬─→ Shared Global Network ←─┬─ Worker 3
Worker 2 ─┘        (CPU)              └─ Worker 4
```

Each worker:
1. Collects experience from its own environment
2. Computes gradients on its local network
3. Pushes gradients to the shared global network
4. Pulls updated weights from global network

{% hint style="warning" %}
**Global Networks on CPU:** PyTorch's `share_memory_()` requires CPU tensors, so global networks are automatically moved to CPU. Local worker networks can still use GPU for forward/backward passes, but gradient sync happens on CPU.
{% endhint %}

## Meta Spec for Hogwild!

Enable distributed training in the **meta spec**:

```javascript
{
  "meta": {
    "distributed": "synced",  // or "shared"
    "max_session": 4          // Number of parallel workers
  }
}
```

### Distributed Modes

| Mode | Behavior | Use Case |
|------|----------|----------|
| `"synced"` | Sync parameters after each training step | A3C (on-policy) |
| `"shared"` | Continuous parameter sharing | Async SAC (off-policy) |
| `false` | Disabled (default) | Standard training |

### Key Requirements

- **`GlobalAdam` or `GlobalRMSprop`** — Optimizers that support shared state across processes
- **`max_session > 1`** — Number of parallel workers

## A3C on Pong

A3C ([Mnih et al., 2016](https://arxiv.org/abs/1602.01783)) uses `"synced"` mode for on-policy training.

**Spec:** [slm_lab/spec/benchmark/a3c/a3c_gae_pong.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/a3c/a3c_gae_pong.json)

```javascript
{
  "a3c_gae_pong": {
    "agent": {
      "name": "A3C",
      "algorithm": {
        "name": "ActorCritic",
        "gamma": 0.99,
        "lam": 0.95,
        "training_frequency": 32
      },
      "memory": {"name": "OnPolicyBatchReplay"},
      "net": {
        "type": "ConvNet",
        "shared": true,
        "conv_hid_layers": [[32, 8, 4, 0, 1], [64, 4, 2, 0, 1], [32, 3, 1, 0, 1]],
        "fc_hid_layers": [512],
        "actor_optim_spec": {"name": "GlobalAdam", "lr": 0.0007},
        "critic_optim_spec": {"name": "GlobalAdam", "lr": 0.0007},
        "gpu": false
      }
    },
    "env": {
      "name": "ALE/Pong-v5",
      "num_envs": 4,
      "max_frame": 5e5
    },
    "meta": {
      "distributed": "synced",
      "max_session": 4
    }
  }
}
```

**Run:**
```bash
slm-lab run slm_lab/spec/benchmark/a3c/a3c_gae_pong.json a3c_gae_pong train
```

## Async SAC on Humanoid

For off-policy algorithms like SAC, use `"shared"` mode for continuous parameter sharing.

**Spec:** [slm_lab/spec/benchmark/async_sac/async_sac_mujoco.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/async_sac/async_sac_mujoco.json)

```javascript
{
  "async_sac_humanoid": {
    "agent": {
      "name": "SoftActorCritic",
      "algorithm": {
        "name": "SoftActorCritic",
        "gamma": 0.99,
        "training_frequency": 1
      },
      "memory": {
        "name": "Replay",
        "batch_size": 256,
        "max_size": 200000,
        "use_cer": true
      },
      "net": {
        "type": "MLPNet",
        "hid_layers": [256, 256],
        "optim_spec": {"name": "GlobalAdam", "lr": 5e-05},
        "gpu": "auto"
      }
    },
    "env": {
      "name": "Humanoid-v5",
      "num_envs": 8,
      "max_frame": 5e7
    },
    "meta": {
      "distributed": "shared",
      "max_session": 16
    }
  }
}
```

**Run:**
```bash
slm-lab run slm_lab/spec/benchmark/async_sac/async_sac_mujoco.json async_sac_humanoid train
```

With 16 parallel sessions, a 50M frame run completes much faster than sequential training.

{% hint style="info" %}
**Frame counting:** The x-axis shows per-session frames. Total frames = per-session × max_session.
{% endhint %}

## Historical Results (v4)

These graphs are from v4 async SAC training:

![Async SAC Humanoid returns](../.gitbook/assets/async_sac_humanoid_t0_trial_graph_mean_returns_vs_frames.png)

![Async SAC Humanoid moving average](../.gitbook/assets/async_sac_humanoid_t0_trial_graph_mean_returns_ma_vs_frames.png)

For validated v5 Humanoid results using synchronous PPO, see [Continuous Benchmark](../benchmark-results/continuous-benchmark.md)—PPO achieves **3774** on Humanoid-v5.

## Comparison: Async vs Vectorized

For most use cases, **vectorized environments are simpler and faster**:

| Aspect | Vectorized (`num_envs`) | Hogwild! (`distributed`) |
|--------|-------------------------|--------------------------|
| **Parallelism** | Environment stepping | Network training |
| **Complexity** | Simple | Complex (multiprocessing) |
| **GPU** | Full GPU acceleration | Global nets on CPU |
| **Use case** | Production | Learning, CPU-bound |

```bash
# Recommended: PPO with vectorized envs
slm-lab run -s env=ALE/Pong-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari train

# Educational: A3C Hogwild
slm-lab run slm_lab/spec/benchmark/a3c/a3c_gae_pong.json a3c_gae_pong train
```

## When Hogwild! Helps

Hogwild! can help when:
- Network training is the bottleneck (not environment stepping)
- You have many CPU cores available
- Learning about async RL architectures

For most RL workloads, environment stepping is the bottleneck, so vectorized environments (`num_envs`) are more effective.

## Historical Context

A3C was groundbreaking when GPUs were expensive and CPU parallelism was the main scaling strategy. Today, GPU-accelerated vectorized training (PPO, A2C) is more practical for most use cases.

SLM Lab includes async training for:
- Understanding async RL architectures
- Reproducing classic papers
- CPU-only training scenarios
