# Async Training: A3C Hogwild! ⚡

This tutorial covers asynchronous training using Hogwild!—a technique for parallelizing network training across multiple CPU processes.

{% hint style="info" %}
**Educational Purpose:** A3C Hogwild! is included primarily for learning about async RL architectures. For production training, use **PPO with vectorized environments** (`num_envs`)—it's simpler, faster, and GPU-accelerated.
{% endhint %}

## When to Use Async Training

| Approach | Best For | GPU Support |
|----------|----------|-------------|
| **Vectorized envs** (`num_envs`) | Most cases—simple and efficient | Yes |
| **A3C Hogwild!** | Learning async RL, CPU-bound training | CPU only |

## How Hogwild! Works

[Hogwild!](https://arxiv.org/abs/1106.5730) enables lock-free parallel training by having multiple workers update shared network parameters simultaneously. SLM Lab implements this using [PyTorch multiprocessing](https://pytorch.org/docs/stable/notes/multiprocessing.html) with shared memory.

```
Worker 1 ─┬─→ Shared Network ←─┬─ Worker 3
Worker 2 ─┘                    └─ Worker 4
```

Each worker:
1. Copies the shared network
2. Collects experience from its own environment
3. Computes gradients
4. Pushes gradients to the shared network

{% hint style="warning" %}
**CPU Only:** A3C Hogwild! runs on CPU because PyTorch's `share_memory_()` requires CPU tensors. For GPU-accelerated training, use PPO or A2C with vectorized environments instead.
{% endhint %}

## Meta Spec for Hogwild!

Enable distributed training in the **meta spec**:

```javascript
{
  "a3c_gae_pong": {
    "agent": {
      "net": {
        "gpu": false,  // Required: Hogwild! is CPU-only
        "optim_spec": {
          "name": "GlobalAdam",  // Shared-memory optimizer
          "lr": 0.0007
        }
      }
    },
    "meta": {
      "distributed": "synced",  // or "shared"
      "max_session": 4          // Number of parallel workers
    }
  }
}
```

### Distributed Modes

| Mode | Behavior | Use Case |
|------|----------|----------|
| `"synced"` | Sync parameters after each training step | A3C (on-policy) |
| `"shared"` | Continuous parameter sharing | Off-policy algorithms |
| `false` | Disabled (default) | Standard training |

### Key Requirements

1. **`gpu: false`** — Hogwild! requires CPU tensors for shared memory
2. **`GlobalAdam` or `GlobalRMSprop`** — Special optimizers that support shared state
3. **`max_session > 1`** — Number of parallel workers

## A3C on Pong

The A3C spec at [slm_lab/spec/benchmark/a3c/a3c_gae_pong.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/a3c/a3c_gae_pong.json) demonstrates async training:

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
      "max_session": 4,
      "max_trial": 1
    }
  }
}
```

### Running A3C

```bash
slm-lab run slm_lab/spec/benchmark/a3c/a3c_gae_pong.json a3c_gae_pong train
```

With 4 workers (`max_session: 4`), each running 4 parallel environments (`num_envs: 4`), you get 16 environments collecting experience simultaneously.

## Comparison: Async vs Vectorized

For most use cases, **vectorized environments are simpler and faster**:

```bash
# Recommended: PPO with vectorized envs (GPU-accelerated)
slm-lab run slm_lab/spec/benchmark/ppo/ppo_pong.json ppo_pong train

# Educational: A3C Hogwild (CPU-only)
slm-lab run slm_lab/spec/benchmark/a3c/a3c_gae_pong.json a3c_gae_pong train
```

| Aspect | Vectorized (PPO) | Hogwild (A3C) |
|--------|------------------|---------------|
| **GPU** | Yes | No (CPU only) |
| **Complexity** | Simple | Complex (multiprocessing) |
| **Throughput** | Higher | Lower |
| **Use case** | Production | Learning |

## Historical Context

A3C ([Mnih et al., 2016](https://arxiv.org/abs/1602.01783)) was groundbreaking when GPUs were expensive and CPU parallelism was the main scaling strategy. Today, GPU-accelerated vectorized training (PPO, A2C) is more practical.

SLM Lab includes A3C for:
- Understanding async RL architectures
- Reproducing classic papers
- CPU-only training scenarios

For validated benchmark results, see [Discrete Benchmark](../benchmark-results/discrete-benchmark.md) and [Atari Benchmark](../benchmark-results/atari-benchmark.md)—all using synchronous PPO.
