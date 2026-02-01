# Hyperparameter Search with ASHA

## Overview

SLM Lab v5 uses [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) with ASHA (Asynchronous Successive Halving Algorithm) for efficient hyperparameter search. ASHA terminates underperforming trials early, focusing compute on promising configurations.

In this tutorial, we'll search for optimal **lambda** (λ) values for PPO on Breakout.

[**Breakout**](https://gymnasium.farama.org/environments/atari/breakout/) is a classic Atari benchmark—break bricks by bouncing a ball with a paddle. Expert human performance is ~30-40 points; optimal RL agents can exceed 400.

{% hint style="info" %}
**v5 Note:** Gymnasium ALE environments (v5) use stricter action handling and deterministic frame skipping. Scores may differ from older OpenAI Gym benchmarks. See [Gymnasium ALE docs](https://gymnasium.farama.org/environments/atari/) for details.
{% endhint %}

{% hint style="info" %}
**RL hyperparameters:**
- **gamma** (γ): Discount factor - how much to value future rewards (0.99 = care about future, 0.9 = focus on near-term)
- **lambda** (λ): GAE parameter - tradeoff between bias and variance in advantage estimation
{% endhint %}

## Search Syntax

Add a **search** section to your spec with `{key}__{space_type}` syntax:

```javascript
{
  "spec_name": {
    "agent": {...},
    "env": {...},
    "meta": {
      "max_session": 1,
      "max_trial": 16,
      "search_scheduler": {
        "grace_period": 100000,
        "reduction_factor": 3
      }
    },
    "search": {
      "agent.algorithm.gamma__uniform": [0.95, 0.999],
      "agent.net.optim_spec.lr__loguniform": [1e-5, 1e-3]
    }
  }
}
```

### Search Space Types

| space\_type | value | description | when to use |
|-------------|-------|-------------|-------------|
| uniform | `[low, high]` | Uniform distribution | Bounded params (gamma, lam) |
| loguniform | `[low, high]` | Log-uniform distribution | Learning rates, small values |
| choice | `[v1, v2, ...]` | Sample from list | Discrete options, architecture |
| randint | `[low, high]` | Random integer | Batch sizes, layer counts |
| grid\_search | `[v1, v2, ...]` | Exhaustive grid | Small grids only (multiplies trials) |

Examples:
* `"gamma__uniform": [0.95, 0.999]` - bounded continuous (recommended)
* `"lr__loguniform": [1e-5, 1e-3]` - log-scale for learning rates
* `"lam__choice": [0.7, 0.85, 0.95]` - specific values to compare

{% hint style="warning" %}
**Prefer continuous distributions** (`uniform`, `loguniform`) over `choice` when possible. Continuous distributions allow ASHA to interpolate and find optimal values, while `choice` only samples from a fixed list.
{% endhint %}

## ASHA Early Stopping

The key v5 feature for efficient search. Add `search_scheduler` to your meta spec:

```javascript
{
  "meta": {
    "max_session": 1,
    "max_trial": 16,
    "search_resources": {"cpu": 1, "gpu": 0.125},
    "search_scheduler": {
      "grace_period": 100000,
      "reduction_factor": 3
    }
  }
}
```

**Key settings:**

* **max_session: 1** - Single session per trial (required for ASHA to compare fairly)
* **max_trial: 16** - Total trials to run
* **grace_period** - Minimum frames before first evaluation (allow learning to start)
* **reduction_factor: 3** - Keep top 1/3 of trials at each rung

ASHA evaluates trials at checkpoints and terminates the bottom 2/3, focusing resources on promising runs. A 16-trial search might only run 5-6 trials to completion.

### Search Budget Sizing

**Rule: ~3-4 trials per search dimension minimum.**

| max_trial | Max Dimensions | Use case |
|-----------|----------------|----------|
| 8 | 2-3 | Very focused search |
| 12-16 | 3-4 | Typical refinement |
| 20 | 5 | Wide exploration |
| 30 | 6-7 | Broad ASHA search |

{% hint style="warning" %}
**Common mistake:** Too many dimensions wastes trials on under-sampled combinations. Focus on high-impact hyperparameters first:
- **Most impactful:** Learning rates, gamma, lam
- **Less impactful:** minibatch_size, training_epoch (fix these based on successful runs)
{% endhint %}

### After Search: Narrowing

After analyzing search results:
1. Check `experiment_df.csv` for top-performing configurations
2. Narrow the search range around best values
3. Re-run with tighter bounds if needed
4. Update spec defaults with final values

## Three-Stage Search Process

For robust hyperparameter tuning, use this workflow:

| Stage | Mode | Config | Purpose |
|-------|------|--------|---------|
| ASHA | `search` | `max_session=1`, `search_scheduler` | Wide exploration |
| Multi | `search` | `max_session=4`, no scheduler | Validate top configs |
| Final | `train` | Best hyperparameters | Confirmation run |

1. **ASHA stage**: Quick exploration across many configurations
2. **Multi stage**: Run top 3-5 configs with multiple seeds (no early stopping)
3. **Final stage**: Update spec defaults with best hyperparameters

## Example: PPO Lambda Search

From [slm\_lab/spec/experimental/ppo/ppo\_lam\_search.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/experimental/ppo/ppo_lam_search.json):

{% code title="slm_lab/spec/experimental/ppo/ppo_lam_search.json (excerpt)" %}
```javascript
{
  "ppo_breakout": {
    "agent": {
      "name": "PPO",
      "algorithm": {
        "name": "PPO",
        "gamma": 0.99,
        "lam": 0.7,
        "time_horizon": 128,
        "minibatch_size": 256,
        "training_epoch": 4
      },
      "memory": {"name": "OnPolicyBatchReplay"},
      "net": {
        "type": "ConvNet",
        "shared": true,
        "gpu": "auto"
      }
    },
    "env": {
      "name": "ALE/Breakout-v5",
      "num_envs": 16,
      "max_frame": 1e7
    },
    "meta": {
      "max_session": 1,
      "max_trial": 16,
      "log_frequency": 10000,
      "eval_frequency": 10000,
      "search_resources": {"cpu": 1, "gpu": 0.125},
      "search_scheduler": {
        "grace_period": 500000,
        "reduction_factor": 3
      }
    },
    "search": {
      "agent.algorithm.lam__choice": [0.5, 0.7, 0.9, 0.95, 0.97, 0.99]
    }
  }
}
```
{% endcode %}

## Running the Search

```bash
slm-lab run slm_lab/spec/experimental/ppo/ppo_lam_search.json ppo_breakout search
```

Ray Tune queues trials and runs them as resources free up. With `gpu: 0.125`, 8 trials run in parallel on a single GPU.

## Analyzing Results

Results are saved to `data/ppo_breakout_{ts}/` with:
* `experiment_df.csv` - all trial results, sorted best-first
* Per-trial subdirectories with session data

The experiment produces comparison graphs:

![Experiment graph comparing lambda values](https://raw.githubusercontent.com/kengz/SLM-Lab/master/docs/plots/ALE_Breakout-v5_multi_trial_graph_mean_returns_vs_frames.png)

![Moving average comparison](https://raw.githubusercontent.com/kengz/SLM-Lab/master/docs/plots/ALE_Breakout-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

PPO achieves **395** MA on Breakout-v5 with λ=0.70. Trained models are available on [HuggingFace](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_breakout_2026_01_06_182709).

{% hint style="info" %}
For full benchmarking methodology and results across 54 Atari games, see [Atari Benchmark](../benchmark-results/atari-benchmark.md).
{% endhint %}
