# Search Spec 🔍

The **search spec** enables hyperparameter optimization using [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) with ASHA (Asynchronous Successive Halving Algorithm). ASHA terminates underperforming trials early, focusing compute on promising configurations.

## The Search Spec Structure

Add a **search** section to your spec with `{key}__{space_type}` syntax:

```javascript
{
  "spec_name": {
    "agent": {...},
    "env": {...},
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
      "agent.algorithm.gamma__uniform": [0.95, 0.999],
      "agent.net.optim_spec.lr__loguniform": [1e-5, 1e-3]
    }
  }
}
```

## Searchable Hyperparameters

You can search any spec field using dot notation. Common hyperparameters:

### Algorithm Hyperparameters

| Parameter | Path | Typical Range | Impact |
|-----------|------|---------------|--------|
| Discount factor | `agent.algorithm.gamma` | 0.9-0.999 | High |
| GAE lambda | `agent.algorithm.lam` | 0.7-0.99 | High |
| Learning rate | `agent.net.optim_spec.lr` | 1e-5 to 1e-3 | High |
| Entropy coefficient | `agent.algorithm.entropy_coef_spec.start_val` | 0.001-0.1 | Medium |
| Clip epsilon (PPO) | `agent.algorithm.clip_eps_spec.start_val` | 0.1-0.3 | Medium |
| Time horizon | `agent.algorithm.time_horizon` | 64-2048 | Medium |
| Minibatch size | `agent.algorithm.minibatch_size` | 32-512 | Low |
| Training epochs | `agent.algorithm.training_epoch` | 3-10 | Low |

### Network Hyperparameters

| Parameter | Path | Typical Range | Impact |
|-----------|------|---------------|--------|
| Hidden layers | `agent.net.hid_layers` | [64,64] to [512,512] | Medium |
| Activation | `agent.net.hid_layers_activation` | relu, tanh | Low |
| Gradient clip | `agent.net.clip_grad_val` | 0.5-10 | Low |

{% hint style="info" %}
**Focus on high-impact parameters first.** Learning rate, gamma, and lambda typically have the largest effect on performance.
{% endhint %}

## Search Space Types

| Type | Syntax | Description | Best For |
|------|--------|-------------|----------|
| `uniform` | `[low, high]` | Uniform distribution | Bounded continuous (gamma, lam) |
| `loguniform` | `[low, high]` | Log-uniform distribution | Learning rates, small values |
| `choice` | `[v1, v2, ...]` | Sample from list | Discrete options, architectures |
| `randint` | `[low, high]` | Random integer | Batch sizes, layer counts |

{% hint style="warning" %}
**Note:** `grid_search` is not supported with Optuna. Use `choice` instead for exhaustive enumeration.
{% endhint %}

### Examples

```javascript
"search": {
  // Continuous ranges (preferred for ASHA)
  "agent.algorithm.gamma__uniform": [0.95, 0.999],
  "agent.net.optim_spec.lr__loguniform": [1e-5, 1e-3],

  // Discrete choices
  "agent.algorithm.lam__choice": [0.7, 0.85, 0.95],
  "agent.net.hid_layers__choice": [[64, 64], [256, 256]],

  // Integers
  "agent.algorithm.training_epoch__randint": [3, 10]
}
```

{% hint style="warning" %}
**Prefer continuous distributions** (`uniform`, `loguniform`) over `choice` when possible. Continuous distributions allow ASHA to interpolate and find optimal values, while `choice` only samples from a fixed list.
{% endhint %}

## ASHA Configuration

ASHA (Asynchronous Successive Halving Algorithm) terminates underperforming trials early. Configure it in the meta spec:

```javascript
"meta": {
  "max_session": 1,
  "max_trial": 16,
  "search_resources": {"cpu": 1, "gpu": 0.125},
  "search_scheduler": {
    "grace_period": 100000,
    "reduction_factor": 3
  }
}
```

### ASHA Settings

| Setting | Description | Typical Value |
|---------|-------------|---------------|
| `max_session` | Sessions per trial | **1** (required for fair comparison) |
| `max_trial` | Total trials to run | 8-30 |
| `grace_period` | Minimum frames before first eval | 5-10% of max_frame |
| `reduction_factor` | Keep top 1/N of trials at each rung | 3 (keeps top 1/3) |
| `search_resources.gpu` | GPU fraction per trial | 0.125 (8 trials per GPU) |

{% hint style="info" %}
**How ASHA works:** At each checkpoint, ASHA terminates the bottom 2/3 of trials and continues the top 1/3. A 16-trial search might only run 5-6 trials to completion.
{% endhint %}

### Grace Period by Environment

| Environment | `grace_period` | Reasoning |
|-------------|----------------|-----------|
| Classic Control | 10000-50000 | Fast learning, quick signal |
| Box2D | 50000-100000 | Medium complexity |
| MuJoCo | 100000-1000000 | Slower learning curves |
| Atari | 500000-1000000 | Need significant training for signal |

## Search Budget Sizing

**Rule: ~3-4 trials per search dimension minimum.**

| max_trial | Max Dimensions | Use Case |
|-----------|----------------|----------|
| 8 | 2-3 | Focused refinement |
| 12-16 | 3-4 | Typical search |
| 20 | 5 | Wide exploration |
| 30 | 6-7 | Broad ASHA search |

{% hint style="warning" %}
**Common mistake:** Too many dimensions wastes trials on under-sampled combinations. Search 2-3 parameters at a time, then fix the best values and search others.
{% endhint %}

## Three-Stage Search Process

For robust hyperparameter tuning:

| Stage | Mode | Config | Purpose |
|-------|------|--------|---------|
| **1. ASHA** | `search` | `max_session=1`, `search_scheduler` | Wide exploration |
| **2. Multi** | `search` | `max_session=4`, no scheduler | Validate top configs |
| **3. Final** | `train` | Best hyperparameters | Confirmation run |

1. **ASHA stage**: Quick exploration across many configurations
2. **Multi stage**: Run top 3-5 configs with multiple seeds (no early stopping)
3. **Final stage**: Update spec defaults with best hyperparameters

{% hint style="warning" %}
**For benchmarks:** Do not use search results directly. Always run a final validation with `train` mode and the committed spec file.
{% endhint %}

## Example: PPO Lambda Search on Breakout

[**Breakout**](https://ale.farama.org/environments/breakout/) is a classic Atari benchmark—break bricks by bouncing a ball with a paddle.

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

Results are saved to `data/ppo_breakout_{timestamp}/`:

| File | Contents |
|------|----------|
| `info/experiment_df.csv` | All trial results, sorted best-first |
| `t{N}/` subdirectories | Per-trial session data and models |

### Reading experiment_df.csv

```python
import pandas as pd

df = pd.read_csv('data/ppo_breakout_2026_01_30/info/experiment_df.csv')
print("Best configuration:")
print(df.iloc[0][['agent.algorithm.lam', 'total_reward_ma']])
```

### After Search

1. Check `experiment_df.csv` for top-performing configurations
2. Narrow search range around best values (if needed)
3. Run validation with `max_session=4` (no ASHA)
4. Update spec defaults with final values

## Results

After a search run, SLM Lab generates experiment-level graphs that help you analyze which hyperparameters work best.

### Multi-Trial Graph

The multi-trial graph overlays all trials, showing how different hyperparameter configurations compare:

![Multi-Trial Graph](../.gitbook/assets/a2c_nstep_breakout_multi_trial_graph_mean_returns_vs_frames.png)

Each color represents a different trial (hyperparameter configuration). This quickly shows which settings learn fastest and achieve the highest scores.

### Experiment Variable Graph

The experiment variable graph plots final performance against hyperparameter values:

![Experiment Variable Graph](../.gitbook/assets/a2c_nstep_breakout_experiment_graph.png)

- **X-axis**: Hyperparameter value (e.g., lambda)
- **Y-axis**: Performance metric (strength)
- **Color**: Overall trial quality (darker = better)

This reveals the relationship between hyperparameters and performance—useful for narrowing search ranges.

### Best Configuration

From this lambda search, PPO achieves **327** MA on Breakout-v5 with λ=0.70.

Trained models available on [HuggingFace](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_breakout_2026_01_07_110559).

{% hint style="info" %}
For full benchmarking methodology and results across 54 Atari games, see [Atari Benchmark](../benchmark-results/atari-benchmark.md).
{% endhint %}

## Algorithm-Specific Search Recommendations

Different algorithms have different sensitive hyperparameters:

| Algorithm | High-Impact Parameters | Typical Search |
|-----------|------------------------|----------------|
| **DQN/DDQN** | `lr`, `gamma`, `explore_var_spec.end_step` | Learning rate and exploration schedule |
| **A2C** | `lr`, `gamma`, `lam`, `entropy_coef_spec.start_val` | GAE parameters and entropy |
| **PPO** | `lr`, `gamma`, `lam`, `clip_eps_spec.start_val` | GAE parameters and clipping |
| **SAC** | `lr`, `gamma`, `alpha` (entropy) | Learning rate and entropy coefficient |

### Example Search Blocks by Algorithm

**PPO** (policy gradient):
```javascript
"search": {
  "agent.algorithm.gamma__uniform": [0.98, 0.999],
  "agent.algorithm.lam__uniform": [0.9, 0.98],
  "agent.net.optim_spec.lr__loguniform": [1e-4, 1e-3]
}
```

**DQN** (value-based):
```javascript
"search": {
  "agent.algorithm.gamma__uniform": [0.98, 0.999],
  "agent.net.optim_spec.lr__loguniform": [1e-4, 5e-4],
  "agent.algorithm.explore_var_spec.end_step__randint": [30000, 100000]
}
```

**SAC** (off-policy):
```javascript
"search": {
  "agent.algorithm.gamma__uniform": [0.98, 0.999],
  "agent.net.optim_spec.lr__loguniform": [1e-4, 1e-3],
  "agent.algorithm.training_iter__choice": [4, 8, 16]
}
```

## Finding Search Examples

Benchmark specs often include search blocks from tuning. Check existing specs:

```bash
# Find specs with search blocks
grep -r '"search"' slm_lab/spec/benchmark/

# See a complete search example
cat slm_lab/spec/benchmark/ppo/ppo_cartpole.json
```

{% hint style="info" %}
**Search blocks don't affect train mode.** You can leave a `search` block in a spec—it's only used when running `slm-lab run ... search`. The `train` mode ignores it.
{% endhint %}

## Quick Reference

```bash
# Run ASHA search
slm-lab run spec.json spec_name search

# Check results
cat data/spec_name_*/info/experiment_df.csv

# Validate best config (update spec first, then)
slm-lab run spec.json spec_name train
```
