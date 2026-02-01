# Meta Spec ⚙️

The **meta spec** controls experiment-level settings: how many sessions to run, how often to checkpoint, and more.

## The Meta Spec Structure

The meta spec is specified using the **meta** key in a spec file:

```javascript
{
  "spec_name": {
    "agent": {...},
    "env": {...},
    "meta": {
      "max_session": 4,           // Sessions per trial (different random seeds)
      "max_trial": 1,             // Trials per experiment (for search mode)
      "log_frequency": 1000,      // Log training metrics every N frames
      "eval_frequency": 1000,     // Evaluate agent every N frames
      "distributed": false,       // Hogwild! parallel training
      "rigorous_eval": null       // Separate eval environments (slower but precise)
    }
  }
}
```

## Meta Spec Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_session` | int | 4 | Sessions per trial (each with different random seed) |
| `max_trial` | int | 1 | Trials per experiment (used in search mode) |
| `log_frequency` | int | 1000 | Frames between training checkpoints |
| `eval_frequency` | int | 1000 | Frames between evaluation checkpoints |
| `distributed` | bool/str | false | Enable Hogwild! distributed training |
| `rigorous_eval` | int/null | null | Spawn separate eval environments |

### Session and Trial Counts

| Setting | Train Mode | Search Mode |
|---------|------------|-------------|
| `max_session: 1` | Single run (fast, less reliable) | Required for ASHA early stopping |
| `max_session: 4` | Standard (4 seeds, statistically robust) | For validation after ASHA |
| `max_trial: 1` | Single configuration | — |
| `max_trial: 8-16` | — | Typical ASHA search budget |

### Checkpoint Frequencies

| Environment | Typical `log_frequency` | Typical `eval_frequency` |
|-------------|-------------------------|--------------------------|
| Classic Control | 500 | 500 |
| Box2D | 1000 | 1000 |
| MuJoCo | 10000 | 10000 |
| Atari | 10000 | 10000 |

{% hint style="info" %}
**Frequency = frames, not episodes.** With `num_envs=16`, a frequency of 10000 means ~625 steps per environment between checkpoints.
{% endhint %}

### Distributed Training (Hogwild!)

| Value | Behavior |
|-------|----------|
| `false` | Standard sequential training |
| `"shared"` | Sessions share network parameters continuously |
| `"synced"` | Sessions sync parameters after each training step |

{% hint style="warning" %}
**Advanced feature.** Hogwild! requires careful tuning and is typically only beneficial for very large-scale experiments.
{% endhint %}

### Rigorous Evaluation

| Value | Behavior | Use Case |
|-------|----------|----------|
| `null` or `0` | Infer eval scores from training (fast) | Default for most environments |
| `8` | Spawn 8 separate eval environments | When train/eval behavior differs |

## Common Configurations

### Quick Development

For fast iteration during development:

```javascript
"meta": {
  "max_session": 1,
  "max_trial": 1,
  "log_frequency": 500,
  "eval_frequency": 500
}
```

### Standard Training

For reliable results with statistical significance:

```javascript
"meta": {
  "max_session": 4,
  "max_trial": 1,
  "log_frequency": 1000,
  "eval_frequency": 1000
}
```

### ASHA Hyperparameter Search

For efficient search with early stopping:

```javascript
"meta": {
  "max_session": 1,
  "max_trial": 16,
  "log_frequency": 10000,
  "eval_frequency": 10000,
  "search_resources": {"cpu": 1, "gpu": 0.125},
  "search_scheduler": {
    "grace_period": 100000,
    "reduction_factor": 3
  }
}
```

See [Search Spec](search-spec-ppo-on-breakout.md) for details on ASHA configuration.

## Example: Atari Configuration

Atari games have multiple lives per episode. SLM Lab's `TrackReward` wrapper tracks true episodic rewards across lives, so you can use fast evaluation:

```javascript
{
  "ppo_atari": {
    "agent": {...},
    "env": {
      "name": "${env}",
      "num_envs": 16,
      "max_frame": 1e7
    },
    "meta": {
      "distributed": false,
      "rigorous_eval": 0,
      "eval_frequency": 10000,
      "log_frequency": 10000,
      "max_session": 4,
      "max_trial": 1
    }
  }
}
```

{% hint style="info" %}
**`eval_frequency` vs `log_frequency`:** Both are independent. For Atari, they're typically set equal since `TrackReward` handles multi-life scoring automatically.
{% endhint %}

## When to Adjust Meta Settings

| Scenario | Adjustment |
|----------|------------|
| Training too slow | Increase `log_frequency` and `eval_frequency` |
| Need more checkpoints | Decrease frequencies (more disk usage) |
| Results inconsistent | Increase `max_session` to 4 or 8 |
| Hyperparameter tuning | Set `max_session: 1`, increase `max_trial` |
| Very long training | Set high `eval_frequency` to reduce overhead |

## Finding Meta Configurations

All benchmark specs have pre-configured meta settings. Use them as references:

```bash
# See meta settings in any benchmark spec
cat slm_lab/spec/benchmark/ppo/ppo_cartpole.json | grep -A 10 '"meta"'

# Compare meta settings across environment types
cat slm_lab/spec/benchmark/ppo/ppo_cartpole.json  # Classic Control
cat slm_lab/spec/benchmark/ppo/ppo_lunar.json     # Box2D
cat slm_lab/spec/benchmark/ppo/ppo_atari.json     # Atari
```

### Meta Settings by Environment Type

| Environment | `max_session` | `log_frequency` | `eval_frequency` | Why |
|-------------|---------------|-----------------|------------------|-----|
| Classic Control | 4 | 500 | 500 | Fast training, frequent checkpoints useful |
| Box2D | 4 | 1000 | 1000 | Medium training time |
| MuJoCo | 4 | 10000 | 10000 | Long training, reduce I/O overhead |
| Atari | 4 | 10000 | 10000 | Very long training (10M frames) |

{% hint style="info" %}
**Benchmark specs are pre-configured.** The specs in `slm_lab/spec/benchmark/` use appropriate meta settings for each environment. Copy and modify them for your experiments.
{% endhint %}
