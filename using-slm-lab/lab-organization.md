# Understanding Experiments 🧪

After running your first training, you'll notice SLM Lab creates folders with names like `ppo_cartpole_t0_s0`. This page explains SLM Lab's experiment hierarchy.

## Sessions, Trials, and Experiments

SLM Lab organizes training into three levels:

| Level | What It Is | Example |
|-------|------------|---------|
| **Session** | One training run with a fixed random seed | Train PPO on CartPole once |
| **Trial** | Multiple sessions with different seeds | Train PPO on CartPole 4 times for reliable results |
| **Experiment** | Multiple trials with different hyperparameters | Find the best learning rate for PPO on CartPole |

**Why multiple seeds?** Deep RL results can vary significantly between runs due to random initialization, environment stochasticity, and exploration noise. Running 4 sessions with different random seeds gives you a reliable average rather than a lucky (or unlucky) single result.

![The graphs for Session, Trial, and Experiment.](<../.gitbook/assets/lab org.png>)

### Mapping to Lab Modes

When using the lab command, different lab modes correspond to different levels:

| Mode | Level | Sessions | Trials | Use Case |
|------|-------|----------|--------|----------|
| **enjoy** | Session | 1 | 1 | Replay trained model |
| **dev** | Trial | 1 | 1 | Quick debugging with rendering |
| **train** | Trial | 4 (default) | 1 | Full training run |
| **search** | Experiment | 1-4 | N | Hyperparameter tuning |

### Concrete Example

Let's trace what happens when you run:

```bash
slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole train
```

1. **Trial starts**: Creates 4 Sessions (controlled by `meta.max_session`)
2. **Each Session**:
   - Gets a different random seed
   - Creates its own Agent and Env
   - Runs until `max_frame` (200,000 frames)
   - Saves checkpoints and metrics
3. **Trial completes**: Aggregates results across all Sessions
4. **Output**: Trial graph showing mean ± std across sessions

The output folder structure looks like:

```
data/ppo_cartpole_2024_01_15_123456/
├── graph/
│   ├── ppo_cartpole_t0_s0_session_graph_*.png  # Session 0 graphs
│   ├── ppo_cartpole_t0_s1_session_graph_*.png  # Session 1 graphs
│   ├── ppo_cartpole_t0_s2_session_graph_*.png  # Session 2 graphs
│   ├── ppo_cartpole_t0_s3_session_graph_*.png  # Session 3 graphs
│   └── ppo_cartpole_t0_trial_graph_*.png       # Trial graph (aggregated)
├── info/
│   ├── ppo_cartpole_t0_s0_session_df.csv       # Session 0 metrics
│   └── ...
├── model/
│   ├── ppo_cartpole_t0_s0_ckpt-best.pt         # Best model for session 0
│   ├── ppo_cartpole_t0_s0_ckpt-last.pt         # Final model for session 0
│   └── ...
└── ppo_cartpole_t0_spec.json                   # Saved spec for reproduction
```

### Naming Convention

Output files follow this pattern: `{spec_name}_t{trial}_s{session}_{type}.{ext}`

- `t0` = Trial index 0
- `s0` = Session index 0
- `ckpt-best` = Best checkpoint (highest `total_reward_ma`)
- `ckpt-last` = Final checkpoint

## Reproducibility

Every experiment in SLM Lab can be reproduced exactly. When you run an experiment, SLM Lab saves:

1. **Spec file** (`*_spec.json`) - all hyperparameters and settings
2. **Git SHA** - the exact code version used (in `info/` folder)
3. **Random seeds** - deterministic per session

### Reproducing Results

To reproduce someone else's results (or your own from months ago):

```bash
# 1. Check out the exact code version
git checkout <git-sha-from-their-experiment>

# 2. Run with their spec file
slm-lab run _ _ enjoy@path/to/their_spec.json
```

{% hint style="success" %}
This makes it easy to share and verify results. If you report results in a paper, others can reproduce them exactly.
{% endhint %}

### Downloading Published Results

SLM Lab benchmark results are published to HuggingFace:

```bash
# List available experiments
slm-lab list

# Download and replay
slm-lab pull ppo_cartpole
slm-lab run _ _ enjoy@data/ppo_cartpole_*/ppo_cartpole_t0_spec.json
```

## The Spec File

A **spec file** is a JSON file that completely defines an experiment. Here's a complete annotated example:

```javascript
{
  "ppo_cartpole": {                    // Spec name - used in output folders
    "agent": {
      "name": "PPO",                   // Agent name for logging
      "algorithm": {
        "name": "PPO",                 // Algorithm class name
        "gamma": 0.99,                 // Discount factor
        "lam": 0.95,                   // GAE lambda
        "time_horizon": 128,           // Steps per update
        "minibatch_size": 64,          // Minibatch size
        "training_epoch": 4            // Epochs per update
      },
      "memory": {
        "name": "OnPolicyBatchReplay"  // Memory class name
      },
      "net": {
        "type": "MLPNet",              // Network class name
        "hid_layers": [64, 64],        // Hidden layer sizes
        "hid_layers_activation": "tanh",
        "optim_spec": {
          "name": "Adam",
          "lr": 0.001
        },
        "gpu": "auto"                  // "auto", true, or false
      }
    },
    "env": {
      "name": "CartPole-v1",           // Gymnasium environment name
      "num_envs": 4,                   // Parallel environments
      "max_frame": 200000              // Total training frames
    },
    "meta": {
      "max_session": 4,                // Sessions per trial
      "max_trial": 1,                  // Trials (for search mode)
      "log_frequency": 500,            // Log every N frames
      "eval_frequency": 256            // Eval every N frames
    }
  }
}
```

### Key Spec Sections

| Section | Purpose | Key Parameters |
|---------|---------|----------------|
| **agent.algorithm** | RL algorithm settings | `gamma`, `lam`, `training_epoch` |
| **agent.memory** | Experience storage | `name`, `batch_size`, `max_size` |
| **agent.net** | Neural network architecture | `type`, `hid_layers`, `optim_spec` |
| **env** | Environment configuration | `name`, `num_envs`, `max_frame` |
| **meta** | Experiment settings | `max_session`, `max_trial` |

## What's Next

Continue with these tutorials:

* [Agent Spec](agent-spec-ddqn+per-on-lunarlander.md) - Configure algorithms and networks
* [Env Spec](environment-spec-a2c-on-bipedalwalker.md) - Configure environments for MuJoCo
* [GPU Training](gpu-usage-ppo-on-pong.md) - Train on Atari with GPU
* [Hyperparameter Search](search-spec-ppo-on-breakout.md) - Find optimal settings with ASHA
