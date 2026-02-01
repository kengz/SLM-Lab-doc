# Core Concepts

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

The output folder structure:

```
data/ppo_cartpole_2026_01_30_221924/
├── ppo_cartpole_spec.json                      # Original spec
├── ppo_cartpole_t0_spec.json                   # Trial spec (for reproduction)
├── ppo_cartpole_t0_trial_graph_*.png           # Trial graphs (aggregated)
├── ppo_cartpole_t0_trial_metrics_scalar.json   # Trial scalar metrics
├── graph/
│   └── ppo_cartpole_t0_s*_session_graph_*.png  # Per-session graphs
├── info/
│   ├── ppo_cartpole_t0_s*_session_df_train.csv # Training metrics
│   └── ppo_cartpole_t0_s*_session_df_eval.csv  # Evaluation metrics
├── log/                                        # TensorBoard events
└── model/
    ├── ppo_cartpole_t0_s0_net_model.pt         # Session 0 final model
    ├── ppo_cartpole_t0_s0_ckpt-best_net_model.pt  # Session 0 best model
    └── ...                                     # (same for s1, s2, s3)
```

### Naming Convention

Output files follow: `{spec_name}_t{trial}_s{session}_{type}.{ext}`

- `t0` = Trial 0, `s0` = Session 0
- `ckpt-best` = Best checkpoint (highest `total_reward_ma`)
- No prefix = final checkpoint

## Reproducibility

Every experiment in SLM Lab can be reproduced exactly. When you run an experiment, SLM Lab saves:

1. **Trial spec file** (`*_t0_spec.json`) - all hyperparameters with fixed values (no search ranges)
2. **Git SHA** - the exact code version, saved in the trial spec's `meta.git_sha`
3. **Random seeds** - deterministic per session

### Reproducing Results

The trial spec (`ppo_cartpole_t0_spec.json`) includes the git SHA for exact reproduction:

```json
{
  "meta": {
    "git_sha": "a1b2c3d",
    ...
  }
}
```

To reproduce:

```bash
git checkout a1b2c3d
slm-lab run data/ppo_cartpole_2026_01_30_221924/ppo_cartpole_t0_spec.json ppo_cartpole train
```

### Published Benchmark Results

SLM Lab auto-uploads results to [HuggingFace](https://huggingface.co/datasets/SLM-Lab/benchmark) when env vars are configured (see [Train: PPO on CartPole](train-ppo-cartpole.md#auto-upload-to-huggingface)). You can download and replay any published experiment:

```bash
slm-lab list              # List available experiments
slm-lab pull ppo_cartpole # Download
slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole enjoy@data/ppo_cartpole_2026_01_30_221924/ppo_cartpole_t0_spec.json
```

## The Spec File

A **spec file** is a JSON file that completely defines an experiment. From [`ppo_cartpole.json`](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_cartpole.json):

```javascript
{
  "ppo_cartpole": {                    // Spec name - used in CLI and output folders
    "agent": {
      "name": "PPO",                   // Agent name for logging
      "algorithm": {
        "name": "PPO",                 // Algorithm class to use
        "gamma": 0.9769,               // Discount factor (value future rewards)
        "lam": 0.9112,                 // GAE lambda (bias-variance tradeoff)
        "time_horizon": 256,           // Steps collected before each update
        "minibatch_size": 128,         // Batch size for gradient updates
        "training_epoch": 15           // Passes through collected data per update
      },
      "memory": {
        "name": "OnPolicyBatchReplay"  // Memory type (on-policy for PPO)
      },
      "net": {
        "type": "MLPNet",              // Network architecture
        "hid_layers": [64, 64],        // Two hidden layers, 64 units each
        "hid_layers_activation": "tanh",
        "actor_optim_spec": {"name": "Adam", "lr": 0.0005909},  // Policy optimizer
        "critic_optim_spec": {"name": "Adam", "lr": 0.0006352}, // Value optimizer
        "gpu": "auto"                  // Use GPU if available
      }
    },
    "env": {
      "name": "CartPole-v1",           // Gymnasium environment name
      "num_envs": 4,                   // Parallel environments for data collection
      "max_frame": 200000              // Total training frames (stop condition)
    },
    "meta": {
      "max_session": 4,                // Sessions per trial (different seeds)
      "max_trial": 1,                  // Trials per experiment
      "log_frequency": 500,            // Log metrics every N frames
      "eval_frequency": 256            // Evaluate every N frames
    }
  }
}
```

### Key Spec Sections

| Section | Purpose |
|---------|---------|
| **agent.algorithm** | RL algorithm settings (gamma, lam, etc.) |
| **agent.memory** | Experience storage configuration |
| **agent.net** | Neural network architecture and optimizer |
| **env** | Environment name and training budget |
| **meta** | Experiment settings (sessions, trials) |

## What's Next

Continue with these tutorials:

* [Agent Spec](agent-spec-ddqn+per-on-lunarlander.md) - Configure algorithms and networks
* [Env Spec](environment-spec-a2c-on-bipedalwalker.md) - Configure environments for MuJoCo
* [GPU Training](gpu-usage-ppo-on-pong.md) - Train on Atari with GPU
* [Hyperparameter Search](search-spec-ppo-on-breakout.md) - Find optimal settings with ASHA
