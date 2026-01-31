# Lab Command

The `slm-lab` CLI is your primary interface to SLM Lab. This page covers all commands and options.

## Command Structure

```bash
slm-lab [command] [options] [spec_file] [spec_name] [mode]
```

## Quick Reference

```bash
# Run default (PPO CartPole)
slm-lab run

# Run with rendering
slm-lab run --render

# Run specific experiment
slm-lab run slm_lab/spec/benchmark/ppo/ppo_lunar.json ppo_lunar train

# Hyperparameter search
slm-lab run slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari search

# Resume training
slm-lab run spec.json spec_name train@latest

# Replay trained model
slm-lab run _ _ enjoy@data/ppo_cartpole_*/ppo_cartpole_t0_spec.json

# Remote training
slm-lab run-remote --gpu spec.json spec_name train -n my-run

# Pull results from HuggingFace
slm-lab pull ppo_cartpole

# List available experiments
slm-lab list
```

## Commands

### `slm-lab run`

Run a training experiment.

```bash
slm-lab run [options] [spec_file] [spec_name] [mode]
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `spec_file` | Path to JSON spec file | `slm_lab/spec/benchmark/ppo/ppo_cartpole.json` |
| `spec_name` | Name of spec within file | `ppo_cartpole` |
| `mode` | Execution mode (see below) | `dev` |

**Modes:**

| Mode | Description | Sessions | Rendering | Saves Results |
|------|-------------|----------|-----------|---------------|
| `dev` | Development/debugging | 1 | Yes | No |
| `train` | Full training | 4 (configurable) | No | Yes |
| `search` | Hyperparameter search | 1 per trial | No | Yes |
| `enjoy@{path}` | Replay trained model | 1 | Yes | No |
| `train@{path}` | Resume training | From checkpoint | No | Yes |

**Examples:**

```bash
# Development mode with rendering (default)
slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole dev

# Full training (4 sessions, different random seeds)
slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole train

# Hyperparameter search
slm-lab run slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari search

# Resume from latest run
slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole train@latest

# Resume from specific run
slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole train@data/ppo_cartpole_2024_01_15_123456

# Replay trained model (placeholders for spec_file and spec_name)
slm-lab run _ _ enjoy@data/ppo_cartpole_2024_01_15_123456/ppo_cartpole_t0_spec.json
```

### `slm-lab run-remote`

Run experiments on cloud GPUs via dstack.

```bash
slm-lab run-remote [options] spec_file spec_name mode -n run_name
```

**Options:**

| Option | Description |
|--------|-------------|
| `--gpu` | Use GPU instance (recommended) |
| `-n, --name` | Name for the dstack run (required) |
| `-s, --set` | Variable substitution |

**Examples:**

```bash
# Train on cloud GPU
source .env && slm-lab run-remote --gpu slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari train -n ppo-breakout

# Search on cloud
source .env && slm-lab run-remote --gpu slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari search -n ppo-search

# With variable substitution
source .env && slm-lab run-remote --gpu -s env=ALE/Pong-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari train -n ppo-pong
```

### `slm-lab list`

List experiments on HuggingFace.

```bash
slm-lab list
```

### `slm-lab pull`

Download experiment from HuggingFace.

```bash
slm-lab pull <spec_name>
```

**Example:**

```bash
slm-lab pull ppo_cartpole
# Downloads to data/ppo_cartpole_*/
```

### `slm-lab push`

Upload experiment to HuggingFace.

```bash
slm-lab push <data_folder>
```

**Example:**

```bash
slm-lab push data/ppo_cartpole_2024_01_15_123456
```

## Global Options

These options work with any command:

| Option | Description | Example |
|--------|-------------|---------|
| `-s, --set KEY=VALUE` | Set spec variables | `-s env=Hopper-v5` |
| `--render` | Enable environment rendering | `--render` |
| `--log-level LEVEL` | Set logging level | `--log-level DEBUG` |
| `--cuda-offset N` | GPU device offset | `--cuda-offset 4` |
| `--upload-hf` | Upload results to HuggingFace | `--upload-hf` |
| `--keep N` | Trials to keep after search | `--keep 3` |
| `--profile` | Enable performance profiling | `--profile` |

## Variable Substitution

Template specs use `${var}` placeholders. Substitute with `-s`:

```bash
# Single variable
slm-lab run -s env=ALE/Breakout-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari train

# Multiple variables
slm-lab run -s env=Hopper-v5 -s max_frame=2e6 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco train
```

**In the spec file:**

```javascript
{
  "ppo_mujoco": {
    "env": {
      "name": "${env}",
      "max_frame": "${max_frame}"
    }
  }
}
```

## Common Workflows

### Train and Evaluate

```bash
# 1. Full training
slm-lab run slm_lab/spec/benchmark/ppo/ppo_lunar.json ppo_lunar train

# 2. Check results
ls data/ppo_lunar_*/

# 3. Replay best model
slm-lab run _ _ enjoy@data/ppo_lunar_*/ppo_lunar_t0_spec.json
```

### Hyperparameter Search

```bash
# 1. Run ASHA search
slm-lab run slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari search

# 2. Check best trial
cat data/ppo_atari_*/info/experiment_df.csv

# 3. Train with best params (update spec first)
slm-lab run slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari train
```

### Cloud Training

```bash
# 1. Source credentials
source .env

# 2. Launch remote job
slm-lab run-remote --gpu slm_lab/spec/benchmark/ppo/ppo_hopper.json ppo_hopper train -n ppo-hopper

# 3. Monitor
dstack ps
dstack logs ppo-hopper

# 4. Pull results when complete
slm-lab pull ppo_hopper
```

### Benchmark Multiple Environments

```bash
# Using template spec with variable substitution
for env in ALE/Pong-v5 ALE/Breakout-v5 ALE/Qbert-v5; do
    slm-lab run -s env=$env slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari train
done
```

## The Spec File

A spec file is a JSON document defining your experiment:

```javascript
{
  "ppo_cartpole": {                    // Spec name (used in commands)
    "agent": {
      "name": "PPO",                   // Agent name for logging
      "algorithm": {
        "name": "PPO",                 // Algorithm class
        "gamma": 0.99,                 // Discount factor
        "lam": 0.95,                   // GAE lambda
        "time_horizon": 128,           // Steps per update
        "minibatch_size": 64,          // Minibatch size
        "training_epoch": 4            // Epochs per update
      },
      "memory": {
        "name": "OnPolicyBatchReplay"  // Memory type
      },
      "net": {
        "type": "MLPNet",              // Network type
        "hid_layers": [64, 64],        // Hidden layer sizes
        "hid_layers_activation": "tanh",
        "gpu": "auto"                  // GPU usage
      }
    },
    "env": {
      "name": "CartPole-v1",           // Gymnasium environment
      "num_envs": 4,                   // Parallel environments
      "max_frame": 200000              // Total training frames
    },
    "meta": {
      "max_session": 4,                // Sessions per trial
      "max_trial": 1,                  // Trials (for search)
      "log_frequency": 500,            // Log every N frames
      "eval_frequency": 256            // Eval every N frames
    }
  }
}
```

See [Lab Organization](lab-organization.md) for detailed spec documentation.

## Output Structure

After a training run, results are saved to `data/{spec_name}_{timestamp}/`:

```
data/ppo_cartpole_2024_01_15_123456/
├── graph/                      # Training curves (PNG)
│   ├── *_session_graph_*.png   # Per-session plots
│   └── *_trial_graph_*.png     # Aggregated plots
├── info/                       # Metrics data
│   ├── *_session_df.csv        # Time series
│   └── *_trial_metrics.json    # Summary stats
├── log/                        # TensorBoard events
├── model/                      # PyTorch checkpoints
│   ├── *_ckpt-best.pt          # Best model
│   └── *_ckpt-last.pt          # Final model
└── *_spec.json                 # Saved spec (for reproduction)
```

See [Data Locations](../analyzing-results/analytics.md) for details.
