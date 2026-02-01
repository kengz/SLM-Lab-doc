# CLI Reference 💻

The `slm-lab` CLI is your primary interface for running experiments. For hands-on tutorials, start with [Train: PPO on CartPole](train-ppo-cartpole.md).

## Getting Help

The CLI has built-in help for all commands:

```bash
slm-lab --help              # List all commands
slm-lab run --help          # Help for specific command
slm-lab run-remote --help   # Help for remote training
```

{% hint style="info" %}
**The `--help` flag is the authoritative reference.** This documentation mirrors the CLI help—when in doubt, check `--help` for the latest options.
{% endhint %}

## Commands Overview

```
╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ run          Run SLM-Lab experiments locally                                 │
│ run-remote   Launch experiment on dstack with auto HF upload                 │
│ pull         Pull experiment results from HuggingFace                        │
│ push         Push local experiment to HuggingFace                            │
│ list         List experiments available on HuggingFace                       │
│ plot         Plot benchmark comparison graphs                                │
│ plot-list    List available experiments in data folder                       │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## `slm-lab run`

Run experiments locally.

```bash
slm-lab run [OPTIONS] [SPEC_FILE] [SPEC_NAME] [MODE]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `SPEC_FILE` | `slm_lab/spec/benchmark/ppo/ppo_cartpole.json` | JSON spec file path |
| `SPEC_NAME` | `ppo_cartpole` | Spec name within the file |
| `MODE` | `dev` | Execution mode: `dev`, `train`, `search`, `enjoy`, `eval` |

### Modes

| Mode | Sessions | Rendering | Saves | Use Case |
|------|----------|-----------|-------|----------|
| `dev` | 1 | Yes | No | Development/debugging |
| `train` | 4 (configurable) | No | Yes | Full training runs |
| `search` | 1 per trial | No | Yes | Hyperparameter search |
| `enjoy@{path}` | 1 | Yes | No | Replay trained model |
| `eval@{path}` | 1 | No | No | Evaluate model (no rendering) |
| `train@{path}` | From checkpoint | No | Yes | Resume training |

### Options

| Option | Description |
|--------|-------------|
| `-s, --set KEY=VALUE` | Set spec variables (can use multiple times) |
| `--render` | Enable environment rendering |
| `--log-level LEVEL` | Logging: DEBUG, INFO, WARNING, ERROR |
| `--cuda-offset N` | GPU device offset (default: 0) |
| `--upload-hf` | Upload to HuggingFace after training |
| `--keep N` | Trials to keep after search (default: 3, -1 to disable) |
| `--profile` | Enable performance profiling |
| `--log-extra` | Enable extra metrics (strength, stability, efficiency) |
| `--stop-ray` | Stop all Ray processes |
| `--no-optimize-perf` | Disable auto CPU/GPU optimization |

### Examples

```bash
# Default: PPO CartPole in dev mode
slm-lab run

# With rendering
slm-lab run --render

# Full training
slm-lab run slm_lab/spec/benchmark/ppo/ppo_lunar.json ppo_lunar train

# Hyperparameter search
slm-lab run slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari search

# Variable substitution
slm-lab run -s env=ALE/Qbert-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari train

# Multiple variables
slm-lab run -s env=Hopper-v5 -s max_frame=2e6 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco train

# Resume from latest
slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole train@latest

# Resume from specific run
slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole train@data/ppo_cartpole_2026_01_30_221924

# Replay trained model
slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole enjoy@data/ppo_cartpole_2026_01_30_221924/ppo_cartpole_t0_spec.json
```

## `slm-lab run-remote`

Launch experiments on cloud GPUs via dstack with automatic HuggingFace upload.

```bash
slm-lab run-remote [OPTIONS] SPEC_FILE SPEC_NAME [MODE]
```

{% hint style="info" %}
**Supported modes:** `run-remote` only supports `train` and `search` modes. Use local `run` for `dev`, `enjoy`, and `eval` modes.
{% endhint %}

### Hardware Configuration

The CLI auto-selects hardware based on mode and `--gpu` flag:

| Config | Hardware | Cost | Use Case |
|--------|----------|------|----------|
| cpu-train | 4-8 CPUs, 16-32GB | ~$0.13/hr | MLP envs validation |
| cpu-search | 8-16 CPUs, 32-64GB | ~$0.20/hr | MLP envs ASHA search |
| gpu-train | L4 GPU | ~$0.39/hr | Image envs validation |
| gpu-search | L4 GPU + 8-16 CPUs | ~$0.50/hr | Image envs ASHA search |

### Options

| Option | Description |
|--------|-------------|
| `-n, --name NAME` | Run name (default: spec_name) |
| `-s, --set KEY=VALUE` | Set spec variables |
| `--gpu` | Use GPU hardware (default: CPU) |

### Examples

```bash
# Source credentials first
source .env

# CPU training (default)
slm-lab run-remote slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole train -n ppo-cartpole

# GPU training (for ConvNet envs)
slm-lab run-remote --gpu slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari train -n ppo-qbert

# GPU search with variable substitution
slm-lab run-remote --gpu -s env=ALE/Qbert-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari search -n qbert-search
```

{% hint style="warning" %}
**Always `source .env` first.** This sets `HF_TOKEN` and `HF_REPO` for automatic result upload.
{% endhint %}

### Monitoring Remote Runs

```bash
dstack ps                    # List running jobs
dstack logs <run-name>       # View logs
dstack metrics <run-name>    # Check GPU/CPU utilization
dstack stop <run-name> -y    # Stop a run
```

## `slm-lab list`

List experiments available on HuggingFace.

```bash
slm-lab list [PATTERN]
```

### Examples

```bash
slm-lab list              # List all experiments
slm-lab list ppo          # Filter by pattern
slm-lab list sac_mujoco   # Filter specific algorithm/env
```

## `slm-lab pull`

Download experiment results from HuggingFace to local `data/` folder.

```bash
slm-lab pull [OPTIONS] PATTERN
```

### Options

| Option | Description |
|--------|-------------|
| `-l, --list` | List matching experiments without downloading |

### Examples

```bash
slm-lab pull ppo_cartpole           # Pull matching experiments
slm-lab pull ppo_cartpole_2026      # Pull specific experiment
slm-lab pull --list ppo             # List without downloading
```

## `slm-lab push`

Upload local experiment to HuggingFace.

```bash
slm-lab push PREDIR
```

### Examples

```bash
slm-lab push data/ppo_cartpole_2026_01_30_221924
```

## `slm-lab plot`

Generate benchmark comparison graphs from experiment data.

```bash
slm-lab plot [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `-f, --folders FOLDERS` | Comma-separated data folder names (required) |
| `-t, --title TITLE` | Plot title (auto-detected from spec if omitted) |
| `-d, --data-folder PATH` | Base data folder (default: `data`) |
| `-o, --output PATH` | Output folder (default: `docs/plots`) |
| `--no-legend` | Hide legend |

### Examples

```bash
# Compare two experiments
slm-lab plot -f ppo_cartpole_2026_01_11,a2c_gae_cartpole_2026_01_11

# Custom title
slm-lab plot -t "CartPole Comparison" -f ppo_cartpole_2026,a2c_cartpole_2026

# Different output folder
slm-lab plot -f folder1,folder2 -o my_plots/
```

## `slm-lab plot-list`

List experiments in data folder that have metrics ready for plotting.

```bash
slm-lab plot-list [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `-d, --data-folder PATH` | Data folder path (default: `data`) |

## Variable Substitution

Template specs use `${var}` placeholders for flexibility:

**In the spec file:**

```json
{
  "ppo_mujoco": {
    "env": {
      "name": "${env}",
      "max_frame": "${max_frame}"
    }
  }
}
```

**On the command line:**

```bash
slm-lab run -s env=HalfCheetah-v5 -s max_frame=10e6 spec.json ppo_mujoco train
```

| Common Variables | Example |
|------------------|---------|
| Environment name | `-s env=Hopper-v5` |
| Training budget | `-s max_frame=1e7` |
| Hyperparameters | `-s gamma=0.99 -s lam=0.95` |

{% hint style="info" %}
**Type handling:** Numeric values (including scientific notation like `1e7`) are automatically converted.
{% endhint %}

## Output Structure

Results are saved to `data/{spec_name}_{timestamp}/`:

```
data/ppo_cartpole_2026_01_30_221924/
├── ppo_cartpole_spec.json                    # Original spec
├── ppo_cartpole_t0_spec.json                 # Trial spec (for reproduction)
├── ppo_cartpole_t0_trial_graph_*.png         # Trial training curves
├── ppo_cartpole_t0_trial_metrics_scalar.json # Trial metrics summary
│
├── graph/                  # Per-session graphs
│   └── ppo_cartpole_t0_s0_session_graph_*.png
│
├── info/                   # Session data
│   ├── ppo_cartpole_t0_s0_session_df.csv     # Session time series
│   └── experiment_df.csv                      # Search results (search mode)
│
├── log/                    # TensorBoard logs
│   └── events.out.tfevents.*
│
└── model/                  # PyTorch checkpoints
    ├── ppo_cartpole_t0_s0_ckpt-best_net_model.pt
    └── ppo_cartpole_t0_s0_net_model.pt
```

**Naming:** `{spec_name}_t{trial}_s{session}_{type}.{ext}`

## Common Workflows

### Train and Evaluate

```bash
# 1. Train
slm-lab run slm_lab/spec/benchmark/ppo/ppo_lunar.json ppo_lunar train

# 2. Check results
ls data/ppo_lunar_*/

# 3. Replay
slm-lab run slm_lab/spec/benchmark/ppo/ppo_lunar.json ppo_lunar enjoy@data/ppo_lunar_*/ppo_lunar_t0_spec.json
```

### Cloud Training Workflow

```bash
# 1. Source credentials
source .env

# 2. Launch
slm-lab run-remote --gpu slm_lab/spec/benchmark/ppo/ppo_hopper.json ppo_hopper train -n ppo-hopper

# 3. Monitor
dstack ps && dstack logs ppo-hopper

# 4. Pull when complete
slm-lab pull ppo_hopper
```

### Batch Benchmarking

```bash
# Multiple environments with template spec
source .env
slm-lab run-remote --gpu -s env=ALE/Qbert-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari train -n qbert
slm-lab run-remote --gpu -s env=ALE/MsPacman-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari_lam85 train -n mspacman
slm-lab run-remote --gpu -s env=ALE/Breakout-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari_lam70 train -n breakout

dstack ps  # Monitor all
```
