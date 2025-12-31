# Lab Command

## The Lab Command

The CLI uses [Typer](https://typer.tiangolo.com/). Use `--help` on any command for details:

```bash
slm-lab --help           # List all commands
slm-lab run --help       # Options for run command
```

## Running Experiments

The basic command for running experiments:

```bash
slm-lab run {spec file} {spec name} {lab mode}
```

{% hint style="success" %}
Spec files are located in `slm_lab/spec/`. Each file can contain multiple specs, identified by name.
{% endhint %}

### Lab Modes

* **dev**: Development mode with verbose logging, TensorBoard, and validation checks. Slower but useful for debugging.
* **train**: Full training run. Disables dev tools for maximum speed.
* **search**: Hyperparameter search using Ray Tune.
* **train@{predir}**: Resume training from a previous run (e.g., `train@latest` or `train@data/ppo_cartpole_2024_01_15_123456`).
* **enjoy@{trial\_spec\_file}**: Replay a trained model (auto-selects best session).

### Examples

```bash
# Default: PPO on CartPole in dev mode
slm-lab run

# With environment rendering
slm-lab run --render

# Train a specific spec
slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole train

# Dev mode with rendering
slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole dev

# Hyperparameter search
slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole search

# Resume training from latest run
slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole train@latest

# Enjoy a trained model (uses _ _ placeholders since spec is in the path)
slm-lab run _ _ enjoy@data/ppo_cartpole_2024_01_15_123456/ppo_cartpole_t0_spec.json
```

### Variable Substitution

Template specs use `${var}` placeholders. Substitute values with `-s`:

```bash
# Run PPO on different Atari games (lam95 is the default variant)
slm-lab run -s env=ALE/Breakout-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari_lam95 train
slm-lab run -s env=ALE/Pong-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari_lam95 train

# Run PPO on different MuJoCo environments
slm-lab run -s env=Hopper-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco train
slm-lab run -s env=HalfCheetah-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco train
```

### Common Options

```bash
-s, --set KEY=VALUE   # Set spec variables (can be used multiple times)
--render              # Enable environment rendering
--log-level DEBUG     # Set log level (DEBUG, INFO, WARNING, ERROR)
--cuda-offset 4       # Offset GPU device selection (for multi-GPU machines)
--upload-hf           # Upload to HuggingFace after training completes
--keep 3              # Number of top trials to keep after search (default: 3)
--profile             # Enable performance profiling
--log-extra           # Enable extra metrics logging (strength, stability, efficiency)
--no-optimize-perf    # Disable auto CPU/GPU optimization
--stop-ray            # Stop Ray processes (useful for cleanup)
```

## Cloud Training (dstack + HuggingFace)

SLM Lab integrates with [dstack](https://dstack.ai/) for cloud GPU training and HuggingFace for experiment storage.

### Remote Execution

```bash
# Launch on cloud GPU
slm-lab run-remote --gpu slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari_lam95 train -n my-experiment

# Launch search on cloud
slm-lab run-remote --gpu slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari_lam95 search -n my-search
```

### Experiment Management

```bash
# List experiments on HuggingFace
slm-lab list

# Download results locally
slm-lab pull ppo_atari_lam95

# Upload local experiment
slm-lab push data/ppo_atari_lam95_2024_01_15_123456
```

{% hint style="info" %}
Remote execution requires dstack configuration. See the [Remote Training](remote-training.md) guide for setup.
{% endhint %}

## The Spec File

The **spec file** contains all hyperparameters for a run. Each file can contain multiple named specs:

```javascript
{
  "spec_name": {
    "agent": {
      "name": "PPO",
      "algorithm": {...},
      "memory": {...},
      "net": {...}
    },
    "env": {
      "name": "CartPole-v1",
      "num_envs": 4,
      "max_frame": 100000
    },
    "meta": {
      "max_session": 4,
      "max_trial": 1
    },
    "search": {...}
  }
}
```

Key sections:

* **agent**: Algorithm, memory, and network configuration
* **env**: Environment name and settings
* **meta**: Session/trial counts, logging frequency
* **search**: Hyperparameter search space (optional)

See the tutorials in this section for detailed spec configuration.
