# Data Locations 📂

Understanding where SLM Lab saves data helps you analyze results, reproduce experiments, and share findings.

## Output Directory Structure

After training, results are saved to `data/{spec_name}_{timestamp}/`:

```
data/ppo_cartpole_2026_01_30_221924/
├── ppo_cartpole_spec.json                              # Original spec
├── ppo_cartpole_t0_spec.json                           # Trial spec (for reproduction)
├── ppo_cartpole_t0_trial_graph_mean_returns_vs_frames.png    # Trial training curve
├── ppo_cartpole_t0_trial_graph_mean_returns_ma_vs_frames.png # Trial moving average
├── ppo_cartpole_t0_trial_metrics_scalar.json           # Trial metrics
│
├── graph/                  # Per-session graphs
│   └── ppo_cartpole_t0_s0_session_graph_*.png
│
├── info/                   # Session data and experiment results
│   ├── ppo_cartpole_t0_s0_session_df.csv              # Session time series
│   └── experiment_df.csv                               # Search results (if search mode)
│
├── log/                    # TensorBoard event files
│   └── events.out.tfevents.*
│
└── model/                  # PyTorch checkpoints
    ├── ppo_cartpole_t0_s0_ckpt-best_net_model.pt      # Best model (session 0)
    └── ppo_cartpole_t0_s0_net_model.pt                # Final model (session 0)
```

**Naming convention:** `{spec_name}_t{trial}_s{session}_{type}.{ext}`
- `t0` = Trial index 0
- `s0` = Session index 0

## Key Files Explained

### Session DataFrame (`*_session_df.csv`)

Time series of training metrics, one row per checkpoint:

| Column | Description |
|--------|-------------|
| `frame` | Total frames processed |
| `total_reward` | Episode reward at this checkpoint |
| `total_reward_ma` | 100-checkpoint moving average of reward |
| `loss` | Training loss |
| `fps` | Frames per second |
| `wall_t` | Wall-clock time (seconds) |
| `epi` | Episode count |
| `opt_step` | Optimization steps |
| `lr` | Current learning rate |

**Example usage:**

```python
import pandas as pd

df = pd.read_csv('data/ppo_cartpole_2026_01_30_221924/info/ppo_cartpole_t0_s0_session_df.csv')
print(f"Final reward MA: {df['total_reward_ma'].iloc[-1]:.2f}")
print(f"Training time: {df['wall_t'].iloc[-1] / 60:.1f} minutes")
```

### Trial Metrics (`*_trial_metrics_scalar.json`)

Summary statistics aggregated across sessions:

```json
{
  "strength": 245.3,
  "efficiency": 0.82,
  "stability": 0.95,
  "consistency": 0.91,
  "final_return_ma": 267.5
}
```

See [Performance Metrics](performance-metrics.md) for detailed definitions.

### Experiment DataFrame (`experiment_df.csv`)

For search mode, contains all trial results sorted by performance:

| Column | Description |
|--------|-------------|
| `trial` | Trial index |
| `{search_var}` | Hyperparameter values tried |
| `total_reward_ma` | Final performance |
| `strength`, `efficiency`, etc. | Performance metrics |

**Example usage:**

```python
import pandas as pd

df = pd.read_csv('data/ppo_breakout_search_2026_01_30/info/experiment_df.csv')
best_trial = df.iloc[0]  # Sorted best-first
print(f"Best trial: {best_trial['trial']}")
print(f"Best lam: {best_trial['agent.algorithm.lam']}")
print(f"Best reward: {best_trial['total_reward_ma']:.2f}")
```

### Model Checkpoints

Two checkpoints are saved per session:

- **`*_ckpt-best_net_model.pt`**: Model with highest `total_reward_ma` during training
- **`*_net_model.pt`**: Model at the end of training (no `ckpt-` prefix)

{% hint style="info" %}
Usually these are similar, but "best" is useful if performance dropped near the end of training (policy collapse).
{% endhint %}

**Loading a checkpoint:**

```python
from slm_lab.agent.net import net_util

# Load weights into an agent
net_util.load(agent.algorithm, 'data/ppo_cartpole_*/model/ppo_cartpole_t0_s0_ckpt-best_net_model.pt')
```

### Spec File (`*_spec.json`)

The complete spec used for this run, including:

- All hyperparameters
- Git SHA of the code version
- Random seeds

This file is all you need to reproduce the experiment.

## Quick Reference

| What you want | Where to find it |
|---------------|------------------|
| Final reward | `total_reward_ma` in last row of `info/*_session_df.csv` |
| Training curves | `*_trial_graph_mean_returns_ma_vs_frames.png` (root folder) |
| Best hyperparameters | First row of `info/experiment_df.csv` |
| Model for inference | `model/*_ckpt-best_net_model.pt` |
| Reproduce this run | `slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole enjoy@data/ppo_cartpole_*/ppo_cartpole_t0_spec.json` |
| TensorBoard data | `log/` folder |

## Working with Results

### Using SLM Lab's Analysis Module

SLM Lab uses Plotly for visualization. You can regenerate graphs using the retro_analysis module:

```bash
uv run python -c 'from slm_lab.experiment import retro_analysis; retro_analysis.retro_analyze("data/ppo_cartpole_2026_01_30_221924")'
```

See [Post-Hoc Analysis](../using-slm-lab/post-hoc-analysis.md) for more details.

### Viewing Graphs

The generated PNG graphs can be opened directly. For interactive viewing, SLM Lab also generates HTML files with Plotly's interactive features (zoom, hover data).

### Export for Publication

SLM Lab automatically creates a zip archive of the data folder. For sharing results:

1. Upload the zip to cloud storage or HuggingFace
2. Share the spec file for reproducibility
3. Reference the git SHA for exact code version

See [Public Benchmark Data](../benchmark-results/public-benchmark-data.md) for examples.
