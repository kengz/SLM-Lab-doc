# Post-Hoc Analysis

## Regenerating Graphs and Metrics

After training completes, you may want to regenerate analysis with:
- Updated visualization styling
- New metrics added to SLM Lab
- Different graph parameters

The `retro_analysis` module recomputes all derived data without re-running training.

## Basic Usage

```bash
uv run python -c 'from slm_lab.experiment import retro_analysis; retro_analysis.retro_analyze("data/ppo_lunar_2024_01_15_123456")'
```

This regenerates:
- All graphs (PNG and HTML)
- Trial-level aggregated metrics
- Experiment summary (for search runs)

## What Gets Regenerated

| Artifact | Original Location | Effect |
|----------|-------------------|--------|
| Session graphs | `graph/*_session_graph_*.png` | Overwritten |
| Trial graphs | `graph/*_trial_graph_*.png` | Overwritten |
| Experiment graphs | `graph/*_experiment_graph.png` | Overwritten |
| Trial metrics | `info/*_trial_metrics.json` | Overwritten |
| Session data | `info/*_session_df.csv` | **Preserved** |
| Model checkpoints | `model/*.pt` | **Preserved** |
| Spec file | `*_spec.json` | **Preserved** |

{% hint style="success" %}
**Safe to run**: Retro analysis only overwrites derived data. Your raw session data, trained models, and spec files are never modified.
{% endhint %}

## Common Use Cases

### Update Graph Styling

SLM Lab updates Plotly styling periodically. Regenerate graphs to get the latest look:

```bash
# Pull latest code
git pull
uv sync

# Regenerate all experiments in data/
for dir in data/*/; do
    uv run python -c "from slm_lab.experiment import retro_analysis; retro_analysis.retro_analyze('$dir')"
done
```

### Recompute Metrics After Code Changes

If you modify the analysis module (e.g., add a new metric):

```bash
# After modifying slm_lab/experiment/analysis.py
uv run python -c 'from slm_lab.experiment import retro_analysis; retro_analysis.retro_analyze("data/ppo_lunar_2024_01_15_123456")'

# Check updated metrics
cat data/ppo_lunar_2024_01_15_123456/info/*_trial_metrics_scalar.json
```

### Generate Publication-Quality Graphs

For papers or presentations, you may want higher-resolution or different formats:

```python
from slm_lab.experiment import retro_analysis
from slm_lab.lib import viz

# Regenerate with custom settings
retro_analysis.retro_analyze('data/ppo_lunar_2024_01_15_123456')

# The HTML files support interactive exploration
# PNG files are ready for documents
```

## Batch Processing

Process multiple experiments:

```python
import os
from slm_lab.experiment import retro_analysis

data_dirs = [d for d in os.listdir('data') if os.path.isdir(f'data/{d}')]

for data_dir in data_dirs:
    print(f"Processing {data_dir}...")
    retro_analysis.retro_analyze(f'data/{data_dir}')
```

## Troubleshooting

### "Session data not found"

Ensure the data folder contains `*_session_df.csv` files. These are required for analysis.

### Graphs look wrong

Check that your SLM Lab version matches the data format. Very old experiments may need manual migration.
