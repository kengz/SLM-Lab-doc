# Resume and Replay

Resume interrupted training or replay trained models.

## Resume Training

Use `train@{predir}` to resume from a previous run:

```bash
# Resume from latest run of this spec
slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole train@latest

# Resume from specific folder
slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole train@data/ppo_cartpole_2026_01_30_221924
```

`train@latest` resolves to the most recent `data/{spec_name}_*/` folder.

### Extending Training

To continue a completed run (e.g., 100k → 200k frames):

1. Edit the spec's `max_frame`
2. Resume with `train@latest`

## Replay Mode

Use `enjoy@{spec_file}` to replay a trained model with rendering:

```bash
# Replay from saved spec file (first two args ignored in enjoy mode)
slm-lab run spec.json spec_name enjoy@data/ppo_cartpole_2026_01_30_221924/ppo_cartpole_t0_spec.json

# Glob pattern (match any timestamp)
slm-lab run spec.json spec_name enjoy@data/ppo_cartpole_*/ppo_cartpole_t0_spec.json
```

In enjoy mode, `spec.json` and `spec_name` are ignored—everything loads from the saved spec file.

Enjoy mode finds the best session (by `total_reward_ma`) and loads its `ckpt-best` model checkpoint.

## Replaying Published Benchmarks

Download and replay trained agents from [HuggingFace](https://huggingface.co/datasets/SLM-Lab/benchmark):

```bash
slm-lab list              # List available experiments
slm-lab pull ppo_cartpole # Download trained model
slm-lab run spec.json spec_name enjoy@data/ppo_cartpole_*/ppo_cartpole_t0_spec.json
```

See [Public Benchmark Data](../benchmark-results/public-benchmark-data.md) for the full list.
