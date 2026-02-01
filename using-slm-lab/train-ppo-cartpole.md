# Train: PPO on CartPole 🎓

Now let's run a full training and save the results. This is your first "real" training run.

## Running Full Training

We'll train PPO on CartPole—the same setup from Quick Start, but this time saving everything for later analysis.

The spec file is at [slm\_lab/spec/benchmark/ppo/ppo\_cartpole.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_cartpole.json). To run a full training:

```bash
slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole train
```

{% hint style="info" %}
This is the same as running `slm-lab run` without arguments—PPO CartPole is the default.
{% endhint %}

This runs a `Trial` with 4 `Sessions` using different random seeds. The training completes in about 5-10 minutes. Watch the terminal for the `total_reward_ma` metric (100-episode moving average) climbing toward 500.

{% hint style="success" %}
PPO reliably solves CartPole when `total_reward_ma` reaches 450-500 (the maximum score).
{% endhint %}

**Training curves** (from benchmark run):

![PPO CartPole Training Curve](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/data/ppo_cartpole_2026_01_30_221924/ppo_cartpole_t0_trial_graph_mean_returns_vs_frames.png)

![PPO CartPole Moving Average](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/data/ppo_cartpole_2026_01_30_221924/ppo_cartpole_t0_trial_graph_mean_returns_ma_vs_frames.png)

When complete, all metrics, graphs, and data are saved to a timestamped folder like `data/ppo_cartpole_2026_01_30_221924/`. SLM Lab saves two model checkpoints:

* **best**: The model with highest evaluation score during training
* **final**: The model at the end of training

Usually these are similar, but "best" is useful if performance dropped near the end.

## What's in the Output Folder? 📁

```
data/ppo_cartpole_2026_01_30_221924/
├── ppo_cartpole_t0_spec.json              # Saved spec (for reproduction)
├── ppo_cartpole_t0_trial_graph_*.png      # Training curves
├── graph/                                  # Per-session graphs
├── info/                                   # Metrics CSV files
├── log/                                    # TensorBoard events
└── model/                                  # PyTorch checkpoints
    ├── ppo_cartpole_t0_s0_ckpt-best.pt
    └── ppo_cartpole_t0_s0_ckpt-last.pt
```

See [Data Locations](../analyzing-results/analytics.md) for full details.

## Next Steps

Now that you have trained models, let's [resume and replay](resume-and-enjoy-reinforce-cartpole.md) them.
