# Train: PPO on CartPole

Run a full training with saved results.

## Train vs Dev Mode

In Quick Start, we used `dev` mode for quick verification. Now we use `train` mode:

| Mode | Sessions | Rendering | Saves Results |
|------|----------|-----------|---------------|
| `dev` | 1 | optional | no |
| `train` | 4 (default) | disabled | yes |

Train mode runs multiple sessions with different random seeds for statistical reliability, and disables rendering for faster training.

## Run Training

```bash
slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole train
```

This runs a Trial with 4 Sessions. Training completes in about 5-10 minutes. Watch `total_reward_ma` climb toward 500 (solved).

**Training curves** (from benchmark run):

![PPO CartPole Training Curve](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/data/ppo_cartpole_2026_01_30_221924/ppo_cartpole_t0_trial_graph_mean_returns_vs_frames.png)

**Moving average** smooths out episode-to-episode noise to show the learning trend:

![PPO CartPole Moving Average](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/data/ppo_cartpole_2026_01_30_221924/ppo_cartpole_t0_trial_graph_mean_returns_ma_vs_frames.png)

## Output Folder

Results are saved to a timestamped folder:

```
data/ppo_cartpole_2026_01_30_221924/
├── ppo_cartpole_spec.json                            # Original spec
├── ppo_cartpole_t0_spec.json                         # Trial spec (for reproduction)
├── ppo_cartpole_t0_trial_graph_*.png                 # Trial training curves
├── ppo_cartpole_t0_trial_metrics_scalar.json         # Trial scalar metrics
├── graph/                                            # Per-session graphs
├── info/                                             # Metrics CSV files
├── log/                                              # TensorBoard events
└── model/                                            # PyTorch checkpoints
    ├── ppo_cartpole_t0_s0_net_model.pt               # Session 0 final model
    ├── ppo_cartpole_t0_s0_ckpt-best_net_model.pt     # Session 0 best model
    └── ...                                           # (same for s1, s2, s3)
```

**Best vs final checkpoints:** "best" is the highest evaluation score during training; "final" is at the end. Usually similar, but "best" helps if performance dropped near the end.

See [Data Locations](../analyzing-results/analytics.md) for full details.

### Auto-Upload to HuggingFace

If you set up environment variables, results auto-upload to HuggingFace after training—useful for [remote training](remote-training.md):

```bash
cp .env.example .env
# Edit .env with your HF_TOKEN from https://huggingface.co/settings/tokens
```

## Next Steps

Learn how SLM Lab organizes experiments in [Core Concepts](lab-organization.md).
