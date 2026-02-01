# Train: PPO CartPole 🎓

## Train Mode

This tutorial shows how to train an agent in SLM Lab and use the saved model for replay in enjoy mode.

We'll use PPO (Proximal Policy Optimization) on CartPole—the same algorithm and environment from the Quick Start demo. PPO is a widely-used RL algorithm that works well across many environments.

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

**Training curve** (from benchmark run, 100-checkpoint moving average):

![PPO CartPole Training Curve](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/data/ppo_cartpole_2026_01_30_221924/graph/ppo_cartpole_t0_trial_graph_mean_returns_ma_vs_frames.png)

When complete, all metrics, graphs, and data are saved to a timestamped folder like `data/ppo_cartpole_2024_01_15_123456/`. SLM Lab saves two model checkpoints:

* **best**: The model with highest evaluation score during training
* **final**: The model at the end of training

Usually these are similar, but "best" is useful if performance dropped near the end.

Next, we'll look at how to resume training and replay a trained model.
