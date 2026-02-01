---
description: Test your installation with a quick demo.
---

# Quick Start

## Command Format

Check the run command options:

```bash
slm-lab run --help
```

```
Usage: slm-lab run [OPTIONS] [SPEC_FILE] [SPEC_NAME] [MODE]

Arguments:
  spec_file   JSON spec file path [default: slm_lab/spec/benchmark/ppo/ppo_cartpole.json]
  spec_name   Spec name within the file [default: ppo_cartpole]
  mode        Execution mode: dev|train|search|enjoy [default: dev]

Options:
  --set, -s   Set spec variables: KEY=VALUE (can be used multiple times)
  --render    Enable environment rendering
```

The full command format is:

```bash
slm-lab run spec.json spec_name mode
```

So `slm-lab run` is equivalent to:

```bash
slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole dev
```

## Run the Demo

```bash
slm-lab run --render
```

This runs PPO on CartPole with visualization (single session, slower due to rendering). **CartPole** is a classic RL benchmark: balance a pole on a cart by moving left or right. The agent receives +1 reward per timestep the pole stays upright (max 500 per episode).

### What to Expect

**Terminal output:**
```
[2026-01-15 12:00:00] INFO: Starting ppo_cartpole trial t0
[2026-01-15 12:00:05] INFO: frame: 1000 | total_reward: 23.5 | total_reward_ma: 23.5 | loss: 0.012
[2026-01-15 12:00:10] INFO: frame: 2000 | total_reward: 45.2 | total_reward_ma: 34.3 | loss: 0.008
...
[2026-01-15 12:02:30] INFO: frame: 50000 | total_reward: 500.0 | total_reward_ma: 487.2 | loss: 0.002
```

Key metrics:
- **frame**: Total environment steps processed
- **total_reward**: Episode reward at this checkpoint
- **total_reward_ma**: Moving average over 100 checkpoints (the primary success metric)
- **loss**: Training loss (should decrease over time)

**Rendering window:**

![CartPole demo](../.gitbook/assets/dqn_cartpole_demo.png)

Early in training, the pole falls quickly. As `total_reward_ma` climbs toward 400-500, you'll see the agent balance for longer periods.

### Success Criteria

Random actions score ~20-30. PPO solves CartPole (450+ reward) within 50,000-100,000 frames.

**Training curves** (from benchmark run):

![PPO CartPole Training Curve](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/data/ppo_cartpole_2026_01_30_221924/ppo_cartpole_t0_trial_graph_mean_returns_vs_frames.png)

![PPO CartPole Moving Average](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/data/ppo_cartpole_2026_01_30_221924/ppo_cartpole_t0_trial_graph_mean_returns_ma_vs_frames.png)

### Stopping the Demo

Press `Ctrl+C` to stop. In dev mode, partial results are not saved.

{% hint style="success" %}
If you see rewards climbing, SLM Lab is working correctly. Continue to [Train: PPO on CartPole](../using-slm-lab/train-ppo-cartpole.md) for a full training run.
{% endhint %}

{% hint style="warning" %}
**Troubleshooting:** If you encounter errors, see [Help](../resources/help.md) for common issues and solutions.
{% endhint %}

## Next Steps

1. **[Train: PPO on CartPole](../using-slm-lab/train-ppo-cartpole.md)** - Full training with saved results
2. **[Resume and Replay](../using-slm-lab/resume-and-enjoy-reinforce-cartpole.md)** - Resume training and replay trained models
3. **[Understanding Experiments](../using-slm-lab/lab-organization.md)** - Sessions, Trials, and Experiments
