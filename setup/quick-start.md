---
description: To test the installation.
---

# Quick Start

## PPO on CartPole

This quick demo tests that the installation works. We'll run PPO on CartPole in dev mode.

```bash
slm-lab run --render
```

You should see:
1. A rendering window showing CartPole being balanced
2. Logs with `total_reward_ma` increasing over time (target: 400+)

![CartPole demo](../.gitbook/assets/dqn_cartpole_demo.png)

After a few minutes, the CartPole should balance for longer periods. Press `Ctrl+C` to stop.

{% hint style="info" %}
Dev mode is for debugging and includes rendering. For full training, use `train` mode (see [Lab Command](../using-slm-lab/slm-lab-command.md)).
{% endhint %}

{% hint style="success" %}
If this works, SLM Lab is ready. Continue to [Lab Command](../using-slm-lab/slm-lab-command.md) to learn the CLI.
{% endhint %}

{% hint style="info" %}
If you encounter an issue, consult the [**Help**](../resources/help.md) page.
{% endhint %}

## What's Next

* [Lab Command](../using-slm-lab/slm-lab-command.md) - Learn CLI options and modes
* [Train: PPO CartPole](../using-slm-lab/train-and-enjoy-dqn-cartpole.md) - Your first full training run
* [Lab Organization](../using-slm-lab/lab-organization.md) - Understand Sessions, Trials, Experiments
