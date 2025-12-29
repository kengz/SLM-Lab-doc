---
description: To test the installation.
---

# Quick Start

## PPO on CartPole

This quick demo will test that the installation works. We will run PPO on the CartPole environment. For now, don't worry about the details of the command, as we will walk through them in a [later section](../using-slm-lab/slm-lab-command.md).

```bash
slm-lab run --render
```

This will run a session that trains a PPO agent on the CartPole-v1 environment. The `--render` flag enables environment rendering, so you should see a window showing the CartPole being balanced.

![](../.gitbook/assets/dqn_cartpole_demo.png)

If you let the training session run for a few minutes, you should see the CartPole getting balanced for a longer period of time. Correspondingly, the `total_reward_ma` in the logs should increase.

{% hint style="success" %}
If this quick start works, then SLM Lab is ready for use.
{% endhint %}

{% hint style="info" %}
If you encounter an issue, consult the [**Help**](../resources/help.md) page.
{% endhint %}
