# Resume and Enjoy

## Train@ (Resume) Mode

Training can be resumed using `train@{predir}` syntax, where `{predir}` is a previous run's data directory. Use `train@latest` to automatically resume from the most recent run.

Continuing from the [previous tutorial](train-and-enjoy-dqn-cartpole.md):

```bash
# Start a run
slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole train
# Terminate early with Ctrl+C

# Resume from latest
slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole train@latest

# Or specify a specific run folder
slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole train@data/ppo_cartpole_2024_01_15_123456
```

### Extending Training

You can also extend a completed run. For example, if you ran 100k frames and want to continue to 200k:

1. Edit the spec's `max_frame` to 200000
2. Resume with `train@latest`

The run picks up exactly where it left off.

## Enjoy Mode

Enjoy mode replays a trained model. It loads the best checkpoint and runs with rendering enabled.

```bash
slm-lab run _ _ enjoy@data/ppo_cartpole_2024_01_15_123456/ppo_cartpole_t0_spec.json
```

{% hint style="info" %}
The `_ _` are placeholders. In enjoy mode, all settings come from the saved spec file, so the spec file and spec name arguments aren't needed.
{% endhint %}

This creates a new Session that:
1. Loads the saved trial spec
2. Finds the best session (by `total_reward_ma`)
3. Loads the **best** model checkpoint (`_ckpt-best` files)
4. Runs with rendering enabled

The trained PPO agent should immediately balance the CartPole, with `total_reward_ma` starting near 500:

![](<../.gitbook/assets/cartpole enjoy.png>)

Next, we'll dive into configuring the agent spec.
