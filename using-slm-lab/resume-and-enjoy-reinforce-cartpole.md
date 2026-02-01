# Resume and Replay 🔄

After training, you can resume interrupted runs or replay trained models. This tutorial shows both.

## Resume Training

Training can be resumed using `train@{predir}` syntax, where `{predir}` is a previous run's data directory. Use `train@latest` to automatically resume from the most recent run.

Continuing from the [previous tutorial](train-ppo-cartpole.md):

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

## Replay Mode

Replay mode (`enjoy@`) loads a trained model and runs with rendering enabled, so you can watch your agent perform.

### Quick Replay (Latest Run)

The easiest way to replay your most recent training:

```bash
slm-lab run _ _ enjoy@latest
```

This automatically finds the most recent run folder and replays it.

### Replay Specific Run

For a specific run, use the spec file path:

```bash
slm-lab run _ _ enjoy@data/ppo_cartpole_2024_01_15_123456/ppo_cartpole_t0_spec.json
```

### Glob Patterns

Use `*` to match any characters (no need to type full timestamps):

```bash
# Match any ppo_cartpole run
slm-lab run _ _ enjoy@data/ppo_cartpole_*/ppo_cartpole_t0_spec.json
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

## Replaying Published Benchmarks

You can download and replay trained agents from [HuggingFace](https://huggingface.co/datasets/SLM-Lab/benchmark):

```bash
# List available experiments
slm-lab list

# Download trained model
slm-lab pull ppo_cartpole

# Replay
slm-lab run _ _ enjoy@data/ppo_cartpole_*/ppo_cartpole_t0_spec.json
```

{% hint style="success" %}
**All benchmark results are public.** Download any trained agent to see it in action or analyze its behavior. See [Public Benchmark Data](../benchmark-results/public-benchmark-data.md) for the full list.
{% endhint %}

Next, we'll dive into configuring the agent spec.
