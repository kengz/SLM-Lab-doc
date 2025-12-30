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

### How Resume Works

Resume mode restores training in a _past-future-consistent_ manner.

Suppose you ran 100k frames but want to extend to 200k. You can edit the spec's `max_frame` and resume—the run picks up where it left off as if it was always configured for 200k frames.

The lab restores three key objects:
* **algorithm weights**: Network parameters via `algorithm.load()`
* **training metrics**: The `train_df` tracking object
* **environment clock**: Timestep tracking via `env.clock`

Since everything runs according to `env.clock`, these are sufficient to resume correctly.

{% hint style="info" %}
For off-policy algorithms, replay memory is not restored (it would be gigabytes of data). The replay buffer refills from the resume point, and training resumes once the buffer reaches the minimum size threshold.
{% endhint %}

## Enjoy Mode

Enjoy mode runs a trained model using `enjoy@{session_spec_file}`. The session spec was saved automatically during training, and the lab loads the best checkpoint.

```bash
slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole enjoy@data/ppo_cartpole_2024_01_15_123456/ppo_cartpole_t0_s0_spec.json
```

This creates a new Session that:
1. Loads the saved session spec
2. Loads the **best** model checkpoint (`_ckpt-best` files)
3. Runs with rendering enabled

The trained PPO agent should immediately balance the CartPole, with `total_reward_ma` starting near 500:

![](<../.gitbook/assets/cartpole enjoy.png>)

Next, we'll dive into configuring the agent spec.
