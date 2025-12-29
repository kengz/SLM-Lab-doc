# Lab Command

## The Lab Command

The CLI uses [Typer](https://typer.tiangolo.com/). Use `--help` on any command for details:

```bash
slm-lab --help           # List all commands
slm-lab run --help       # Options for run command
```

In SLM Lab, everything is run with the lab command with the following form:

```bash
slm-lab run {spec file} {spec name} {lab mode}
```

{% hint style="success" %}
_This command runs any algorithm/environment specified in a spec file in SLM Lab. Spec files are located in the `slm_lab/spec/` folder._
{% endhint %}

### The Spec File

The **spec file** contains the **spec** - a set of fully exposed hyperparameters that configure a run, including the agent, environment, and hyperparameter search. The **spec name** refers to a specific spec in the spec file.

All the spec files are defined in the **slm\_lab/spec/** folder. The spec file has the following format:

```javascript
{
  "{spec name}": {
    "agent": {...},
    "env": {...},
    "meta": {...},
    "search": {...}
  },
  "{spec name 2}": {
    ...
  },
  ...
}
```

### The Lab Modes

We will take a deep dive into the spec file in the coming sections, since it is crucial to running SLM Lab. Next, the **lab mode** specifies one of the modes used to run the lab:

* **dev**: for development with verbose logging, environment rendering, and helpful checks like gradient updates. This is slower but useful for development.
* **train**: for training an agent to completion. This disables the development helper tools and thus runs the fastest.
* **train@{predir}**: for resuming training, e.g. `train@latest` will use the latest run for a spec, and `train@data/reinforce_cartpole_2020_04_13_232521` will use the specified run.
* **enjoy@{session\_spec\_file}**: for replaying a trained model from a trial-session; `session_spec_file` specifies the spec file from a session, e.g. `enjoy@data/reinforce_cartpole_2020_04_13_232521/reinforce_cartpole_t0_s0_spec.json`.
* **search**: for running an experiment / hyperparameter search.

### Examples

Run the default demo (PPO on CartPole):

```bash
slm-lab run
```

Run with environment rendering:

```bash
slm-lab run --render
```

Run a specific spec in train mode:

```bash
slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole train
```

Run in dev mode to see rendering and verbose logging:

```bash
slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole dev
```

### Variable Substitution

For template specs with `${var}` placeholders, use the `-s` flag:

```bash
slm-lab run -s env=ALE/Breakout-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari train
slm-lab run -s env=Hopper-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco train
```

In the coming sections we will learn to use SLM Lab with more hands-on tutorials.
