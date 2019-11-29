# Lab Command

## 🚀 The Lab Command

Before running anything in SLM Lab, be sure to activate the Conda environment:

```bash
conda activate lab
```

In SLM Lab, everything is run with the lab command with the following form:

```bash
python run_lab.py {spec file} {spec name} {lab mode}
```

### The Spec File

The **spec file** contains the **spec** – a set of fully exposed hyperparameters that configure a run, including the agent, environment, and hyperparameter search. The **spec name** refers to a specific spec in the spec file.

All the spec files are defined in the **slm\_lab/spec/** folder. The spec file has the following format:

```javascript
{
  "{spec name}": {
    "agent": [{...}],
    "env": [{...}],
    ...
    "search": [{...}]
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
* **enjoy@{prename}**: for replaying a trained model from a trial-session; `prename` specifies the trial and session, e.g. `enjoy@a2c_gae_pong_t0_s0`.
* **search**: for running an experiment / hyperparameter search.

In [Quick Start](../setup/quick-start.md), we used the lab command to read the demo spec file at `slm_lab/spec/demo.json`, use the `dqn_cartpole` spec in it, and run the spec in **dev** mode. To rerun the demo in train mode, we can simply change the lab mode to **train** to get the following:

```bash
python run_lab.py slm_lab/spec/demo.json dqn_cartpole train
```

In the coming sections we will learn to use SLM Lab with more hands-on tutorials.

