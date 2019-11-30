# Meta Spec: High Level Specifications

## 📂The Meta Spec

In this tutorial we look at how to adjust the checkpointing frequency.

The **meta spec** is used to specify higher level configuration that don't fit within specific trial or session, such as how the experiments and trials should behave. It is specified using the **meta** key in a spec file with the following format:

```javascript
{
  "{spec_name}": {
    "agent": [{...}],
    "env": [{...}],
    "meta": {
      // Parameter for Trial.run(), to enable network sharing in memory among Sessions.
      // - false: disable Hogwild!
      // - "shared": enable Hogwild!, share parameters all the time
      // - "synced": enabled Hogwild!, sync parameters only after training step
      "distributed": str|bool,
      
      // Whether to use rigorous eval.
      // Note that this will be slower since it spawns vector environments separately to run evaluations, but it is more rigorous.
      // This is useful when the agent and environment respond differently when in LAB_MODE=eval
      // However, for all SLM-Lab-supported environments, this is optional since it can infer the total eval-rewards from training environments (see env.wrapper.TrackReward)
      // - int: If > 0, spawn a number of vector environments to run eval separately
      // - int|null: If 0 or null, infer eval scores from training checkpoints
      // - defaults to null for performance
      "rigorous_eval": int|null,

      // Frequency to checkpoint on evaluation environment: logging and saving.
      // This uses env.clock, which counts the total timesteps across vector environments by summing them up.
      "eval_frequency": int,

      // Frequency to checkpoint on training environment: logging and saving.
      // This uses env.clock, which counts the total timesteps across vector environments by summing them up.
      "log_frequency": int,

      // The maximum number of Sessions to spawn per Trial, indexed 0 to (max_session - 1)
      "max_session": int,

      // The maximum number of Trials to spawn per Experiment, indexed 0 to (max_session - 1)
      "max_trial": int,

      // The number of CPUs to allocate for Ray.tune per Trial.
      // If null, the trial is given {max_session} CPUs
      "num_cpus": int|null,

      // The number of GPUs to allocate for Ray.tune per Trial
      // If null, the trial is given {max_session} GPUs
      "num_gpus": int|null,
    }
  }
}
```

We have already encountered some of the meta spec hyperparameters. Hogwild can be enabled using "distributed". We can also adjust the checkpoint mode and frequency. The max session and trials \(when running an experiment\) are specified here. Finally, although an experiment will automatically assign resources for each trial, advanced users can also configure the resource allocation for Ray.tune through **"num\_cpus"** and **"num\_gpus"**.

## ✍ Meta Spec for Fast Checkpointing in Atari

Atari games usually have multiple lives per episode. During training, we split it up and treat each life as an episode to encourage the agent to appreciate all its lives. This split also implies that total rewards tracked during training need to be summed up over the lives to yield the true episodic rewards for evaluation.

One method for evaluation is to spawn a new vector environments, say using "rigorous\_eval": 8 and run them in evaluation mode so they do not split up an episode. However, this can be quite slow and expensive to run.

Thankfully, SLM Lab has a preprocessor which tracks the true episodic rewards for Atari environments at [env.wrapper.TrackReward](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/env/wrapper.py#L323). This means we can use the true episodic rewards from the training environment and run a trial much faster. To do so, disable the rigorous eval, and set the evaluation frequency to a desired value, as shown below.

```javascript
{
  "agent_pong": {
    "agent": [{...}],
    "env": [{
      "name": "PongNoFrameskip-v4",
      "frame_op": "concat",
      "frame_op_len": 4,
      "reward_scale": "sign",
      "num_envs": 16,
      "max_t": null,
      "max_frame": 1e7
    }],
    ...
    "meta": {
      "distributed": false,
      "rigorous_eval": 0,
      "eval_frequency": 10000,
      "log_frequency": 5000,
      "max_session": 4,
      "max_trial": 1
    }
  }
}
```

Note that **"eval\_frequency"** \(evaluation checkpointing\) and **"log\_frequency"** \(training checkpointing\) are independent. Since evaluation needs multiple lives to tally the episodic rewards, it is usually set higher than "log\_frequency".

