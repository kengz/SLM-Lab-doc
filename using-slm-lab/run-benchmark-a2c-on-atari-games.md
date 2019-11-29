# Run Benchmark: A2C on Atari Games

## ✍ Spec Params for A2C on Atari Games

Benchmark results for an algorithm requires running it for a number of environments. This can easily be done in SLM Lab by parametrizing the spec file, which is similar to how it was done in [Experiment and Search Spec: PPO on Breakout](search-spec-ppo-on-breakout.md). Running benchmark in SLM Lab is easy by using **spec\_params** in the spec file, which uses [the same underlying function as experiment search](https://github.com/kengz/SLM-Lab/blob/master/run_lab.py#L59).

Let's run a benchmark for A2C on 4 Atari environments. We can look at an example spec from [slm\_lab/spec/benchmark/a2c/a2c\_gae\_atari.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/a2c/a2c_gae_atari.json)

{% code title="slm\_lab/spec/benchmark/a2c/a2c\_gae\_atari.json" %}
```javascript
{
  "a2c_gae_atari": {
    "agent": [{
      "name": "A2C",
      "algorithm": {
        "name": "ActorCritic",
        ...
      },
      ...
      }
    }],
    "env": [{
      "name": "${env}",
      "frame_op": "concat",
      "frame_op_len": 4,
      "reward_scale": "sign",
      "num_envs": 16,
      "max_t": null,
      "max_frame": 1e7
    }],
    ...
    "spec_params": {
      "env": [
        "BreakoutNoFrameskip-v4", "PongNoFrameskip-v4", "QbertNoFrameskip-v4", "SeaquestNoFrameskip-v4"
      ]
    }
  }
}
```
{% endcode %}

{% hint style="info" %}
Spec param uses template string replacement to modify the spec and append to the spec name. Replace the value of the environment name with `"${env}"`.
{% endhint %}

## 🚀 Running A2C Atari Benchmark

This benchmark will run 4 trials in total. The command to run it is familiar:

```bash
python run_lab.py slm_lab/spec/benchmark/a2c/a2c_gae_atari.json a2c_gae_atari train
```

The spec name will be replaced with each value of the spec param, so the resulting trials will be named "a2c\_gae\_atari\_BreakoutNoFrameskip-v4", "a2c\_gae\_atari\_PongNoFrameskip-v4", "a2c\_gae\_atari\_QbertNoFrameskip-v4", and "a2c\_gae\_atari\_SeaquestNoFrameskip-v4".

{% hint style="info" %}
All the SLM Lab benchmark results are run from files in [slm\_lab/spec/benchmark/](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/spec/benchmark).
{% endhint %}

Refer to the following pages for benchmark results in SLM Lab.

{% page-ref page="../benchmark-results/discrete-benchmark.md" %}

{% page-ref page="../benchmark-results/continuous-benchmark.md" %}

{% page-ref page="../benchmark-results/atari-benchmark.md" %}

