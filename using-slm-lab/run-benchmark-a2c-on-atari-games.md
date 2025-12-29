# Run Benchmark: A2C on Atari Games

## Spec Params for A2C on Atari Games

Benchmark results for an algorithm requires running it for a number of environments. This can easily be done in SLM Lab by parametrizing the spec file, which is similar to how it was done in [Experiment and Search Spec: PPO on Breakout](search-spec-ppo-on-breakout.md). Running benchmark in SLM Lab is easy by using variable substitution with the `-s` flag.

Let's run a benchmark for A2C on 4 Atari environments. We can look at an example spec from [slm\_lab/spec/benchmark/a2c/a2c\_gae\_atari.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/a2c/a2c_gae_atari.json)

{% code title="slm_lab/spec/benchmark/a2c/a2c_gae_atari.json" %}
```javascript
{
  "a2c_gae_atari": {
    "agent": {
      "name": "A2C",
      "algorithm": {
        "name": "ActorCritic",
        ...
      },
      ...
    },
    "env": {
      "name": "${env}",
      "frame_op": "concat",
      "frame_op_len": 4,
      "reward_scale": "sign",
      "num_envs": 16,
      "max_t": null,
      "max_frame": 1e7
    },
    ...
  }
}
```
{% endcode %}

{% hint style="info" %}
Spec param uses template string replacement to modify the spec. Replace the value of the environment name with `"${env}"`.
{% endhint %}

## Running A2C Atari Benchmark

To run the benchmark, use the `-s` flag to substitute the environment variable. The command to run it is:

```bash
slm-lab run -s env=ALE/Breakout-v5 slm_lab/spec/benchmark/a2c/a2c_gae_atari.json a2c_gae_atari train
```

To run multiple environments, simply run multiple commands with different environment values:

```bash
slm-lab run -s env=ALE/Breakout-v5 slm_lab/spec/benchmark/a2c/a2c_gae_atari.json a2c_gae_atari train
slm-lab run -s env=ALE/Pong-v5 slm_lab/spec/benchmark/a2c/a2c_gae_atari.json a2c_gae_atari train
slm-lab run -s env=ALE/Qbert-v5 slm_lab/spec/benchmark/a2c/a2c_gae_atari.json a2c_gae_atari train
slm-lab run -s env=ALE/Seaquest-v5 slm_lab/spec/benchmark/a2c/a2c_gae_atari.json a2c_gae_atari train
```

{% hint style="info" %}
All the SLM Lab benchmark results are run from files in [slm\_lab/spec/benchmark/](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/spec/benchmark).
{% endhint %}

Refer to the following pages for benchmark results in SLM Lab.

{% content-ref url="../benchmark-results/discrete-benchmark.md" %}
[discrete-benchmark.md](../benchmark-results/discrete-benchmark.md)
{% endcontent-ref %}

{% content-ref url="../benchmark-results/continuous-benchmark.md" %}
[continuous-benchmark.md](../benchmark-results/continuous-benchmark.md)
{% endcontent-ref %}

{% content-ref url="../benchmark-results/atari-benchmark.md" %}
[atari-benchmark.md](../benchmark-results/atari-benchmark.md)
{% endcontent-ref %}
