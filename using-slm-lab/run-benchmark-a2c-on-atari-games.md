# Running Benchmarks 🏆

This tutorial shows how to run systematic benchmarks across multiple environments using variable substitution.

## Variable Substitution

Benchmarking requires running the same algorithm across multiple environments. SLM Lab makes this easy with the `-s` flag.

## Template Specs

Template specs use `${var}` placeholders for values that vary across runs. The PPO Atari spec at [slm\_lab/spec/benchmark/ppo/ppo\_atari.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_atari.json) uses `${env}` for the environment name:

{% code title="slm_lab/spec/benchmark/ppo/ppo_atari.json (excerpt)" %}
```javascript
{
  "ppo_atari": {
    "agent": {
      "name": "PPO",
      "algorithm": {
        "name": "PPO",
        "gamma": 0.99,
        "lam": 0.95,
        "time_horizon": 128,
        "minibatch_size": 256,
        "training_epoch": 4
      },
      "memory": {"name": "OnPolicyBatchReplay"},
      "net": {
        "type": "ConvNet",
        "shared": true,
        "gpu": "auto"
      }
    },
    "env": {
      "name": "${env}",
      "num_envs": 16,
      "max_frame": 1e7,
      "life_loss_info": true
    },
    "meta": {
      "max_session": 4,
      "max_trial": 1
    }
  }
}
```
{% endcode %}

{% hint style="info" %}
The `${env}` placeholder is replaced at runtime with the value passed via `-s env=...`.
{% endhint %}

## Running PPO Atari Benchmark

![MsPacman](https://user-images.githubusercontent.com/8209263/63994685-5cb30d00-caaa-11e9-8f35-78e29a7d60f5.gif)

Use the `-s` flag to substitute environment names:

```bash
# Single environment
slm-lab run -s env=ALE/MsPacman-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari_lam85 train

# Multiple environments (run separately)
slm-lab run -s env=ALE/Breakout-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari_lam70 train
slm-lab run -s env=ALE/Qbert-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari train
slm-lab run -s env=ALE/Seaquest-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari train
```

{% hint style="info" %}
Different games benefit from different lambda values. The `ppo_atari` spec works well for most games. Use `ppo_atari_lam85` for platformers (Qbert, Kangaroo) and `ppo_atari_lam70` for racing/physics games (Breakout, Enduro).
{% endhint %}

## MuJoCo Benchmark Example

The same pattern works for MuJoCo environments:

```bash
slm-lab run -s env=Hopper-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco train
slm-lab run -s env=HalfCheetah-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco train
slm-lab run -s env=Walker2d-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco train
```

## Cloud Benchmarking

For running benchmarks on cloud GPUs with automatic result upload:

```bash
source .env && slm-lab run-remote --gpu -s env=ALE/Breakout-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari_lam70 train -n ppo-breakout
```

See [Remote Training](remote-training.md) for setup.

## Benchmark Results

All SLM Lab benchmark specs are in [slm\_lab/spec/benchmark/](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/spec/benchmark). For results and methodology, see:

{% content-ref url="../benchmark-results/discrete-benchmark.md" %}
[discrete-benchmark.md](../benchmark-results/discrete-benchmark.md)
{% endcontent-ref %}

{% content-ref url="../benchmark-results/continuous-benchmark.md" %}
[continuous-benchmark.md](../benchmark-results/continuous-benchmark.md)
{% endcontent-ref %}

{% content-ref url="../benchmark-results/atari-benchmark.md" %}
[atari-benchmark.md](../benchmark-results/atari-benchmark.md)
{% endcontent-ref %}
