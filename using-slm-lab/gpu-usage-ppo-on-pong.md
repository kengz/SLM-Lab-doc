# GPU Training 🎮

This tutorial shows how to train on Atari games using GPU acceleration.

![Qbert](https://user-images.githubusercontent.com/8209263/63994672-54f36880-caaa-11e9-9757-7780725b53af.gif)

**[Q*bert](https://ale.farama.org/environments/qbert/)** is a classic Atari puzzle game—hop around a pyramid changing tile colors while avoiding enemies. It's a great environment for learning GPU-accelerated training because the ConvNet architecture benefits significantly from GPU.

## Why GPU for Atari?

Training a convolutional network is slow on a CPU due to the large network size. When training on image-based environments like Atari, GPU acceleration provides significant speedup.

{% hint style="warning" %}
GPU does not always accelerate your training. For vector-state environments like CartPole or LunarLander, the speedup isn't enough to counteract the data transfer overhead. Use GPU only for image-based environments with large networks.
{% endhint %}

{% hint style="info" %}
If you encounter CUDA driver issues, see [Help](../resources/help.md) for troubleshooting.
{% endhint %}

## GPU Monitoring

Monitor GPU usage with [glances](https://github.com/nicolargo/glances):

```bash
uv tool install glances
glances
```

{% embed url="https://glances.readthedocs.io/en/stable/aoa/gpu.html" %}

## The Atari Spec

The PPO Atari spec from [slm\_lab/spec/benchmark/ppo/ppo\_atari.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_atari.json):

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

The key setting is **"gpu": "auto"** in the net spec. This automatically uses GPU if available, or falls back to CPU. You can also use `"gpu": true` to force GPU usage.

## Running PPO on Qbert

```bash
slm-lab run -s env=ALE/Qbert-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari train
```

You should see higher **fps** (frames per second) compared to CPU training. The trial takes a few hours to complete on a modern GPU.

### Results

PPO achieves **15094** MA on Qbert-v5.

**Training curve** (average of 4 sessions):

![PPO Qbert Training](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/data/ppo_atari_qbert_2026_01_06_111801/ppo_atari_qbert_t0_trial_graph_mean_returns_vs_frames.png)

**Moving average** (100-checkpoint window):

![PPO Qbert Training MA](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/data/ppo_atari_qbert_2026_01_06_111801/ppo_atari_qbert_t0_trial_graph_mean_returns_ma_vs_frames.png)

Trained models available on [HuggingFace](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_qbert_2026_01_06_111801).

## Other Atari Games

The same spec works for all 54 Atari games. Different games benefit from different lambda values:

| Game | Command | Lambda | Score |
|------|---------|--------|-------|
| Qbert | `slm-lab run -s env=ALE/Qbert-v5 ... ppo_atari train` | 0.95 | 15094 |
| MsPacman | `slm-lab run -s env=ALE/MsPacman-v5 ... ppo_atari_lam85 train` | 0.85 | 2372 |
| Breakout | `slm-lab run -s env=ALE/Breakout-v5 ... ppo_atari_lam70 train` | 0.70 | 327 |

{% hint style="info" %}
**Lambda tuning:** Different games benefit from different lambda values. See [Atari Benchmark](../benchmark-results/atari-benchmark.md) for optimal settings per game.
{% endhint %}

{% hint style="warning" %}
**v5 vs v4 Difficulty:** Gymnasium ALE v5 environments use sticky actions and stricter termination, making them harder than OpenAI Gym v4. Expect 10-40% lower scores.
{% endhint %}

## Using Multiple GPUs

### Automatic GPU Rotation

SLM Lab automatically cycles through available GPUs. With 4 sessions and 2 GPUs:

* Session 0: GPU 0
* Session 1: GPU 1
* Session 2: GPU 0
* Session 3: GPU 1

### Using CUDA\_OFFSET

For manual control when running multiple experiments:

```bash
# First experiment uses GPUs 0-3
slm-lab run -s env=ALE/Qbert-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari train

# Second experiment uses GPUs 4-7
slm-lab run --cuda-offset 4 -s env=ALE/MsPacman-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari_lam85 train
```

{% hint style="info" %}
For search mode and benchmarks, SLM Lab automatically handles GPU allocation across all trials and sessions.
{% endhint %}
