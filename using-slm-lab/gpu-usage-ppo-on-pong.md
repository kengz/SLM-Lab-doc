# GPU Training 🎮

This tutorial shows how to train on Atari games using GPU acceleration.

![Breakout](https://user-images.githubusercontent.com/8209263/63994695-650b4800-caaa-11e9-9982-2462738caa45.gif)

**[Breakout](https://gymnasium.farama.org/environments/atari/breakout/)** is a classic Atari benchmark—break bricks by bouncing a ball with a paddle. It's a great environment for learning GPU-accelerated training because the ConvNet architecture benefits significantly from GPU.

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

## Running PPO on Breakout

```bash
slm-lab run -s env=ALE/Breakout-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari_lam70 train
```

You should see higher **fps** (frames per second) compared to CPU training. The trial takes a few hours to complete on a modern GPU.

### 📊 Results

PPO achieves **327** MA on Breakout-v5.

**Training curve** (session 3):

![PPO Breakout Training](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/data/ppo_atari_lam70_breakout_2026_01_07_110559/graph/ppo_atari_lam70_breakout_t0_s3_session_graph_train_mean_returns_ma_vs_frames.png)

Trained models available on [HuggingFace](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_breakout_2026_01_07_110559).

## Other Atari Games

The same spec works for all 54 Atari games:

| Game | Command | Lambda |
|------|---------|--------|
| MsPacman | `slm-lab run -s env=ALE/MsPacman-v5 ... ppo_atari_lam85 train` | 0.85 |
| Qbert | `slm-lab run -s env=ALE/Qbert-v5 ... ppo_atari train` | 0.95 |
| Pong | `slm-lab run -s env=ALE/Pong-v5 ... ppo_atari_lam85 train` | 0.85 |

{% hint style="info" %}
**Lambda tuning:** Different games benefit from different lambda values. See [Atari Benchmark](../benchmark-results/atari-benchmark.md) for optimal settings per game.
{% endhint %}

{% hint style="warning" %}
**v5 vs v4 Difficulty:** Gymnasium ALE v5 environments use sticky actions and stricter termination, making them harder than OpenAI Gym v4. Expect 10-40% lower scores. Pong is notably harder in v5 (16.9 vs 20.6 in v4).
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
slm-lab run -s env=ALE/Breakout-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari_lam70 train

# Second experiment uses GPUs 4-7
slm-lab run --cuda-offset 4 -s env=ALE/Qbert-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari train
```

{% hint style="info" %}
For search mode and benchmarks, SLM Lab automatically handles GPU allocation across all trials and sessions.
{% endhint %}
