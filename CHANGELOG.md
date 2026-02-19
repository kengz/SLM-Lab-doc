# Changelog 📋

This page documents major framework releases.

For detailed code changes, see the [GitHub releases](https://github.com/kengz/SLM-Lab/releases) and [CHANGELOG.md](https://github.com/kengz/SLM-Lab/blob/master/CHANGELOG.md) in the code repository.

---

## SLM-Lab v5.1.0

TorchArc YAML benchmarks replace original hardcoded network architectures across all benchmark categories.

- **TorchArc integration**: All algorithms (REINFORCE, SARSA, DQN, DDQN+PER, A2C, PPO, SAC) now use TorchArc YAML-defined networks instead of hardcoded PyTorch modules.
- **Full benchmark validation**: Classic Control, Box2D, MuJoCo (11 envs), and Atari (54 games) re-benchmarked with TorchArc — results match or exceed original scores.
- **SAC Atari**: New SAC Atari benchmarks (48 games) with discrete action support.
- **A2C Atari**: A2C benchmarks across all 57 Atari games.
- **Pre-commit hooks**: Conventional commit message validation via `.githooks/commit-msg`.

---

## SLM-Lab v5.0.0

Modernization release for the current RL ecosystem. This release updates SLM Lab to work with the modern Python RL stack while maintaining backward compatibility with the book *Foundations of Deep Reinforcement Learning*.

{% hint style="info" %}
**Book readers:** For exact code from *Foundations of Deep Reinforcement Learning*, use `git checkout v4.1.1`
{% endhint %}

### Critical: Atari v5 Sticky Actions

**SLM-Lab uses Gymnasium ALE v5 defaults.** v5 default `repeat_action_probability=0.25` (sticky actions) randomly repeats agent actions to simulate console stochasticity, making evaluation harder but more realistic than v4 default 0.0 used by most benchmarks (CleanRL, SB3, RL Zoo). This follows [Machado et al. (2018)](https://arxiv.org/abs/1709.06009) research best practices. See [ALE version history](https://ale.farama.org/environments/#version-history-and-naming-schemes).

### Why v5?

The RL ecosystem has evolved significantly since SLM Lab v4:

1. **OpenAI Gym → Gymnasium**: OpenAI deprecated Gym in 2022. Gymnasium (by Farama Foundation) is the maintained fork with better API design
2. **Roboschool → MuJoCo**: Roboschool was abandoned. MuJoCo became free in 2022 and is now the standard for continuous control
3. **Conda → uv**: Modern Python dependency management is faster and more reliable with `uv`
4. **Simpler specs**: Removed legacy multi-agent abstractions that added complexity without benefit

### Key Changes

| Category | v4 | v5 |
|----------|----|----|
| **Package manager** | conda | uv |
| **Environment library** | OpenAI Gym | Gymnasium |
| **Continuous control** | Roboschool | MuJoCo |
| **Entry point** | `python run_lab.py` | `slm-lab run` |
| **Spec format** | Arrays with `body` | Simple objects |

### Migration Summary

| v4 | v5 |
|----|----|
| `conda activate lab && python run_lab.py` | `slm-lab run` |
| `CartPole-v0` | `CartPole-v1` |
| `Acrobot-v1` | `Acrobot-v1` (unchanged) |
| `Pendulum-v0` | `Pendulum-v1` |
| `LunarLander-v2` | `LunarLander-v3` |
| `PongNoFrameskip-v4` | `ALE/Pong-v5` |
| `BreakoutNoFrameskip-v4` | `ALE/Breakout-v5` |
| `RoboschoolHopper-v1` | `Hopper-v5` (MuJoCo) |
| `RoboschoolHalfCheetah-v1` | `HalfCheetah-v5` (MuJoCo) |
| `RoboschoolHumanoid-v1` | `Humanoid-v5` (MuJoCo) |
| `agent: [{...}]`, `env: [{...}]`, `body: {...}` | `agent: {...}`, `env: {...}` |

### Gymnasium API Change

The most significant change is how episode endings are handled. v5 uses the modern Gymnasium API which separates episode endings into two distinct signals:

```python
# Old (OpenAI Gym)
state, reward, done, info = env.step(action)

# New (Gymnasium)
state, reward, terminated, truncated, info = env.step(action)
```

**What's the difference?**

* **terminated**: Episode ended due to the task itself (goal reached, agent died, game over)
* **truncated**: Episode ended due to external limits (time limit, max steps reached)

**Why does this matter?**

This distinction is critical for correct value bootstrapping in RL algorithms:

```python
# Correct handling (v5)
if terminated:
    # True episode end - don't bootstrap from next state
    target = reward
else:
    # Truncated or continuing - bootstrap from next state value
    target = reward + gamma * V(next_state)
```

In v4, algorithms had to guess whether `done=True` meant a real ending or just a time limit. This led to subtle bugs and inconsistent behavior. All SLM Lab v5 algorithms handle this correctly.

### New v5 Features

**Algorithm improvements:**
* **PPO:** `normalize_v_targets` for running statistics normalization, `symlog_transform` (from DreamerV3), `clip_vloss` (CleanRL-style)
* **SAC:** Discrete action support uses exact expectation (Christodoulou 2019). Target entropy auto-calculated.
* **Networks:** Optional `layer_norm` for MLP hidden layers
* `life_loss_info`: Proper Atari game-over handling (continue after life loss)

**Infrastructure:**
* Ray Tune ASHA search for efficient hyperparameter tuning
* dstack integration for cloud GPU training
* HuggingFace integration for experiment storage and sharing

**Benchmarks:**

All algorithms validated on Gymnasium. Full results in [Benchmark Results](benchmark-results/public-benchmark-data.md).

| Category | REINFORCE | SARSA | DQN | DDQN+PER | A2C | PPO | SAC |
|----------|-----------|-------|-----|----------|-----|-----|-----|
| Classic Control | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Box2D | — | — | ✅ | ✅ | ⚠️ | ✅ | ✅ |
| MuJoCo (11 envs) | — | — | — | — | ⚠️ | ✅ All | ✅ All |
| Atari | — | — | — | — | ✅ 57 | ✅ 57 | ✅ 48 |

**Atari benchmarks** use ALE v5 with sticky actions (`repeat_action_probability=0.25`). PPO tested with lambda variants (0.95, 0.85, 0.70) to optimize per-game performance. A2C uses GAE with lambda 0.95. SAC uses Categorical action distribution with training_iter=3 at 2M frames.

Trained models available on [HuggingFace](https://huggingface.co/datasets/SLM-Lab/benchmark).

### Deprecations

* **Roboschool** → Use Gymnasium MuJoCo (`Hopper-v5`, `HalfCheetah-v5`, etc.)
* **Unity ML-Agents / VizDoom** → Removed from core; use their gymnasium wrappers
* **Multi-agent specs** → Simplified to single-agent single-env

### Upgrading Specs

**v4 spec format:**
```javascript
{
  "ppo_cartpole": {
    "agent": [{
      "name": "PPO",
      "algorithm": {...},
      "memory": {...},
      "net": {...}
    }],
    "env": [{
      "name": "CartPole-v0",
      ...
    }],
    "body": {
      "product": "outer",
      "num": 1
    },
    "meta": {...}
  }
}
```

**v5 spec format:**
```javascript
{
  "ppo_cartpole": {
    "agent": {
      "name": "PPO",
      "algorithm": {...},
      "memory": {...},
      "net": {...}
    },
    "env": {
      "name": "CartPole-v1",
      ...
    },
    "meta": {...}
  }
}
```

Key differences:
1. Remove array wrappers `[{...}]` → `{...}`
2. Remove `body` section entirely
3. Update environment names to Gymnasium versions

See [Installation](setup/installation.md) for full setup instructions.
