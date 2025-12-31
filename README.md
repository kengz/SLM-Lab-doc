---
description: Modular Deep Reinforcement Learning framework in PyTorch.
---

# SLM Lab

![GitHub tag (latest SemVer)](https://img.shields.io/github/tag/kengz/slm-lab) ![CI](https://github.com/kengz/SLM-Lab/workflows/CI/badge.svg) [![Maintainability](https://api.codeclimate.com/v1/badges/20c6a124c468b4d3e967/maintainability)](https://codeclimate.com/github/kengz/SLM-Lab/maintainability) [![Test Coverage](https://api.codeclimate.com/v1/badges/20c6a124c468b4d3e967/test_coverage)](https://codeclimate.com/github/kengz/SLM-Lab/test_coverage)

SLM Lab is a software framework for reproducible reinforcement learning (RL) research. It enables easy development of RL algorithms using modular components and file-based configuration. It also enables flexible experimentation with hyperparameter search, result analysis and benchmarking.

**SLM Lab is also the companion library of the book** [**Foundations of Deep Reinforcement Learning**](https://www.amazon.com/dp/0135172381)**.**

{% hint style="info" %}
**Book readers:** For the exact code from *Foundations of Deep Reinforcement Learning*, use `git checkout v4.1.1`. The book's [website and errata is here](https://slm-lab.gitbook.io/foundations-of-deep-rl/).
{% endhint %}

## What's New in v5

SLM Lab v5 is a modernization release for the current RL ecosystem:

* **Gymnasium** replaces OpenAI Gym with proper `terminated`/`truncated` handling
* **uv** replaces conda for fast, reliable dependency management
* **Simpler specs** — no more `body` section or array wrappers
* **Cloud training** via dstack with HuggingFace result sync
* **ASHA search** for efficient hyperparameter tuning with early stopping
* **PPO enhancements**: `normalize_v_targets`, `symlog_transform`, `clip_vloss`
* **Network options**: `layer_norm` for MLP stability

See [Installation](setup/installation.md) for migration details.

### Gymnasium API: terminated vs truncated

v5 uses the modern Gymnasium API which separates episode endings:

```python
# Old (OpenAI Gym): single 'done' flag
state, reward, done, info = env.step(action)

# New (Gymnasium): separate 'terminated' and 'truncated'
state, reward, terminated, truncated, info = env.step(action)
```

* **terminated**: Episode ended due to task completion (goal reached, agent died, etc.)
* **truncated**: Episode ended due to time limit or external constraint

This distinction is important for correct value bootstrapping—truncated episodes should bootstrap from the final state while terminated episodes should not. All SLM Lab algorithms handle this correctly.

## Quick Start

```bash
git clone https://github.com/kengz/SLM-Lab.git && cd SLM-Lab
uv sync && uv tool install --editable .
slm-lab run --render   # PPO on CartPole in dev mode with visualization
```

See [Installation](setup/installation.md) for uv setup and [Quick Start](setup/quick-start.md) to verify your installation.

## Features

* [Modular design](development/modular-lab-components/) for building deep RL algorithms
* [Reproducibility](using-slm-lab/lab-organization.md#reproducibility-design) using spec file and git SHA
* [Experiment framework](using-slm-lab/lab-organization.md#session-trial-and-experiment) with [automatic analysis](analyzing-results/analytics.md)
* [Extensive benchmark results](benchmark-results/public-benchmark-data.md)
* Well-tuned algorithm implementations
* Multiple RL environment offerings

### Algorithms

SLM Lab implements most of the [canonical RL algorithms](development/modular-lab-components/algorithm-taxonomy.md):

| Algorithm | v5 Status | Environments |
|-----------|-----------|--------------|
| PPO | ✅ Validated | Classic, Box2D, MuJoCo (11), Atari (24+ solved) |
| SAC | ✅ Validated | Classic, Box2D, MuJoCo (11) |
| DQN/DDQN+PER | ✅ Validated | Classic, Box2D |
| A2C | ✅ Validated | Classic, Box2D |
| REINFORCE | ✅ Validated | Classic |
| SARSA | ⏸️ Pending | - |
| SIL | ⏸️ Pending | - |
| Async (A3C, DPPO) | ⏸️ Pending | - |

See [Benchmark Results](benchmark-results/public-benchmark-data.md) for detailed performance data.

### Environments

SLM Lab uses [Gymnasium](https://gymnasium.farama.org/) (the maintained fork of OpenAI Gym) for environment support:

* **Classic control:** CartPole, Pendulum, Acrobot, MountainCar
* **Box2D:** LunarLander, BipedalWalker
* **MuJoCo:** Hopper, HalfCheetah, Walker2d, Ant, Humanoid, and more
* **Atari:** All 57 Atari 2600 games via ALE (Arcade Learning Environment)

Any gymnasium-compatible environment can be used by specifying its name in the spec file.

## Citation

If you use SLM Lab in your publication, please cite below:

```
@misc{kenggraesser2017slmlab,
    author = {Keng, Wah Loon and Graesser, Laura},
    title = {SLM Lab},
    year = {2017},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/kengz/SLM-Lab}},
}
```

## License

This project is licensed under the [MIT License](https://github.com/kengz/SLM-Lab/blob/master/LICENSE).

