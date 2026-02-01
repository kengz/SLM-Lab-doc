---
description: Modular Deep Reinforcement Learning framework in PyTorch.
---

# SLM Lab ⚗️

<p align="center">
  <i>Modular Deep Reinforcement Learning framework in PyTorch.</i>
  <br>
  <a href="https://github.com/kengz/SLM-Lab">GitHub</a> · <a href="benchmark-results/public-benchmark-data.md">Benchmark Results</a>
</p>

{% hint style="info" %}
**v5.0** updates to Gymnasium, `uv` tooling, and modern dependencies with ARM support—see [Changelog](CHANGELOG.md).
{% endhint %}

|||||
|:---:|:---:|:---:|:---:|
| ![ppo beamrider](https://user-images.githubusercontent.com/8209263/63994698-689ecf00-caaa-11e9-991f-0a5e9c2f5804.gif) | ![ppo breakout](https://user-images.githubusercontent.com/8209263/63994695-650b4800-caaa-11e9-9982-2462738caa45.gif) | ![ppo kungfumaster](https://user-images.githubusercontent.com/8209263/63994690-60469400-caaa-11e9-9093-b1cd38cee5ae.gif) | ![ppo mspacman](https://user-images.githubusercontent.com/8209263/63994685-5cb30d00-caaa-11e9-8f35-78e29a7d60f5.gif) |
| BeamRider | Breakout | KungFuMaster | MsPacman |
| ![ppo pong](https://user-images.githubusercontent.com/8209263/63994680-59b81c80-caaa-11e9-9253-ed98370351cd.gif) | ![ppo qbert](https://user-images.githubusercontent.com/8209263/63994672-54f36880-caaa-11e9-9757-7780725b53af.gif) | ![ppo seaquest](https://user-images.githubusercontent.com/8209263/63994665-4dcc5a80-caaa-11e9-80bf-c21db818115b.gif) | ![ppo spaceinvaders](https://user-images.githubusercontent.com/8209263/63994624-15c51780-caaa-11e9-9c9a-854d3ce9066d.gif) |
| Pong | Qbert | Seaquest | Sp.Invaders |
| ![sac ant](https://user-images.githubusercontent.com/8209263/63994867-ff6b8b80-caaa-11e9-971e-2fac1cddcbac.gif) | ![sac halfcheetah](https://user-images.githubusercontent.com/8209263/63994869-01354f00-caab-11e9-8e11-3893d2c2419d.gif) | ![sac hopper](https://user-images.githubusercontent.com/8209263/63994871-0397a900-caab-11e9-9566-4ca23c54b2d4.gif) | ![sac humanoid](https://user-images.githubusercontent.com/8209263/63994883-0befe400-caab-11e9-9bcc-c30c885aad73.gif) |
| Ant | HalfCheetah | Hopper | Humanoid |
| ![sac doublependulum](https://user-images.githubusercontent.com/8209263/63994879-07c3c680-caab-11e9-974c-06cdd25bfd68.gif) | ![sac pendulum](https://user-images.githubusercontent.com/8209263/63994880-085c5d00-caab-11e9-850d-049401540e3b.gif) | ![sac reacher](https://user-images.githubusercontent.com/8209263/63994881-098d8a00-caab-11e9-8e19-a3b32d601b10.gif) | ![sac walker](https://user-images.githubusercontent.com/8209263/63994882-0abeb700-caab-11e9-9e19-b59dc5c43393.gif) |
| Inv.DoublePendulum | InvertedPendulum | Reacher | Walker |

## Quick Start

```bash
# Install
git clone https://github.com/kengz/SLM-Lab.git && cd SLM-Lab
uv sync
uv tool install --editable .

# Run demo (PPO CartPole)
slm-lab run                                    # PPO CartPole
slm-lab run --render                           # with visualization

# Run custom experiment
slm-lab run spec.json spec_name train          # local training
slm-lab run-remote spec.json spec_name train   # cloud training (dstack)

# Help (CLI uses Typer)
slm-lab --help                                 # list all commands
slm-lab run --help                             # options for run command

# Troubleshoot: if slm-lab not found, use uv run
uv run slm-lab run
```

See [Installation](setup/installation.md) for prerequisites and detailed setup.

---

SLM Lab is a software framework for **reinforcement learning** (RL) research and application in PyTorch. RL trains agents to make decisions by learning from trial and error—like teaching a robot to walk or an AI to play games.

## What SLM Lab Offers

| Feature | Description |
|---------|-------------|
| **Ready-to-use algorithms** | PPO, SAC, DQN, A2C, REINFORCE—validated on 70+ environments |
| **Easy configuration** | JSON spec files fully define experiments—no code changes needed |
| **Reproducibility** | Every run saves its spec + git SHA for exact reproduction |
| **Automatic analysis** | Training curves, metrics, and TensorBoard logging out of the box |
| **Cloud integration** | dstack for GPU training, HuggingFace for sharing results |

**SLM Lab is also the companion library of the book** [**Foundations of Deep Reinforcement Learning**](https://www.amazon.com/dp/0135172381)**.**

{% hint style="info" %}
**Book readers:** For the exact code from *Foundations of Deep Reinforcement Learning*, use `git checkout v4.1.1`. The book's [website and errata is here](https://slm-lab.gitbook.io/foundations-of-deep-rl/).
{% endhint %}

## Core Concepts 🏗️

SLM Lab organizes experiments hierarchically:

```
Experiment (hyperparameter search)
 └── Trial (one configuration, multiple seeds)
      └── Session (one training run)
           ├── Agent (algorithm + memory + network)
           └── Env (gymnasium environment)
```

A **spec file** defines everything:

```javascript
{
  "ppo_cartpole": {
    "agent": {
      "name": "PPO",
      "algorithm": {"name": "PPO", "gamma": 0.99, "lam": 0.95},
      "memory": {"name": "OnPolicyBatchReplay"},
      "net": {"type": "MLPNet", "hid_layers": [64, 64]}
    },
    "env": {"name": "CartPole-v1", "num_envs": 4, "max_frame": 200000},
    "meta": {"max_session": 4}
  }
}
```

Run it: `slm-lab run spec.json ppo_cartpole train`

See [Understanding Experiments](using-slm-lab/lab-organization.md) for the full picture.

## Algorithms 🧠

SLM Lab implements the canonical RL algorithms with a [taxonomy-based inheritance](development/modular-lab-components/algorithm-taxonomy.md) design:

| Algorithm | Type | Best For | Validated Environments |
|-----------|------|----------|------------------------|
| **PPO** | On-policy | General purpose | Classic, Box2D, MuJoCo (11), Atari (54) |
| **SAC** | Off-policy | Continuous control | Classic, Box2D, MuJoCo |
| **DQN/DDQN+PER** | Off-policy | Discrete actions | Classic, Box2D, Atari |
| **A2C** | On-policy | Fast iteration | Classic, Box2D, Atari |
| **REINFORCE** | On-policy | Learning/teaching | Classic |
| **SARSA** | On-policy | Tabular-like | Classic |

See [Benchmark Results](benchmark-results/public-benchmark-data.md) for detailed performance data.

## Environments 🌍

SLM Lab uses [Gymnasium](https://gymnasium.farama.org/) (the maintained fork of OpenAI Gym):

| Category | Examples | Difficulty | Docs |
|----------|----------|------------|------|
| **Classic Control** | CartPole, Pendulum, Acrobot | Easy | [Gymnasium Classic](https://gymnasium.farama.org/environments/classic_control/) |
| **Box2D** | LunarLander, BipedalWalker | Medium | [Gymnasium Box2D](https://gymnasium.farama.org/environments/box2d/) |
| **MuJoCo** | Hopper, HalfCheetah, Humanoid | Hard | [Gymnasium MuJoCo](https://gymnasium.farama.org/environments/mujoco/) |
| **Atari** | Pong, Breakout, and 54 more | Varied | [Gymnasium ALE](https://gymnasium.farama.org/environments/atari/) |

Any gymnasium-compatible environment works—just specify its name in the spec.

{% hint style="warning" %}
**v5 vs v4:** Gymnasium environments are harder than OpenAI Gym. Expect 10-30% lower scores vs older benchmarks. See [Benchmark Results](benchmark-results/public-benchmark-data.md) for validated scores.
{% endhint %}

## Key RL Terms 📚

Quick reference for terms used throughout this documentation. For deeper coverage, see [Foundations of Deep Reinforcement Learning](https://www.amazon.com/dp/0135172381).

| Term | Meaning |
|------|---------|
| **Agent** | The learner that takes actions and receives rewards |
| **Environment** | The world the agent interacts with (e.g., CartPole, Atari game) |
| **State** | What the agent observes (e.g., pole angle, screen pixels) |
| **Action** | What the agent does (e.g., move left/right) |
| **Reward** | Feedback signal indicating how good an action was |
| **Policy** | The agent's strategy—maps states to actions |
| **Episode** | One complete run from start to terminal state |
| **Frame** | One environment step (action → reward → next state) |
| **Gamma (γ)** | Discount factor—how much to value future vs immediate rewards |
| **On-policy** | Learn from actions taken by current policy (PPO, A2C) |
| **Off-policy** | Learn from any actions, including past data (DQN, SAC) |

## Documentation Guide 📖

**Getting Started:**
1. [Installation](setup/installation.md) - Set up SLM Lab
2. [Quick Start](setup/quick-start.md) - Verify installation
3. [Train: PPO on CartPole](using-slm-lab/train-ppo-cartpole.md) - First training run

**Tutorials:**
- [Agent Spec](using-slm-lab/agent-spec-ddqn+per-on-lunarlander.md) - Configure algorithms (LunarLander)
- [Env Spec](using-slm-lab/environment-spec-a2c-on-bipedalwalker.md) - Configure environments (MuJoCo)
- [GPU Training](using-slm-lab/gpu-usage-ppo-on-pong.md) - Train on Atari with GPU
- [Hyperparameter Search](using-slm-lab/search-spec-ppo-on-breakout.md) - Find optimal settings

**Reference & Advanced:**
- [CLI Reference](using-slm-lab/slm-lab-command.md) - All CLI commands and options
- [Understanding Experiments](using-slm-lab/lab-organization.md) - Sessions, Trials, Experiments
- [Remote Training](using-slm-lab/remote-training.md) - Cloud GPU training with dstack
- [Architecture](development/architecture.md) - How SLM Lab works

## Citation 📝

If you use SLM Lab in your publication, please cite:

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
