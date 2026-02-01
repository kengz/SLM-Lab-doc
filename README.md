---
description: Modular Deep Reinforcement Learning framework in PyTorch.
---

# SLM Lab 🧪

SLM Lab is a software framework for **reinforcement learning** (RL) research and application in PyTorch. RL trains agents to make decisions by learning from trial and error—like teaching a robot to walk 🤖 or an AI to play games 🎮.

## What SLM Lab Offers ✨

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

## Quick Start 🚀

Install and run in under 2 minutes:

```bash
# Install uv (package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/kengz/SLM-Lab.git
cd SLM-Lab
uv sync
uv tool install --editable .

# Train PPO on CartPole with visualization
slm-lab run --render
```

You should see a CartPole balancing task with rewards climbing toward 500. See [Quick Start](setup/quick-start.md) for details.

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

See [Lab Organization](using-slm-lab/lab-organization.md) for the full picture.

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
3. [Lab Command](using-slm-lab/slm-lab-command.md) - CLI reference
4. [Lab Organization](using-slm-lab/lab-organization.md) - Core concepts

**Tutorials:**
- [Train: PPO CartPole](using-slm-lab/train-ppo-cartpole.md) - First training run
- [Agent Spec](using-slm-lab/agent-spec-ddqn+per-on-lunarlander.md) - Configure algorithms
- [Env Spec](using-slm-lab/environment-spec-a2c-on-bipedalwalker.md) - Configure environments
- [Hyperparameter Search](using-slm-lab/search-spec-ppo-on-breakout.md) - Find optimal settings

**Advanced:**
- [Architecture](development/architecture.md) - How SLM Lab works
- [Using SLM Lab In Your Project](using-slm-lab/using-slm-lab-in-your-project.md) - Integration guide
- [Remote Training](using-slm-lab/remote-training.md) - Cloud GPU training

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
