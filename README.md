---
description: Modular Deep Reinforcement Learning framework in PyTorch.
---

# SLM Lab

<p align="center">
  <i>Modular Deep Reinforcement Learning framework in PyTorch.</i>
  <br>
  <i>Companion library of the book <a href="https://www.amazon.com/dp/0135172381">Foundations of Deep Reinforcement Learning</a>.</i>
  <br>
  <a href="https://github.com/kengz/SLM-Lab">GitHub</a> · <a href="benchmark-results/public-benchmark-data.md">Benchmark Results</a>
</p>

{% hint style="info" %}
**v5.0** updates to Gymnasium, `uv` tooling, and modern dependencies with ARM support—see [Changelog](CHANGELOG.md).

**Book readers:** use `git checkout v4.1.1` for the book code. See [book website and errata](https://slm-lab.gitbook.io/foundations-of-deep-rl/).
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

SLM Lab is a software framework for **reinforcement learning** (RL) research and application in PyTorch. RL trains agents to make decisions by learning from trial and error—like teaching a robot to walk or an AI to play games.

## What SLM Lab Offers

| Feature | Description |
|---------|-------------|
| **Ready-to-use algorithms** | PPO, SAC, DQN, A2C, REINFORCE—validated on 70+ environments |
| **Easy configuration** | JSON spec files fully define experiments—no code changes needed |
| **Reproducibility** | Every run saves its spec + git SHA for exact reproduction |
| **Automatic analysis** | Training curves, metrics, and TensorBoard logging out of the box |
| **Cloud integration** | dstack for GPU training, HuggingFace for sharing results |

## Algorithms

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

## Environments

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

## Citation

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
