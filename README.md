---
description: Modular Deep Reinforcement Learning framework in PyTorch.
---

# SLM Lab

[![CircleCI](https://circleci.com/gh/kengz/SLM-Lab.svg?style=shield)](https://circleci.com/gh/kengz/SLM-Lab)[![Maintainability](https://api.codeclimate.com/v1/badges/20c6a124c468b4d3e967/maintainability)](https://codeclimate.com/github/kengz/SLM-Lab/maintainability)[![Test Coverage](https://api.codeclimate.com/v1/badges/20c6a124c468b4d3e967/test_coverage)](https://codeclimate.com/github/kengz/SLM-Lab/test_coverage)

SLM Lab is a software framework for reproducible reinforcement learning \(RL\) research. It enables easy development of RL algorithms using modular components and file-based configuration. It also enables flexible experimentation completed with hyperparameter search, result analysis and benchmark results.

SLM Lab is also the companion library of the book [Foundations of Deep Reinforcement Learning](https://www.amazon.com/dp/0135172381).

## ✨ Features

* [Modular design](development/modular-lab-components/) for building deep RL algorithms
* [Reproducibility](using-slm-lab/lab-organization.md#reproducibility-design) using spec file and git SHA
* [Experiment framework](using-slm-lab/lab-organization.md#session-trial-and-experiment) with [automatic analysis](analyzing-results/analytics.md)
* [Extensive benchmark results](benchmark-results/public-benchmark-data.md)
* Well-tuned algorithm implementations
* Multiple RL environment offerings

### Algorithms

SLM Lab implements most of the [canonical RL algorithms](development/modular-lab-components/algorithm-taxonomy.md):

* SARSA
* DQN \(Deep Q-Network\)
* Double-DQN, Dueling-DQN, PER \(Prioritized Experience Replay\)
* REINFORCE
* A2C \(Advantage Actor-Critic\) with GAE & n-step
* PPO \(Proximal Policy Optimization\)
* SAC \(Soft Actor-Critic\)
* SIL \(Self Imitation Learning\)
* Asynchronous version of all the above

They are implemented in a modular way such that differences in algorithm performance can be confidently ascribed to differences between algorithms, not between implementations.

### Environments

SLM Lab currently includes the following environment offerings:

* [OpenAI gym](https://github.com/openai/gym)
* [OpenAI Roboschool](https://github.com/openai/roboschool)
* [VizDoom](https://github.com/mwydmuch/ViZDoom#documentation) \(credit: joelouismarino\)
* [Unity environments](https://github.com/Unity-Technologies/ml-agents) with prebuilt binaries

## Citation

If you use SLM Lab in your publication, please cite below:

```text
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

