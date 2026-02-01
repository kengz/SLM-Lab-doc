# Book: Foundations of Deep Reinforcement Learning 📚

SLM Lab is the companion software library of the book [Foundations of Deep Reinforcement Learning](https://www.amazon.com/dp/0135172381) by Laura Graesser and Wah Loon Keng.

![](<../.gitbook/assets/book cover outline.png>)

**Book website and errata:** [https://slm-lab.gitbook.io/foundations-of-deep-rl](https://slm-lab.gitbook.io/foundations-of-deep-rl/)

{% hint style="info" %}
**For book readers:**

The book was written with SLM Lab v4. **Use v5** (this documentation) to run experiments—v4 has compatibility issues on modern machines (ARM, Python 3.10+).

**v5 is mostly compatible with the book:**
- Algorithms and concepts are the same
- Main changes are in spec format and command syntax (see [Changelog](../CHANGELOG.md))
- Gymnasium replaces OpenAI Gym (environment names updated)

**For exact code reference:** checkout `v4.1.1` branch. Note that v4 may not run on ARM Macs or newer Python versions.

```bash
git checkout v4.1.1  # Only for code reference, not recommended for running
```
{% endhint %}

## About the Book

**Foundations of Deep Reinforcement Learning** is an introduction to deep RL that uniquely combines both theory and implementation. It starts with intuition, then carefully explains the theory of deep RL algorithms, discusses implementations in SLM Lab, and finishes with practical details for getting deep RL to work.

**Topics covered:**
- Policy- and value-based algorithms: REINFORCE, SARSA, DQN, Double DQN, PER
- Combined algorithms: Actor-Critic, PPO
- Parallelization: synchronous and asynchronous methods
- Practical implementation details and hyperparameter tuning

The book is ideal for computer science students and software engineers familiar with basic machine learning and Python.

{% hint style="warning" %}
**Disclaimer:** The book is *not required* for using SLM Lab. It's a separate effort providing approachable access to deep RL theory with practical implementations.
{% endhint %}

## Editorial Reviews

> *"This book provides an accessible introduction to deep reinforcement learning covering the mathematical concepts behind popular algorithms as well as their practical implementation."*
> — **Volodymyr Mnih**, lead developer of DQN

> *"An excellent book to quickly develop expertise in the theory, language, and practical implementation of deep reinforcement learning algorithms."*
> — **Vincent Vanhoucke**, principal scientist, Google

> *"I imagine this will become an invaluable resource for individuals interested in learning about deep reinforcement learning for years to come."*
> — **Arthur Juliani**, senior ML engineer, Unity Technologies
