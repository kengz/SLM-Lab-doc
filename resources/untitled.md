# Deep RL Resources 📖

## Master List

For a comprehensive, continuously updated collection:

{% embed url="https://github.com/kengz/awesome-deep-rl" %}

## Books

**Recommended order for learning:**

1. **Graesser and Keng**, [Foundations of Deep Reinforcement Learning](https://www.amazon.com/dp/0135172381)
   - Best for beginners; SLM Lab is the companion library
   - Covers REINFORCE, DQN, A2C, PPO with code examples

2. **Sutton and Barto**, [Reinforcement Learning: An Introduction](https://www.amazon.com/dp/0262039249)
   - The classic RL textbook; comprehensive theory
   - [Free online version](http://incompleteideas.net/book/the-book-2nd.html)

3. **Francois-Lavet et al.**, [An Introduction to Deep Reinforcement Learning](https://www.amazon.com/dp/1680835386)
   - More technical; good for researchers
   - [Free arXiv version](https://arxiv.org/abs/1811.12560)

## Video Courses

| Course | Level | Focus |
|--------|-------|-------|
| [David Silver UCL RL Course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html) | Intro | Classic RL theory |
| [Deep RL Bootcamp 2017](https://sites.google.com/view/deep-rl-bootcamp/lectures) | Intermediate | Practical deep RL |
| [DeepMind UCL Deep RL 2018](https://www.youtube.com/playlist?list=PLqYmG7hTraZDNJre23vqCGIVpfZ_K2RZs) | Intermediate | Modern deep RL |
| [Sergey Levine CS294](http://rail.eecs.berkeley.edu/deeprlcourse-fa17/index.html) | Advanced | Research-level |

## Tutorials

**Getting started:**
- [Andrew Karpathy: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/) - Classic intro, policy gradients from scratch
- [OpenAI Spinning Up](https://spinningup.openai.com/) - Excellent intro with clean implementations

**Code-focused:**
- [dennybritz/reinforcement-learning](https://github.com/dennybritz/reinforcement-learning) - Many algorithm implementations
- [higgsfield/RL-Adventure](https://github.com/higgsfield/RL-Adventure) - DQN variants
- [higgsfield/RL-Adventure-2](https://github.com/higgsfield/RL-Adventure-2) - Policy gradient methods

## Papers by Algorithm

### Value-Based Methods

| Algorithm | Paper | Key Idea |
|-----------|-------|----------|
| DQN | [Mnih et al. 2013](https://arxiv.org/abs/1312.5602) | Deep Q-learning with experience replay |
| Double DQN | [van Hasselt et al. 2015](https://arxiv.org/abs/1509.06461) | Reduce overestimation bias |
| Dueling DQN | [Wang et al. 2016](https://arxiv.org/abs/1511.06581) | Separate value and advantage streams |
| PER | [Schaul et al. 2015](https://arxiv.org/abs/1511.05952) | Prioritize important transitions |
| CER | [Zhang & Sutton 2017](https://arxiv.org/abs/1712.01275) | Always include latest transition |

### Policy Gradient Methods

| Algorithm | Paper | Key Idea |
|-----------|-------|----------|
| A3C | [Mnih et al. 2016](https://arxiv.org/abs/1602.01783) | Asynchronous actor-critic |
| GAE | [Schulman et al. 2015](https://arxiv.org/abs/1506.02438) | Better advantage estimation |
| PPO | [Schulman et al. 2017](https://arxiv.org/abs/1707.06347) | Clipped surrogate objective |
| SAC | [Haarnoja et al. 2018](https://arxiv.org/abs/1812.05905) | Maximum entropy RL |

### Other Notable Papers

| Topic | Paper | Relevance |
|-------|-------|-----------|
| Benchmarking | [Henderson et al. 2017](https://arxiv.org/abs/1709.06560) | Why implementation details matter |
| HER | [Andrychowicz et al. 2017](https://arxiv.org/abs/1707.01495) | Learning from failures |
| QT-Opt | [Kalashnikov et al. 2018](https://arxiv.org/abs/1806.10293) | Real-world robot learning |
| SIL | [Oh et al. 2018](https://arxiv.org/abs/1806.05635) | Learn from good past experiences |

## Reference Implementations

| Library | Focus | Notes |
|---------|-------|-------|
| [SLM Lab](https://github.com/kengz/SLM-Lab) | Research, modularity | This project |
| [CleanRL](https://github.com/vwxyzjn/cleanrl) | Single-file implementations | Great for learning |
| [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) | Production use | Well-tested, easy API |
| [RLlib](https://docs.ray.io/en/latest/rllib/) | Distributed training | Scalable |

## Staying Current

- [r/reinforcementlearning](https://www.reddit.com/r/reinforcementlearning/) - Community discussions
- [Papers With Code - RL](https://paperswithcode.com/area/playing-games) - Benchmarks and papers
- [@hardmaru](https://twitter.com/hardmaru) - ML research highlights
