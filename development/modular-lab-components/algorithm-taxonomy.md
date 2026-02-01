# Algorithm Taxonomy 🌳

## Algorithm Family Tree

Deep RL algorithms can be classified into a family tree based on their methods / functions they learn, such as the one shown below.

![Source: Foundations of Deep Reinforcement Learning, Graesser & Keng.](../../.gitbook/assets/algorithm_tree.png)

Algorithms often extends an existing one by modifying or adding components. Most model-free algorithms are descended from SARSA and REINFORCE. The figure below shows some of the algorithms in SLM Lab, and their relationships.

![Source: Foundations of Deep Reinforcement Learning, Graesser & Keng.](../../.gitbook/assets/algo_hierarchy-ak.png)

Naturally, implementations can be consistent with this theoretical taxonomy by using **class inheritance** and **modular components**. This is precisely what SLM Lab does.

## Implemented Algorithms

### Policy Gradient Family (from REINFORCE)

| Algorithm | Class | Advantage | Memory | Description |
|-----------|-------|-----------|--------|-------------|
| REINFORCE | `Reinforce` | Monte Carlo | `OnPolicyReplay` | Vanilla policy gradient |
| A2C (GAE) | `ActorCritic` | GAE | `OnPolicyBatchReplay` | Actor-Critic with Generalized Advantage Estimation |
| A2C (n-step) | `ActorCritic` | n-step returns | `OnPolicyBatchReplay` | Actor-Critic with n-step bootstrapping |
| A3C (GAE) | `ActorCritic` | GAE | `OnPolicyBatchReplay` | Async A2C with Hogwild! |
| A3C (n-step) | `ActorCritic` | n-step returns | `OnPolicyBatchReplay` | Async n-step A2C |
| PPO | `PPO` | GAE | `OnPolicyBatchReplay` | Proximal Policy Optimization |

### Value-Based Family (from SARSA)

| Algorithm | Class | Target Network | Memory | Description |
|-----------|-------|----------------|--------|-------------|
| SARSA | `SARSA` | No | `OnPolicyReplay` | On-policy TD learning |
| VanillaDQN | `VanillaDQN` | No | `Replay` | DQN without target network |
| DQN | `DQN` | Yes (replace) | `Replay` | Standard DQN with periodic target updates |
| Double DQN | `DoubleDQN` | Yes | `Replay` | Separate action selection/evaluation |
| DQN + PER | `DQN` | Yes | `PrioritizedReplay` | Prioritized experience replay |
| DDQN + PER | `DoubleDQN` | Yes | `PrioritizedReplay` | Combined Double DQN + PER |
| Dueling DQN | `DQN` | Yes | `Replay` | Separate value/advantage streams (via DuelingMLPNet) |

### Off-Policy Actor-Critic

| Algorithm | Class | Entropy | Memory | Description |
|-----------|-------|---------|--------|-------------|
| SAC | `SoftActorCritic` | Auto-tuned | `Replay` / `PrioritizedReplay` | Maximum entropy RL |
| Async SAC | `SoftActorCritic` | Auto-tuned | `Replay` | SAC with Hogwild! |

## Supported Environments

SLM Lab validates algorithms across four environment categories:

| Category | Envs | Action Space | Algorithms |
|----------|------|--------------|------------|
| Classic Control | CartPole, Acrobot, Pendulum | Discrete/Continuous | All |
| Box2D | LunarLander, BipedalWalker | Discrete/Continuous | DQN, A2C, PPO, SAC |
| MuJoCo | 11 locomotion tasks | Continuous | PPO, SAC |
| Atari | 54 games | Discrete | PPO |

See [Running Benchmarks](../../using-slm-lab/benchmark-specs.md) for complete algorithm × environment matrix.
