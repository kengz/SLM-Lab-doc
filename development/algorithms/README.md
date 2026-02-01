# Algorithm Families 🧠

{% hint style="info" %}
**v4 Book Readers:** This documentation uses v5 spec format. If using v4.1.1, see [Changelog](../../CHANGELOG.md) for spec format differences.
{% endhint %}

## Overview

Algorithm classes implement RL algorithms: network architecture, action selection, and gradient updates. SLM Lab's algorithms use a taxonomy-based inheritance design where each algorithm extends its parent by adding only its distinguishing features.

**Code:** [slm\_lab/agent/algorithm](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/agent/algorithm)

## Algorithm Taxonomy

```
Algorithm (base class)
 ├── SARSA (tabular-like Q-learning)
 │    └── VanillaDQN → DQNBase → DQN → DoubleDQN
 └── Reinforce (policy gradient)
      └── ActorCritic (adds value function, GAE/n-step)
           ├── PPO (adds clipped objective)
           └── SoftActorCritic (adds entropy regularization)
```

Each level adds only its distinguishing features. For example, PPO inherits everything from ActorCritic and only overrides the policy loss calculation. Note: ActorCritic **is** A2C—there's no separate A2C class.

See [Class Inheritance: A2C > PPO](../modular-lab-components/class-inheritance-a2c-greater-than-ppo.md) for a detailed example.

## Implemented Algorithms

| Algorithm | Type | Action Space | Key Features |
|-----------|------|--------------|--------------|
| **SARSA** | Value-based | Discrete | On-policy TD learning |
| **VanillaDQN** | Value-based | Discrete | Basic Q-learning with neural network |
| **DQN** | Value-based | Discrete | + Target network |
| **DoubleDQN** | Value-based | Discrete | + Double Q-learning |
| **REINFORCE** | Policy gradient | Both | Monte Carlo policy gradient |
| **ActorCritic** | Actor-Critic | Both | Separate actor and critic |
| **A2C** | Actor-Critic | Both | + Synchronized updates |
| **PPO** | Actor-Critic | Both | + Clipped surrogate objective |
| **SAC** | Actor-Critic | Continuous | + Maximum entropy RL |

## Algorithm Interface

All algorithms implement this interface:

```python
class Algorithm:
    def init_algorithm_params(self):
        """Initialize hyperparameters from spec."""
        pass

    def init_nets(self, global_nets=None):
        """Create neural networks and optimizers."""
        pass

    def act(self, state) -> action:
        """Select action given current state."""
        pass

    def train(self) -> loss:
        """Sample from memory and update networks."""
        pass

    def update(self) -> explore_var:
        """Update exploration parameters (epsilon, entropy)."""
        pass
```

## Algorithm Spec

Configure algorithms in the agent spec:

```javascript
{
  "agent": {
    "name": "PPO",
    "algorithm": {
      // Required
      "name": "PPO",              // Algorithm class name
      "gamma": 0.99,              // Discount factor

      // Action selection
      "action_pdtype": "default", // Probability distribution type
      "action_policy": "default", // Action selection policy

      // Algorithm-specific parameters
      "lam": 0.95,                // GAE lambda (PPO, A2C)
      "time_horizon": 128,        // Steps before update (PPO)
      "minibatch_size": 64,       // Minibatch size (PPO)
      "training_epoch": 4,        // Epochs per update (PPO)
      "clip_eps_spec": {...},     // Clipping schedule (PPO)
      "entropy_coef_spec": {...}  // Entropy bonus schedule
    }
  }
}
```

## Key Parameters

### Common Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `gamma` | Discount factor (how much to value future rewards) | 0.99 (long-horizon), 0.9 (short-horizon) |
| `action_pdtype` | Probability distribution for actions | `"default"` (auto-select), `"Categorical"`, `"Normal"` |
| `action_policy` | How to select actions | `"default"`, `"epsilon_greedy"`, `"boltzmann"` |

### Policy Gradient Parameters (A2C, PPO)

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `lam` | GAE lambda (bias-variance tradeoff) | 0.95 (balanced), 0.99 (high variance), 0.7 (low variance) |
| `entropy_coef_spec` | Entropy bonus for exploration | 0.01 (typical), 0.001 (less exploration) |
| `val_loss_coef` | Value loss weight | 0.5 (typical) |

### PPO-Specific Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `time_horizon` | Steps collected before each update | 128 (typical), 2048 (MuJoCo) |
| `minibatch_size` | Samples per gradient step | 64-256 |
| `training_epoch` | Passes through collected data | 4-10 |
| `clip_eps_spec` | Clipping parameter | 0.1-0.2 |

### DQN-Specific Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `explore_var_spec` | Epsilon schedule | Start 1.0, end 0.01 |
| `training_frequency` | Steps between updates | 1-4 |
| `training_start_step` | Steps before training starts | 1000-10000 |

## Exploration Schedules

Many parameters use schedules for decay during training:

```javascript
{
  "explore_var_spec": {
    "name": "linear_decay",   // Decay type
    "start_val": 1.0,         // Initial value
    "end_val": 0.01,          // Final value
    "start_step": 0,          // When to start decay
    "end_step": 50000         // When to reach end_val
  }
}
```

Available schedules:
- `"no_decay"` - Constant value
- `"linear_decay"` - Linear interpolation
- `"rate_decay"` - Exponential decay

## Example Specs

### PPO for CartPole (Discrete)

```javascript
{
  "algorithm": {
    "name": "PPO",
    "gamma": 0.99,
    "lam": 0.95,
    "time_horizon": 128,
    "minibatch_size": 64,
    "training_epoch": 4,
    "clip_eps_spec": {"name": "no_decay", "start_val": 0.2, "end_val": 0.2},
    "entropy_coef_spec": {"name": "no_decay", "start_val": 0.01, "end_val": 0.01}
  }
}
```

### DQN for LunarLander (Discrete)

```javascript
{
  "algorithm": {
    "name": "DQN",
    "action_pdtype": "Argmax",
    "action_policy": "epsilon_greedy",
    "explore_var_spec": {
      "name": "linear_decay",
      "start_val": 1.0,
      "end_val": 0.01,
      "start_step": 0,
      "end_step": 50000
    },
    "gamma": 0.99,
    "training_batch_iter": 2,
    "training_iter": 2,
    "training_frequency": 4
  }
}
```

### SAC for MuJoCo (Continuous)

```javascript
{
  "algorithm": {
    "name": "SoftActorCritic",
    "gamma": 0.99,
    "training_frequency": 1,
    "training_iter": 1
  }
}
```

## Adding a New Algorithm

1. Create `slm_lab/agent/algorithm/your_algo.py`
2. Inherit from the appropriate base class
3. Override only the methods that differ
4. Register in `slm_lab/agent/algorithm/__init__.py`

**Example: Custom DQN Variant**

```python
from slm_lab.agent.algorithm.dqn import DQN

class MyDQN(DQN):
    def init_algorithm_params(self):
        super().init_algorithm_params()
        self.my_param = self.algorithm_spec.get('my_param', 0.5)

    def calc_q_loss(self, batch):
        loss = super().calc_q_loss(batch)
        return loss * self.my_param  # Example modification
```

See [Architecture](../architecture.md) for more on extending SLM Lab.

## Algorithm Performance Notes

Based on v5 benchmark results, here's guidance on algorithm selection:

### Recommended by Environment

| Environment Type | Best Algorithm | Notes |
|------------------|----------------|-------|
| **Classic Control** | PPO, A2C | Fast convergence, reliable |
| **Box2D Discrete** | DDQN+PER | Better than DQN, PPO close second |
| **Box2D Continuous** | SAC | Best for continuous LunarLander |
| **MuJoCo** | PPO | Robust across all 11 envs |
| **Atari** | PPO | Validated on 54 games |

### Known Limitations

These algorithm-environment combinations underperform:

| Algorithm | Environment | Issue | Alternative |
|-----------|-------------|-------|-------------|
| **DQN** | CartPole | Slow convergence (188 vs 499 PPO) | Use DDQN+PER or PPO |
| **A2C** | LunarLander | Fails discrete (9.5) and continuous (-38) | Use PPO or SAC |
| **A2C** | Pendulum | Poor performance (-553 vs -168 PPO) | Use PPO or SAC |
| **SAC** | Discrete envs | Mixed results, high variance | Use PPO or DDQN+PER |

{% hint style="info" %}
**SAC on MuJoCo:** Not included in v5 benchmarks due to compute requirements. Off-policy algorithms require significantly more resources for systematic benchmarking. Use PPO for validated MuJoCo results.
{% endhint %}

### Lambda Tuning for Atari

Different games benefit from different GAE lambda values:

| Lambda | Best For | Examples |
|--------|----------|----------|
| 0.95 | Strategic games | Qbert, BeamRider, Seaquest |
| 0.85 | Mixed games | Pong, MsPacman, Enduro |
| 0.70 | Action games | Breakout, KungFuMaster |

See [Atari Benchmark](../../benchmark-results/atari-benchmark.md) for per-game results.

## Learning Resources

For deep dives into these algorithms:
- [Deep RL Resources](../../resources/untitled.md) - Recommended papers and courses
- [Foundations of Deep RL](../../publications-and-talks/instruction-for-the-book-+-intro-to-rl-section.md) - The companion book
- [Algorithm Taxonomy](../modular-lab-components/algorithm-taxonomy.md) - Visual overview
