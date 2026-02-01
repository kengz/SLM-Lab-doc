# Architecture 🏛️

This page describes SLM-Lab's architecture and control flow.

## Control Hierarchy

SLM-Lab organizes training into a hierarchical structure:

```
CLI (slm-lab command)
 └── Experiment (hyperparameter search)
      └── Trial (one configuration)
           └── Session (one random seed)
                ├── Agent
                │    ├── Algorithm
                │    └── Memory
                └── Env
                     └── MetricsTracker
```

| Level | Purpose | Configured By |
|-------|---------|---------------|
| **Experiment** | Orchestrates hyperparameter search via Ray Tune | `meta.max_trial`, `search` block |
| **Trial** | Runs multiple sessions with one configuration | `meta.max_session` |
| **Session** | Single training run, owns Agent and Env | Random seed |

In most cases, you run a single Trial (which creates multiple Sessions). Experiments are used for hyperparameter tuning.

## Agent Components

The **Agent** is a container that wires together three components:

```
Agent
 ├── Algorithm (policy, training logic)
 │    ├── Networks (actor, critic, Q-function)
 │    └── Optimizers
 ├── Memory (experience storage)
 └── MetricsTracker (logging)
```

### Algorithm

Implements the RL algorithm: network architecture, action selection, and gradient updates.

**Class Hierarchy:**

```
Algorithm (base)
 ├── SARSA
 │    └── VanillaDQN → DQNBase → DQN → DoubleDQN
 └── Reinforce
      └── ActorCritic (A2C)
           ├── PPO
           └── SoftActorCritic (SAC)
```

Each algorithm extends its parent, adding only the differences:

| Algorithm | Parent | Key Difference |
|-----------|--------|----------------|
| VanillaDQN | SARSA | Neural network Q-function |
| DQNBase | VanillaDQN | Adds target network infrastructure |
| DQN | DQNBase | Periodic target updates |
| DoubleDQN | DQN | Uses online network for action selection |
| ActorCritic | Reinforce | Adds value function (critic), supports GAE and n-step |
| PPO | ActorCritic | Adds clipped surrogate objective, minibatch training |
| SAC | ActorCritic | Adds entropy regularization, twin Q-networks |

See [Class Inheritance: A2C > PPO](modular-lab-components/class-inheritance-a2c-greater-than-ppo.md) for a deep dive.

**Key Algorithm Methods:**

```python
class Algorithm:
    def init_algorithm_params(self):
        """Initialize hyperparameters from spec."""
        pass

    def init_nets(self, global_nets=None):
        """Create neural networks and optimizers."""
        pass

    def act(self, state) -> action:
        """Select action given state."""
        pass

    def train(self) -> loss:
        """Sample from memory and update networks."""
        pass

    def update(self) -> explore_var:
        """Update exploration parameters (epsilon, entropy)."""
        pass
```

### Memory

Stores and retrieves experience for training.

| Type | Algorithms | Behavior | Key Config |
|------|------------|----------|------------|
| **OnPolicyBatchReplay** | PPO, A2C | Fixed-size buffer, cleared after use | `training_frequency` |
| **Replay** | DQN, SAC | Ring buffer with random sampling | `batch_size`, `max_size` |
| **PrioritizedReplay** | DDQN+PER | Samples by TD-error priority | `alpha`, `epsilon` |

**Memory Interface:**

```python
class Memory:
    def update(self, state, action, reward, next_state, done, terminated, truncated):
        """Store a transition."""
        pass

    def sample(self) -> batch:
        """Sample a batch for training."""
        pass

    def __len__(self) -> int:
        """Current number of stored transitions."""
        pass
```

### Network

Neural network architectures configured in the `net` spec:

| Type | Use Case | Input | Architecture |
|------|----------|-------|--------------|
| **MLPNet** | Low-dimensional states | Vector | Fully-connected layers |
| **ConvNet** | Image observations | Images | CNN + FC layers |
| **RecurrentNet** | Partial observability | Sequences | LSTM/GRU + FC |

**Network Configuration Example:**

```javascript
"net": {
  "type": "MLPNet",
  "shared": false,              // Separate actor/critic networks
  "hid_layers": [256, 256],     // Two hidden layers of 256 units
  "hid_layers_activation": "tanh",
  "init_fn": "orthogonal_",     // Weight initialization
  "clip_grad_val": 0.5,         // Gradient clipping
  "optim_spec": {
    "name": "Adam",
    "lr": 3e-4
  },
  "lr_scheduler_spec": {        // Optional learning rate schedule
    "name": "LinearToZero",
    "frame": 1000000
  },
  "gpu": "auto"
}
```

## Training Loop

A Session runs this loop until `max_frame`:

```python
# Simplified training loop
state, info = env.reset()

while env.frame < env.max_frame:
    # 1. Select action from policy
    action = agent.act(state)

    # 2. Execute in environment
    next_state, reward, terminated, truncated, info = env.step(action)

    # 3. Store transition in memory
    agent.memory.update(state, action, reward, next_state, ...)

    # 4. Train networks (when ready)
    loss = agent.algorithm.train()

    # 5. Update exploration parameters
    agent.algorithm.update()

    # 6. Log metrics at checkpoints
    if frame % log_frequency == 0:
        tracker.log()

    # 7. Handle episode boundaries
    if terminated or truncated:
        state, info = env.reset()
    else:
        state = next_state
```

### Training Frequency

How often training happens depends on the algorithm:

| Algorithm Type | Training Trigger | Example |
|----------------|------------------|---------|
| **On-policy** (PPO, A2C) | Every `time_horizon` steps | PPO trains every 128 steps |
| **Off-policy** (DQN, SAC) | Every `training_frequency` steps | DQN trains every 4 steps |

## Environment

SLM-Lab uses [Gymnasium](https://gymnasium.farama.org/) environments with automatic vectorization:

### Environment Creation Flow

```
1. Parse spec["env"]["name"]
     ↓
2. Create base gymnasium environment
     ↓
3. Apply wrappers based on env type:
   - Atari: AtariPreprocessing, FrameStack
   - MuJoCo: NormalizeObservation, NormalizeReward (if enabled)
   - All: ClockWrapper for frame counting
     ↓
4. Vectorize with SyncVectorEnv (num_envs copies)
     ↓
5. Return wrapped vector environment
```

### Key Wrappers

| Wrapper | Purpose | Applied To |
|---------|---------|------------|
| `ClockWrapper` | Track frame/episode counts | All envs |
| `AtariPreprocessing` | Grayscale, resize, frame skip | ALE envs |
| `FrameStackObservation` | Stack N frames | ALE envs |
| `NormalizeObservation` | Running mean/std normalization | MuJoCo (optional) |
| `NormalizeReward` | Running reward normalization | MuJoCo (optional) |
| `TrackReward` | Track true episode rewards | ALE envs |

### Vectorized Environments

`num_envs` controls parallelization:

```javascript
"env": {
  "name": "CartPole-v1",
  "num_envs": 4,        // 4 parallel environments
  "max_frame": 200000   // Total frames across all envs
}
```

With `num_envs=4`:
- Each `env.step()` returns batched data: `(4, state_dim)`, `(4,)`, etc.
- Frame count increments by 4 per step
- Useful for on-policy algorithms that need diverse samples

## Spec System

JSON specs fully configure experiments:

```javascript
{
  "ppo_cartpole": {
    "agent": {
      "name": "PPO",
      "algorithm": { "name": "PPO", "gamma": 0.99, ... },
      "memory": { "name": "OnPolicyBatchReplay" },
      "net": { "type": "MLPNet", ... }
    },
    "env": { "name": "CartPole-v1", "num_envs": 4, ... },
    "meta": { "max_frame": 200000, "max_session": 4, ... }
  }
}
```

### Variable Substitution

Specs support `${var}` placeholders:

```javascript
"env": { "name": "${env}", "max_frame": "${max_frame}" }
```

Substitute at runtime:

```bash
slm-lab run -s env=Hopper-v5 -s max_frame=1e6 spec.json spec_name train
```

### Search Blocks

Define hyperparameter ranges for Ray Tune:

```javascript
"search": {
  "agent.algorithm.gamma__uniform": [0.95, 0.999],
  "agent.net.optim_spec.lr__loguniform": [1e-5, 1e-3]
}
```

## Extending SLM-Lab

### Adding a New Algorithm

1. Create `slm_lab/agent/algorithm/your_algo.py`
2. Inherit from appropriate base (`Algorithm`, `ActorCritic`, `DQN`, etc.)
3. Override necessary methods:
   - `init_algorithm_params()` - Set hyperparameter defaults
   - `init_nets()` - Create networks
   - `act()` - Action selection
   - `train()` - Training step
4. Register in `slm_lab/agent/algorithm/__init__.py`
5. Create a spec file for testing

**Example: Custom DQN Variant**

```python
from slm_lab.agent.algorithm.dqn import DQN

class MyDQN(DQN):
    def init_algorithm_params(self):
        super().init_algorithm_params()
        # Add custom parameters
        self.my_param = self.algorithm_spec.get('my_param', 0.5)

    def calc_q_loss(self, batch):
        # Custom loss calculation
        loss = super().calc_q_loss(batch)
        return loss * self.my_param  # Example modification
```

### Adding Environment Support

SLM-Lab works with any gymnasium-compatible environment:

```javascript
{"env": {"name": "YourEnv-v1"}}
```

For custom environments, ensure gymnasium API compliance:

```python
import gymnasium as gym

class MyEnv(gym.Env):
    def reset(self, seed=None, options=None):
        # Return (observation, info)
        return obs, {}

    def step(self, action):
        # Return (observation, reward, terminated, truncated, info)
        return obs, reward, terminated, truncated, {}
```

Register and use:

```python
gym.register(id='MyEnv-v1', entry_point='my_module:MyEnv')
```
