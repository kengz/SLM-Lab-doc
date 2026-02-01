# Memory 💾

## Overview

Memory classes handle experience storage and retrieval for RL training. Every time an agent acts, it stores a transition `(state, action, reward, next_state, done)` in memory. When it's time to train, the algorithm samples from this memory.

**Code:** [slm\_lab/agent/memory](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/agent/memory)

## Memory Types

| Memory Type | Algorithm Type | Behavior | Use Case |
|-------------|----------------|----------|----------|
| [**Replay**](replay.md) | Off-policy (DQN, SAC) | Ring buffer, random sampling | Standard experience replay |
| [**PrioritizedReplay**](prioritizedreplay.md) | Off-policy (DDQN+PER) | Priority-based sampling | Learn more from surprising transitions |
| [**OnPolicyReplay**](onpolicyreplay.md) | On-policy (REINFORCE) | Flush after each episode | Episode-based training |
| [**OnPolicyBatchReplay**](onpolicybatchreplay.md) | On-policy (PPO, A2C) | Flush after N steps | Batch-based training |

## On-Policy vs Off-Policy

The key distinction in RL memory is whether the algorithm can learn from past experience:

**On-policy** (PPO, A2C, REINFORCE):
- Can only learn from data collected by the *current* policy
- Must discard data after each training update
- Requires fresh data for each update
- Uses `OnPolicyReplay` or `OnPolicyBatchReplay`

**Off-policy** (DQN, SAC):
- Can learn from data collected by *any* policy
- Stores data in a replay buffer for reuse
- More sample-efficient (reuses data multiple times)
- Uses `Replay` or `PrioritizedReplay`

## Memory Interface

All memory classes implement this interface:

```python
class Memory:
    def update(self, state, action, reward, next_state, done, terminated, truncated):
        """Store a transition."""
        pass

    def sample(self) -> dict:
        """Sample a batch for training. Returns dict with keys:
        - states: (batch_size, state_dim)
        - actions: (batch_size, action_dim)
        - rewards: (batch_size,)
        - next_states: (batch_size, state_dim)
        - dones: (batch_size,)
        - terminated: (batch_size,)  # v5: true episode end
        - truncated: (batch_size,)   # v5: time limit reached
        """
        pass

    def __len__(self) -> int:
        """Current number of stored transitions."""
        pass
```

## Training Signal

Memory controls when training happens via the `to_train` flag:

```python
# In the training loop
agent.memory.update(state, action, reward, next_state, done, terminated, truncated)

if agent.memory.to_train:
    loss = agent.algorithm.train()
    agent.memory.to_train = False  # Reset flag
```

Different memory types set `to_train = True` based on different conditions:

| Memory Type | Training Trigger |
|-------------|------------------|
| `Replay` | Every `training_frequency` steps |
| `PrioritizedReplay` | Every `training_frequency` steps |
| `OnPolicyReplay` | End of episode |
| `OnPolicyBatchReplay` | Every `training_frequency` steps |

## Memory Spec

Configure memory in the agent spec:

```javascript
{
  "agent": {
    "memory": {
      "name": "Replay",           // Memory class name
      "batch_size": 32,           // Samples per training batch
      "max_size": 10000,          // Maximum buffer capacity
      "use_cer": true             // Combined Experience Replay
    }
  }
}
```

## Choosing the Right Memory

| Algorithm | Memory | Why |
|-----------|--------|-----|
| PPO | `OnPolicyBatchReplay` | Collects fixed-size batches, trains multiple epochs |
| A2C | `OnPolicyBatchReplay` | Similar to PPO but single epoch |
| REINFORCE | `OnPolicyReplay` | Trains on complete episodes |
| DQN | `Replay` | Standard experience replay |
| DDQN+PER | `PrioritizedReplay` | Prioritize high-error transitions |
| SAC | `Replay` | Off-policy, high replay ratio |

## Advanced: Combined Experience Replay (CER)

CER ([de Bruin et al., 2018](https://arxiv.org/abs/1712.01275)) always includes the most recent transition in each batch:

```javascript
{
  "memory": {
    "name": "Replay",
    "use_cer": true  // Guarantees latest transition is sampled
  }
}
```

This helps with environments where recent experience is particularly valuable.

## Data Format

Memory stores transitions in numpy arrays for efficiency. When training, data is converted to PyTorch tensors:

```python
batch = memory.sample()
# batch = {
#     'states': tensor of shape (batch_size, *state_shape),
#     'actions': tensor of shape (batch_size, *action_shape),
#     'rewards': tensor of shape (batch_size,),
#     'next_states': tensor of shape (batch_size, *state_shape),
#     'dones': tensor of shape (batch_size,),
#     'terminated': tensor of shape (batch_size,),
#     'truncated': tensor of shape (batch_size,),
# }
```

The `terminated` vs `truncated` distinction (new in v5/Gymnasium) enables correct value bootstrapping—see [Changelog](../../CHANGELOG.md) for details.
