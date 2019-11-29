# Memory

## 🏗 Memory API

Code: [slm\_lab/agent/memory](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/agent/memory)

Memory is a class for data storage and access consistent with the RL agent API, i.e. implementing `update` and `sample` methods. The underlying data format is numpy, which can efficiently be put into PyTorch tensors using shared memory via `torch.from_numpy`. There are two types of memory in RL:

For off-policy algorithms

* **Replay**
* **PrioritizedReplay**

For on-policy algorithms:

* **OnPolicyReplay**
* **OnPolicyBatchReplay**

Reinforcement Learning \(RL\) agents learn by acting in environments. Each time an agent acts, it stores `<s, a, s', r>`, the state, action taken, next state, and reward received, as an experience in memory.

RL algorithms vary in the way they make use of past experiences, for example by sampling batches from the last N experiences the agent has had or by using all the experiences gathered since the agent was last trained.

{% hint style="info" %}
Since RL agent trains based on the time steps/experience collected, the signal for training comes from the memory class in `memory.to_train`. That is, the memory class sets `self.to_train = True` when it is time to train.
{% endhint %}

