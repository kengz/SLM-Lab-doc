# CrossQ

## **CrossQ: Batch Normalization in Deep RL**

CrossQ ([Bhatt et al., ICLR 2024](https://arxiv.org/abs/1902.05605)) extends SAC by eliminating target networks through cross batch normalization in the critics. This reduces gradient steps by 20x while maintaining competitive performance.

**Key idea**: SAC uses a target network to stabilize Q-value bootstrapping. CrossQ replaces this with Batch Renormalization in the critics and a cross batch forward pass — current states `(s, a)` and next states `(s', a')` are concatenated into a single batch, so they share BatchNorm statistics. The Q-next values extracted from this single forward pass are stable enough to serve as targets without a separate target network.

### **Algorithm: CrossQ**

$$
\begin{aligned}
& \text{For k = 1 .... N:} \\
& \quad \text{Sample batch } \{(s_i, a_i, r_i, s'_i)\} \text{ from replay buffer} \\
& \quad \text{Sample next actions: } a'_i \sim \pi_\phi(s'_i) \\
& \quad \text{Cross forward pass through each critic:} \\
& \quad \quad \text{Input: } [(s_i, a_i); (s'_i, a'_i)] \text{ (concatenated batch)} \\
& \quad \quad \text{Split output: } Q_\theta(s, a), Q_\theta(s', a') \text{ (shared BN stats)} \\
& \quad \text{Compute targets: } y_i = r_i + \gamma \left( \min_{j} Q_{\theta_j}(s'_i, a'_i) - \alpha \log \pi_\phi(a'_i | s'_i) \right) \\
& \quad \text{Update critics: } L(\theta) = \frac{1}{2} \sum_i \| y_i - Q_\theta(s_i, a_i) \|^2 \\
& \quad \text{Update actor via reparameterization (same as SAC)} \\
& \quad \text{Update entropy temperature } \alpha \text{ (same as SAC)}
\end{aligned}
$$

**No target network update step** — cross batch normalization makes Q-next estimates stable without separate target parameters.

See [slm_lab/spec/benchmark/crossq/](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/spec/benchmark/crossq) for example CrossQ specs.

### **Basic Parameters**

```python
"agent": {
  "name": str,
  "algorithm": {
    "name": "CrossQ",
    "action_pdtype": str,
    "action_policy": "default",
    "gamma": float,
    "training_frequency": int,
    "training_iter": int,
    "training_start_step": int,
  },
  "memory": {
    "name": "Replay",
    "batch_size": int,
    "max_size": int
  },
  "net": {
    "type": "TorchArcNet",
    "arc": dict,            // actor architecture (no BN needed)
    "optim_spec": dict,
  },
  "critic_net": {
    "type": "TorchArcNet",
    "arc": dict,            // critic architecture with LazyBatchRenorm1d
    "optim_spec": dict,
  }
}
```

* `algorithm`
  * `name`: `"CrossQ"`
  * `action_pdtype`: `"Normal"` for continuous, `"Categorical"` for discrete
  * `action_policy`: `"default"`
  * `gamma` [_general param_](./)
  * `training_frequency`: how often to train (steps between updates)
  * `training_iter`: gradient steps per update. CrossQ uses UTD=1 (`training_iter=1`) for Classic Control and UTD=4 (`training_iter=4`) for MuJoCo — far fewer than SAC's UTD=4-20
  * `training_start_step`: steps before training begins (fill replay buffer first)
* `memory`
  * Compatible types: `"Replay"`, `"PrioritizedReplay"` (see [Memory](../memory/))
  * `batch_size`: examples per training batch
  * `max_size`: replay buffer capacity
* `net`: actor network — standard MLP, no BatchNorm needed
* `critic_net`: critic network — uses `LazyBatchRenorm1d` layers between linear layers

### **Critic Architecture: Batch Renormalization**

The critic must use Batch Renormalization ([Ioffe, 2017](https://arxiv.org/abs/1702.03275)), not standard BatchNorm. Standard BN has high variance at small batch sizes. BRN adds running-stats correction terms `r` and `d` that clip variance, making small-batch BN stable:

```yaml
# Critic with BatchRenorm (required for CrossQ)
_critic_brn: &critic_brn
  modules:
    body:
      Sequential:
        - LazyLinear:
            out_features: 256
        - LazyBatchRenorm1d:
            momentum: 0.01
            eps: 0.001
            warmup_steps: 5000   # steps before r/d clipping activates
        - ReLU:
        - LazyLinear:
            out_features: 256
        - LazyBatchRenorm1d:
            momentum: 0.01
            eps: 0.001
            warmup_steps: 5000
        - ReLU:
```

`warmup_steps` controls when BRN correction activates — during warmup, BRN behaves like standard BN to initialize running stats.

### **Comparison with SAC**

| Feature | SAC | CrossQ |
|---------|-----|--------|
| Target networks | Yes (2 Q-targets) | **No** |
| Critic architecture | Plain MLP | MLP + Batch Renorm |
| UTD ratio | 4–20 | **1–4** |
| Training speed | Baseline | **2–7x faster** |
| Performance | Strong | Competitive |

CrossQ's main advantage is wall-clock speed: by eliminating target network copies and using UTD=1, it trains 2-7x faster than SAC with similar final performance on MuJoCo tasks.

{% hint style="warning" %}
**Atari limitation**: CrossQ is experimental on Atari. Cross-batch BN is less effective with temporally correlated image frames (consecutive frames are nearly identical). SAC and PPO outperform CrossQ on most Atari games.
{% endhint %}
