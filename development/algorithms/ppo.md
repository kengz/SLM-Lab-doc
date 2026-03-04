# PPO

## **Proximal Policy Optimization**

PPO ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)) extends Actor-Critic (A2C) by constraining policy updates to avoid destructively large steps. It replaces the standard policy gradient loss with a clipped surrogate objective that prevents the new policy from deviating too far from the old one.

PPO is on-policy: it collects a batch of experience with the current policy, performs multiple gradient updates on that batch, then discards it and collects fresh data.

See [slm_lab/spec/benchmark_arc/ppo/](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/spec/benchmark_arc/ppo) for example PPO specs.

### **Algorithm: PPO**

$$
\begin{aligned}
& \text{For iteration = 1, 2, 3, ...} \\
& \quad \text{Collect T timesteps of experience using current policy } \pi_\theta \\
& \quad \text{Compute advantages } \hat{A}_1, ..., \hat{A}_T \text{ using GAE} \\
& \quad \text{For epoch = 1, ..., K:} \\
& \quad \quad \text{For each minibatch:} \\
& \quad \quad \quad \text{Compute ratio: } r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \\
& \quad \quad \quad \text{Clipped objective: } L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t \hat{A}_t,\ \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right] \\
& \quad \quad \quad \text{Total loss: } L = -L^{CLIP} + c_1 L^{VF} - c_2 S[\pi_\theta] \\
& \quad \quad \quad \text{Update } \theta \text{ via gradient descent on } L
\end{aligned}
$$

The clip parameter $$\epsilon$$ limits how much the policy can change per update, preventing performance collapse.

### **Basic Parameters**

```python
"agent": {
  "name": str,
  "algorithm": {
    "name": "PPO",
    "action_pdtype": str,
    "action_policy": str,
    "gamma": float,
    "lam": float,
    "clip_eps_spec": dict,
    "entropy_coef_spec": dict,
    "time_horizon": int,
    "minibatch_size": int,
    "training_epoch": int,
  },
  "memory": {
    "name": "OnPolicyBatchReplay",
  },
  "net": {
    "type": str,
    "arc": dict,
    "optim_spec": dict,
  }
}
```

* `algorithm`
  * `name`: `"PPO"`
  * `action_pdtype` [_general param_](./)
  * `action_policy` [_general param_](./)
  * `gamma` [_general param_](./)
  * `lam`: GAE lambda ∈ [0, 1]. Controls bias-variance tradeoff. 0 = pure TD (low variance), 1 = Monte Carlo (low bias). Typical: 0.95
  * `clip_eps_spec`: schedule for clipping parameter ε. Constrains how much the policy can change per update. Typical: 0.1–0.2
  * `entropy_coef_spec`: weight for entropy bonus to encourage exploration
  * `time_horizon`: steps to collect before each update (T). Typical: 128 (Atari), 2048 (MuJoCo)
  * `minibatch_size`: size of minibatches drawn from the collected batch. Typical: 64–256
  * `training_epoch`: how many passes through the collected data per update (K). Typical: 4–10
* `memory`
  * Compatible type: `"OnPolicyBatchReplay"` — stores one horizon of experience then discards
* `net`
  * `type` [_general param_](./). Use `"TorchArcNet"` with TorchArc YAML specs

### **PPO vs A2C**

PPO is essentially A2C with:
1. **Clipped objective** instead of standard policy gradient — prevents large policy updates
2. **Multiple epochs** over collected data — more sample efficiency from each rollout
3. **Minibatch updates** — divides the collected rollout into minibatches for gradient steps

A2C does one gradient step per collected batch; PPO does K epochs × (T/M) minibatch steps.

### **Lambda Tuning for Atari**

Different games respond to different GAE lambda values:

| Lambda | Best for | Examples |
|--------|----------|---------|
| 0.95 | Strategic games | Qbert, BeamRider, Seaquest |
| 0.85 | Mixed games | Pong, MsPacman, Enduro |
| 0.70 | Action games | Breakout, KungFuMaster |

See [Atari Benchmark](../../benchmark-results/atari-benchmark.md) for per-game best lambda values.
