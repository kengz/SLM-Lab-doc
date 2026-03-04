# SAC

## **Soft Actor-Critic**

SAC ([Haarnoja et al., 2018](https://arxiv.org/abs/1801.01290)) is an off-policy Actor-Critic algorithm that maximizes a trade-off between expected reward and entropy — encouraging the policy to be as random as possible while still performing well. This leads to better exploration and more robust policies.

SAC supports both continuous actions (reparameterization trick) and discrete actions (exact expectation, [Christodoulou, 2019](https://arxiv.org/abs/1910.07207)).

See [slm_lab/spec/benchmark/sac/](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/spec/benchmark/sac) and [slm_lab/spec/benchmark_arc/sac/](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/spec/benchmark_arc/sac) for example SAC specs.

### **Algorithm: SAC**

$$
\begin{aligned}
& \text{For k = 1 .... N:} \\
& \quad \text{Sample batch } \{(s_i, a_i, r_i, s'_i)\} \text{ from replay buffer} \\
& \quad \text{Compute soft targets: } y_i = r_i + \gamma\left(\min_{j=1,2} Q_{\bar\theta_j}(s'_i, a'_i) - \alpha \log\pi_\phi(a'_i | s'_i)\right) \\
& \quad \text{Update critics: } J(\theta) = \frac{1}{2}\sum_i \left(Q_\theta(s_i, a_i) - y_i\right)^2 \\
& \quad \text{Update actor: } J(\phi) = \mathbb{E}_{s,a\sim\pi}\left[\alpha\log\pi_\phi(a|s) - \min_j Q_{\theta_j}(s, a)\right] \\
& \quad \text{Update temperature: } J(\alpha) = \mathbb{E}\left[-\alpha\left(\log\pi_\phi(a|s) + \mathcal{H}_\text{target}\right)\right] \\
& \quad \text{Soft update target networks: } \bar\theta \leftarrow \tau\theta + (1-\tau)\bar\theta
\end{aligned}
$$

The temperature parameter $$\alpha$$ is automatically tuned to maintain a target entropy $$\mathcal{H}_\text{target}$$.

### **Basic Parameters**

```python
"agent": {
  "name": str,
  "algorithm": {
    "name": "SoftActorCritic",
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
    "type": str,
    "arc": dict,
    "optim_spec": dict,
    "polyak_coef": float,
  }
}
```

* `algorithm`
  * `name`: `"SoftActorCritic"`
  * `action_pdtype`: `"Normal"` for continuous, `"Categorical"` for discrete
  * `action_policy`: `"default"`
  * `gamma` [_general param_](./)
  * `training_frequency`: steps between updates. 1 = update every step
  * `training_iter`: gradient steps per update (UTD ratio). Typical: 1–4
  * `training_start_step`: steps before training begins. Fills replay buffer first
* `memory`
  * Compatible types: `"Replay"`, `"PrioritizedReplay"` (see [Memory](../memory/))
  * `batch_size`: examples per training batch. Typical: 256
  * `max_size`: replay buffer capacity. Typical: 1e6
* `net`
  * `polyak_coef`: soft update coefficient for target networks. τ = 1 - polyak_coef. Typical: 0.995 (τ = 0.005)

### **Advanced Parameters**

```python
"algorithm": {
  "policy_delay": int,         # update actor every N critic updates (default 1)
  "fixed_alpha": float,        # disable auto-tuning, use fixed temperature (e.g. 0.02)
  "alpha_lr": float,           # separate learning rate for temperature optimizer
  "spectral_norm": bool,       # spectral norm on penultimate critic layer
  "symlog": bool,              # symlog Q-value compression (DreamerV3)
}
```

* `policy_delay`: update actor every N critic updates. 2 = TD3-style delayed actor updates
* `fixed_alpha`: disable automatic entropy tuning and use a fixed temperature. Set to e.g. `0.02` for Atari
* `spectral_norm`: apply spectral normalization to penultimate critic layer for stability

### **SAC vs PPO**

| | SAC | PPO |
|--|-----|-----|
| **Type** | Off-policy | On-policy |
| **Data reuse** | Replay buffer | Discards after update |
| **Sample efficiency** | High | Moderate |
| **Hyperparameter sensitivity** | Low | Moderate |
| **Best for** | Continuous control, sample efficiency | Discrete, Atari, stable training |

SAC is the recommended algorithm for continuous control (MuJoCo, LunarLanderContinuous). For discrete environments, PPO is generally more reliable. See [CrossQ](crossq.md) for a faster SAC variant.
