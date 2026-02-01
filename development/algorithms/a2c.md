# Actor Critic

Actor-Critic algorithms combine value function and policy estimation. They consist of an actor, which learns a parameterized policy, $$\pi$$, mapping states to probability distributions over actions, and a critic, which learns a parameterized function which assigns a value to actions taken.

There are a variety of approaches to training the critic. Three options are provided with the baseline algorithms.

* Learn the V function and use it to approximate the Q function.
* Advantage with n-step forward returns from [Mnih et. al. 2016](https://arxiv.org/abs/1602.01783).
* Generalized advantage estimation from [Schulman et. al, 2015](https://arxiv.org/abs/1506.02438).

See [slm_lab/spec/benchmark/a2c/](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/spec/benchmark/a2c) for example A2C specs.

The actor and critic can be trained separately or jointly, depending on whether the networks are structured to share parameters.

The critic is trained using temporal difference learning, similarly to the [DQN](dqn.md) algorithm. The actor is trained using policy gradients, similarly to [REINFORCE](reinforce.md). However in this case, the estimate of the "goodness" of actions is evaluated using the advantage, $$A^{\pi}$$, which is calculated using the critic's estimation of $$V$$. $$A^{\pi}$$ measures how much better the action taken in states `s` was compared with expected value of being in that state `s`, i.e. the expectation over all possible actions in state `s`.

Actor-Critic algorithms are on policy. Only data that is gathered using the current policy can be used to update the parameters. Once the policy parameters have been updated all previous data gathered must be discarded and the collection process started again with the new policy.

**Algorithm: Simple Actor Critic with separate actor and critic parameters**

$$
\begin{aligned}
&\text{For i = 1 .... N:} \\
&\quad \text{1. Gather data } {(s_i, a_i, r_i, s'_i)} \ \text{by acting in the environment using your policy} \\
&\quad \text{2. Update V} \\
&\quad \quad \quad \text{for j = 1 ... K:} \\
&\quad \quad \quad \quad \text{Calculate target values, } ~y_i~ \text{for each example} \\
&\quad \quad \quad \quad \text{Update critic network parameters, using a regression loss, e.g. MSE} \\
&\quad \quad \quad \quad \quad \quad L(\theta) = \frac{1}{2} \sum_i || (y_i - V(s_i; \theta)) ||^2 \\
&\quad \text{3. Calculate advantage, } ~A^{\pi}~ \text{, for each example, using value function, } V \\
&\quad \text{4. Calculate policy gradient} \\
& \quad \quad \quad \quad \nabla_{\phi}J(\phi) \approx \sum_i A^\pi_t(s_i, a_i) \nabla_\phi log \pi_\phi (a_i | s_i)\\
&\quad \text{5. Use gradient to update actor parameters } \phi  \\
\end{aligned}
$$

See [slm_lab/spec/benchmark/a2c/](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/spec/benchmark/a2c) for example Actor-Critic specs.

**Basic Parameters**

```python
    "agent": {
      "name": str,
      "algorithm": {
        "name": str,
        "action_pdtype": str,
        "action_policy": str,
        "gamma": float,
        "lam": float,
        "entropy_coef_spec": {...},
        "training_frequency": int,
      },
      "memory": {
        "name": str,
      },
      "net": {
        "type": str,
        "shared": bool,
        "hid_layers": list,
        "hid_layers_activation": str,
        "optim_spec": dict,
      }
    },
    ...
}
```

* `algorithm`
  * `name` [_general param_](./)
  * `action_pdtype` [_general param_](./)
  * `action_policy` string specifying which policy to use to act. "Categorical" (for discrete action spaces), "Normal" (for continuous actions spaces with one dimension), or "default" to automatically switch between the two depending on the environment.
  * `gamma` [_general param_](./)
  * `lam` $$\in [0,1]$$ GAE lambda parameter. If set, uses Generalized Advantage Estimation. Trade-off between bias and variance: 0 = low variance/high bias, 1 = high variance/low bias. Typical value: 0.95-0.97.
  * `num_step_returns` if set (and `lam` is not), uses n-step returns instead of GAE. Number of forward steps for advantage estimation. **Note:** When using n-step returns, `training_frequency` is automatically set to `num_step_returns`.
  * `entropy_coef_spec` schedule for entropy coefficient added to the loss to encourage exploration. Example: `{"name": "no_decay", "start_val": 0.01, "end_val": 0.01, "start_step": 0, "end_step": 0}`
  * `training_frequency` when using episodic data storage (memory) such as "OnPolicyReplay", this means how many episodes of data to collect before each training iteration - a common value is 1; or when using batch data storage (memory) such as "OnPolicyBatchReplay", how often to train the algorithm. Value of 32 means train every 32 steps the agent takes in the environment using the 32 examples since the agent was previously trained.
* `memory`
  * `name` [_general param_](./). Compatible types; ["OnPolicyReplay", "OnPolicyBatchReplay"](../memory/)
* `net`
  * `type` [_general param_](./). All [networks](../neural-networks/) are compatible.
  * `shared` whether the actor and the critic should share parameters
  * `hid_layers` [_general param_](./)
  * `hid_layers_activation` [_general param_](./)
  * `actor_optim_spec` [_general param_](./) optimizer for the actor
  * `critic_optim_spec` [_general param_](./) optimizer for the critic

**Advanced Parameters**

```python
    "agent": {
      "algorithm": {
        "training_epoch": int,
        "val_loss_coef": float
      },
      "net": {
        "rnn_hidden_size": int,
        "rnn_num_layers": int,
        "seq_len": int,
        "clip_grad_val": float,
        "lr_scheduler_spec": dict,
        "gpu": str
      }
    },
    ...
}
```

* `algorithm`
  * `training_epoch` how many gradient steps to take when training the critic. Only applies when the actor and critic have separate parameters.
  * `policy_loss_coef` how much weight to give to the policy (actor) component of the loss when the actor and critic have shared parameters, so are trained jointly.
  * `val_loss_coef` how much weight to give to the critic component of the loss when the actor and critic have shared parameters, so are trained jointly.
  * `normalize_v_targets` normalize value targets to prevent gradient explosion. Uses running statistics normalization (like SB3's VecNormalize).
* `net`
  * `use_same_optim` whether to use the same optimizer for both actor and critic. Useful for parameter search.
  * `rnn_hidden_size` [_general param_](./)
  * `rnn_num_layers` [_general param_](./)
  * `seq_len` [_general param_](./)
  * `clip_grad_val`: [_general param_](./)
  * `lr_scheduler_spec`: optional learning rate scheduler config
  * `gpu`: [_general param_](./)

## PPO (Proximal Policy Optimization)

PPO extends Actor-Critic with clipped surrogate objective and minibatch updates. See [slm_lab/spec/benchmark/ppo/](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/spec/benchmark/ppo) for example PPO specs.

**PPO-Specific Parameters**

```python
    "agent": {
      "algorithm": {
        "name": "PPO",
        "clip_eps_spec": {...},  # Clipping parameter schedule
        "minibatch_size": int,   # Minibatch size for updates
        "time_horizon": int,     # Steps per actor before update
        "training_epoch": int,   # Epochs per update
        "normalize_v_targets": bool,  # v5: Normalize value targets
        "clip_vloss": bool            # v5: CleanRL-style value clipping
      },
      ...
    },
}
```

* `clip_eps_spec` PPO clipping parameter, typically starts at 0.2. Can use a schedule for decay.
* `minibatch_size` number of samples per minibatch during training
* `time_horizon` number of environment steps collected before each training update (training_frequency = num_envs × time_horizon)
* `training_epoch` number of passes through collected data per update
* `normalize_v_targets` (v5) normalize value targets using running statistics to prevent gradient explosion with varying reward scales
* `clip_vloss` (v5) CleanRL-style value loss clipping—clips value predictions relative to old predictions using `clip_eps`. Improves stability for some environments.

{% hint style="info" %}
**MuJoCo**: Use `normalize_v_targets: true` for continuous control tasks.
**Atari**: Use `clip_vloss: true` for image-based tasks.
{% endhint %}
