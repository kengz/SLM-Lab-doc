# Actor Critic

Actor-Critic algorithms combine value function and policy estimation. They consist of an actor, which learns a parameterized policy, $$\pi$$, mapping states to probability distributions over actions, and a critic, which learns a parameterized function which assigns a value to actions taken.

There are a variety of approaches to training the critic. Three options are provided with the baseline algorithms.

* Learn the V function and use it to approximate the Q function. See [ac.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/ac.json) for some examples specs.
* Advantage with n-step forward returns from [Mnih et. al. 2016](https://arxiv.org/abs/1602.01783). See [a2c.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/a2c.json) for some example specs.
* Generalized advantage estimation from [Schulman et. al, 2015](https://arxiv.org/abs/1506.02438). See [a2c.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/a2c.json) for some example specs.

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

See [ac.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/a2c.json) and [a2c.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/a2c.json) for example specs of variations of the Actor-Critic algorithm.

**Basic Parameters**

```python
    "agent": [{
      "name": str,
      "algorithm": {
        "name": str,
        "action_pdtype": str,
        "action_policy": str,
        "gamma": float,
        "use_gae": bool,
        "lam": float,
        "use_nstep": bool,
        "num_step_returns": int,
        "add_entropy": bool,
        "entropy_coef": float,
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
        "actor_optim_spec": dict,
        "critic_optim_spec": dict,
      }
    }],
    ...
}
```

* `algorithm`
  * `name` [_general param_](./)
  * `action_pdtype` [_general param_](./)
  * `action_policy` string specifying which policy to use to act. "Categorical" \(for discrete action spaces\), "Normal" \(for continuous actions spaces with one dimension\), or "default" to automatically switch between the two depending on the environment.
  * `gamma` [_general param_](./)
  * `use_gae` whether to calculate the advantage using generalized advantage estimation.
  * `lam` $$\in [0,1]$$ trade off between bias and variance when using generalized advantage estimation. 0 corresponds to low variance and high bias. 1 corresponds to high variance and low bias.
  * `use_nstep`: whether to calculate the advantage using n-step forward returns. If `use_gae` is `true` this parameter will be ignored.
  * `num_step_returns` number of forward steps to use when calculating the target for advantage estimation using nstep forward returns.
  * `add_entropy` whether to add entropy to the advantage to encourage exploration
  * `entropy_coef` coefficient to multiply the entropy of the distribution with when adding it to advantage
  * `training_frequency` when using episodic data storage \(memory\) such as "OnPolicyReplay", this means how many episodes of data to collect before each training iteration - a common value is 1; or when using batch data storage \(memory\) such as "OnPolicyBatchReplay", how often to train the algorithm. Value of 32 means train every 32 steps the agent takes in the environment using the 32 examples since the agent was previously trained.
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
    "agent": [{
    "algorithm" : {
        "training_epoch": int,
        "policy_loss_coef": float,
        "val_loss_coef": float
      }
      "net": {
        "use_same_optim": bool,
        "rnn_hidden_size": int,
        "rnn_num_layers": int,
        "seq_len": int,
        "clip_grad": bool,
        "clip_grad_val": float,
        "lr_decay": str,
        "lr_decay_frequency": int,
        "lr_decay_min_timestep": int,
        "lr_anneal_timestep": int,
        "gpu": int
      }
    }],
    ...
}
```

* `algorithm`
  * `training_epoch` how many gradient steps to take when training the critic. Only applies when the actor and critic have separate parameters.
  * `policy_loss_coef` how much weight to give to the policy \(actor\) component of the loss when the actor and critic have shared parameters, so are trained jointly.
  * `val_loss_coef` how much weight to give to the critic component of the loss when the actor and critic have shared parameters, so are trained jointly.
* `net`
  * `use_same_optim` whether to use the `optim_actor` for both the actor and critic. This can be useful when using conducting a parameter search.
  * `rnn_hidden_size` [_general param_](./)
  * `rnn_num_layers` [_general param_](./)
  * `seq_len` [_general param_](./)
  * `clip_grad`: [_general param_](./)
  * `clip_grad_val`: [_general param_](./)
  * `lr_decay`: [_general param_](./)
  * `lr_decay_frequency`: [_general param_](./)
  * `lr_decay_min_timestep`: [_general param_](./)
  * `lr_anneal_timestep`: [_general param_](./)
  * `gpu`: [_general param_](./)

