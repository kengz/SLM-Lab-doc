# REINFORCE

REINFORCE [Williams, 1992](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf) directly learns a parameterized policy, $$\pi$$, which maps states to probability distributions over actions.

Starting with random parameter values, the agent uses this policy to act in an environment and receive rewards. After an episode has finished, the "goodness" of each action, represented by, $$f(\tau)$$, is calculated using the episode trajectory. The parameters of the policy are then updated in a direction which makes good actions $$(f(\tau) > 0)$$ more likely, and bad actions $$(f(\tau) < 0)$$ less likely. Good actions are reinforced, bad actions are discouraged.

The agent then uses the updated policy to act in the environment, and the training process repeats.

REINFORCE is an on policy algorithm. Only data that is gathered using the current policy can be used to update the parameters. Once the policy parameters have been updated all previous data gathered must be discarded and the collection process started again with the new policy.

There are a number of different approaches to calculating $$f(\tau)$$. Method 3, outlined below, is common. It captures the idea that the absolute quality of the actions matters less than their quality relative to some baseline. One option for a baseline is the average of $$f(\tau)$$ over the training data \(typically one episode trajectory\).

**Algorithm: REINFORCE with baseline**

$$
\begin{aligned}
& \text{Initialize weights } \theta \text{, learning rate } \alpha \\
& \text{for each episode (trajectory) } \tau = \{s_0, a_0, r_0, s_1, \cdots, r_T\} \sim \pi_\theta \\
& \quad \text{for } t = 0 \text{ to } T \text{ do} \\
& \quad \quad \theta \leftarrow \theta + \alpha \ f(\tau)_t \nabla_\theta log ~ \pi_\theta(a_t|s_t) \\
& \quad \text{end for} \\
& \text{end for} \\
\end{aligned}
$$

Methods for calculating $$f(\tau)_t$$:

$$
\begin{aligned}
& \text{Given } \nabla_\theta J(\theta) \ \approx \sum_{t \geq 0} f(\tau) \nabla_\theta log \pi_\theta(a_t|s_t), ~ \text{improve baseline with: }\\
& \quad \quad 1.\ \text{reward as weightage } f(\tau) = \sum\limits_{t' \geq t} r_{t'} \\
& \quad \quad 2.\ \text{add discount factor } f(\tau) = \sum\limits_{t' \geq t} \gamma^{t'-t} r_{t'} \\
& \quad \quad 3.\ \text{introduce baseline } f(\tau) = \sum\limits_{t' \geq t} \gamma^{t'-t} r_{t'} - b(s_t) \\

\end{aligned}
$$

See [reinforce.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/reinforce.json) for example specs of variations of the REINFORCE algorithm.

**Basic Parameters**

```python
    "agent": [{
      "name": str,
      "algorithm": {
        "name": str,
        "action_pdtype": str,
        "action_policy": str,
        "gamma": float,
        "training_frequency": int,
        "add_entropy": bool,
        "entropy_coef": float,
      },
      "memory": {
        "name": str,
        "max_size": int
        "batch_size": int
      },
      "net": {
        "type": str,
        "hid_layers": list,
        "hid_layers_activation": str,
        "optim_spec": dict,
      }
    }],
    ...
}
```

* `algorithm`
  * `name` [_general param_](https://kengz.gitbooks.io/slm-lab/content/algorithms.html)
  * `action_pdtype` [_general param_](https://kengz.gitbooks.io/slm-lab/content/algorithms.html)
  * `action_policy` string specifying which policy to use to act. For example, "Categorical" \(for discrete action spaces\), "Normal" \(for continuous actions spaces with one dimension\), or "default" to automatically switch between the two depending on the environment.
  * `gamma` [_general param_](https://kengz.gitbooks.io/slm-lab/content/algorithms.html)
  * `training_frequency` how many episodes of data to collect before each training iteration. A common value is 1.
  * `entropy` whether to add entropy to the $$f(\tau)_t$$ to encourage exploration
  * `entropy_coef` coefficient to multiply the entropy of the distribution with when adding it to $$f(\tau)_t$$
* `memory`
  * `name` [_general param_](./). Compatible types; ["OnPolicyReplay", "OnPolicyBatchReplay"](../memory/)
  * `batch_size` number of examples to collect before training. Only relevant for batch on policy memory: "OnPolicyBatchReplay"
* `net`
  * `type` [_general param_](./). Compatible types; [all networks](../neural-networks/).
  * `hid_layers` [_general param_](./)
  * `hid_layers_activation` [_general param_](./)
  * `optim_spec` [_general param_](./)

**Advanced Parameters**

```python
    "agent": [{
      "net": {
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

* `net`
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

